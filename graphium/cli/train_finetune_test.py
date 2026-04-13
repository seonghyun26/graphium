from typing import List, Literal, Union
import csv
import os
import time
import timeit
from datetime import datetime
from pathlib import Path

import fsspec
import hydra
import numpy as np
import torch
import wandb
import yaml
from datamol.utils import fs
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from lightning.pytorch.utilities.model_summary import ModelSummary
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from graphium.config._loader import (
    load_accelerator,
    load_architecture,
    load_datamodule,
    load_metrics,
    load_predictor,
    load_trainer,
    save_params_to_wandb,
    get_checkpoint_path,
)
from graphium.finetuning import (
    FINETUNING_CONFIG_KEY,
    GraphFinetuning,
    modify_cfg_for_finetuning,
)
from graphium.finetuning.utils import filter_cfg_based_on_admet_benchmark_name
from graphium.hyper_param_search import (
    HYPER_PARAM_SEARCH_CONFIG_KEY,
    extract_main_metric_for_hparam_search,
)
from graphium.trainer.predictor import PredictorModule
from graphium.utils.load_pretrained_backbone import load_pretrained_backbone
from graphium.utils.safe_run import SafeRun

import graphium.cli.finetune_utils

TESTING_ONLY_CONFIG_KEY = "testing_only"
CONTINUAL_PRETRAINING_CONFIG_KEY = "continual_pretraining"


def _extract_pretrain_dataset(tags) -> str:
    """Extract pretrain dataset name from W&B tags list.

    Tags follow the pattern: ['model', 'finetune', 'admet', 'pretrain_dataset', ...]
    The pretrain dataset is the tag that is not a known keyword.
    """
    if not tags or not isinstance(tags, (list, tuple)):
        return None
    skip = {"finetune", "scratch", "admet", "moe", "pretrain", "arch_comparison",
            "speed_benchmark", "scratch_benchmark"}
    # First tag is model name; skip it and known keywords
    candidates = [t for t in tags[1:] if t not in skip]
    return candidates[0] if candidates else None


def _save_results_csv(results: dict, cfg: dict, output_dir: str) -> None:
    """Append a single row of results to a centralized CSV in the results/ directory.

    The CSV is designed to be easily loaded by pandas for visualization in notebooks.
    Each row = one training run. Columns = metadata + all metrics.
    """
    results_dir = cfg.get("constants", {}).get("results_csv_dir", None)
    if results_dir is None:
        # Default: results/ directory next to the graphium package root
        results_dir = os.path.join(os.getcwd(), "results")

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    csv_path = os.path.join(results_dir, "experiment_results.csv")

    # Collect metadata from config
    finetuning_cfg = cfg.get("finetuning", {})
    constants = cfg.get("constants", {})
    is_finetuning = "finetuning" in cfg

    # Extract pretrain dataset name from W&B tags (more reliable than checkpoint path)
    if is_finetuning:
        wandb_tags = constants.get("wandb", {}).get("tags", [])
        pretrain_dataset = _extract_pretrain_dataset(wandb_tags)
        if pretrain_dataset is None:
            pretrain_dataset = finetuning_cfg.get("pretrained_model", "unknown")
    else:
        pretrain_dataset = "scratch"

    row = {
        "timestamp": datetime.now().isoformat(),
        "model": cfg.get("architecture", {}).get("gnn", {}).get("layer_type", "unknown"),
        "task": constants.get("task", "multitask"),
        "seed": constants.get("seed", 0),
        "pretrain_dataset": pretrain_dataset,
        "is_finetuning": is_finetuning,
        "unfreeze_depth": finetuning_cfg.get("training_kwargs", {}).get("unfreeze_pretrained_depth", "N/A") if is_finetuning else "N/A",
        "epoch_unfreeze_all": finetuning_cfg.get("training_kwargs", {}).get("epoch_unfreeze_all", "N/A") if is_finetuning else "N/A",
        "hidden_dim": cfg.get("architecture", {}).get("gnn", {}).get("hidden_dims", "unknown"),
        "gnn_depth": cfg.get("architecture", {}).get("gnn", {}).get("depth", "unknown"),
        "output_dir": output_dir,
    }

    # Add W&B tags if present
    wandb_tags = constants.get("wandb", {})
    if isinstance(wandb_tags, dict):
        row["wandb_tags"] = str(wandb_tags.get("tags", ""))

    # Flatten all metrics into columns
    for key, value in sorted(results.items()):
        row[key] = value

    # Write header if file doesn't exist; expand columns if needed; then append
    file_exists = os.path.isfile(csv_path)

    if file_exists:
        with open(csv_path, "r", newline="") as rf:
            reader = csv.DictReader(rf)
            existing_cols = list(reader.fieldnames or [])
        new_cols = set(row.keys()) - set(existing_cols)
        if new_cols:
            all_fieldnames = existing_cols + sorted(new_cols)
            _rewrite_csv_header(csv_path, all_fieldnames)
        else:
            all_fieldnames = existing_cols
    else:
        all_fieldnames = sorted(row.keys())

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    logger.info(f"Results appended to {csv_path}")


def _rewrite_csv_header(csv_path: str, all_columns) -> None:
    """Rewrite the CSV file with an expanded header when new metric columns appear."""
    import shutil

    if isinstance(all_columns, set):
        all_columns = sorted(all_columns)
    tmp_path = csv_path + ".tmp"
    with open(csv_path, "r", newline="") as fin, open(tmp_path, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=all_columns, extrasaction="ignore")
        writer.writeheader()
        for existing_row in reader:
            writer.writerow(existing_row)
    shutil.move(tmp_path, csv_path)


@hydra.main(version_base=None, config_path="../../expts/hydra-configs", config_name="main")
def cli(cfg: DictConfig) -> None:
    """
    The main CLI endpoint for training, fine-tuning and evaluating Graphium models.
    """
    return run_training_finetuning_testing(cfg)


def get_replication_factor(cfg):
    try:
        ipu_config = cfg.get("accelerator", {}).get("ipu_config", [])
        for item in ipu_config:
            if "replicationFactor" in item:
                # Extract the number between parentheses
                start = item.find("(") + 1
                end = item.find(")")
                if start != 0 and end != -1:
                    return int(item[start:end])
    except Exception as e:
        print(f"An error occurred: {e}")

    # Return default value if replicationFactor is not found or an error occurred
    return 1


def get_gradient_accumulation_factor(cfg):
    """
    WARNING: This MUST be called after accelerator overrides have been applied
    (i.e. after `load_accelerator` has been called)
    """
    try:
        # Navigate through the nested dictionaries and get the gradient accumulation factor
        grad_accumulation_factor = cfg.get("trainer", {}).get("trainer", {}).get("accumulate_grad_batches", 1)

        # Ensure that the extracted value is an integer
        return int(grad_accumulation_factor)
    except Exception as e:
        print(f"An error occurred: {e}")

    # Return default value if an error occurred
    return 1


def get_training_batch_size(cfg):
    """
    WARNING: This MUST be called after accelerator overrides have been applied
    (i.e. after `load_accelerator` has been called)
    """
    try:
        # Navigate through the nested dictionaries and get the training batch size
        batch_size_training = cfg.get("datamodule", {}).get("args", {}).get("batch_size_training", 1)

        # Ensure that the extracted value is an integer
        return int(batch_size_training)
    except Exception as e:
        print(f"An error occurred: {e}")

    # Return default value if an error occurred
    return 1


def get_training_device_iterations(cfg):
    try:
        ipu_config = cfg.get("accelerator", {}).get("ipu_config", [])
        for item in ipu_config:
            if "deviceIterations" in item:
                # Extract the number between parentheses
                start = item.find("(") + 1
                end = item.find(")")
                if start != 0 and end != -1:
                    return int(item[start:end])
    except Exception as e:
        print(f"An error occurred: {e}")

    # Return default value if deviceIterations is not found or an error occurred
    return 1


def run_training_finetuning_testing(cfg: DictConfig) -> None:
    """
    The main (pre-)training and fine-tuning loop.
    """

    unresolved_cfg = OmegaConf.to_container(cfg, resolve=False)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    print(OmegaConf.to_yaml(cfg, resolve=True))


    # Get the current date and time
    now = datetime.now()
    # Format the datetime as a string
    filename_datetime_suffix = now.strftime("%Y%m%d_%H%M%S")
    # Append the datetime string to the existing filename in the cfg dictionary
    cfg["trainer"]["model_checkpoint"]["filename"] += f"_{filename_datetime_suffix}"
    cfg["trainer"]["model_checkpoint"]["dirpath"] = (
        cfg["trainer"]["model_checkpoint"]["dirpath"][:-1] + f"_{filename_datetime_suffix}"
    )

    dst_dir = cfg["constants"].get("results_dir")
    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg["runtime"]["output_dir"]

    if dst_dir is not None and fs.exists(dst_dir) and len(fs.get_mapper(dst_dir).fs.ls(dst_dir)) > 0:
        logger.warning(
            "The destination directory is not empty. "
            "If files already exist, this would lead to a crash at the end of training."
        )
        # We pause here briefly, to make sure the notification is seen as there's lots of logs afterwards
        time.sleep(5)
    # Modify the config for finetuning
    if FINETUNING_CONFIG_KEY in cfg:
        cfg = modify_cfg_for_finetuning(cfg)
    # Filter config for ADMET single-task training (without finetuning)
    elif (
        cfg.get("datamodule", {}).get("module_type") == "ADMETBenchmarkDataModule"
        and cfg.get("datamodule", {}).get("args", {}).get("tdc_benchmark_names") is not None
    ):
        cfg['metrics'] = {k: v for k, v in cfg["metrics"].items() if k == cfg['constants']['task']}
        cfg['architecture']['task_heads'] = {k: v for k, v in cfg["architecture"]["task_heads"].items() if k == cfg['constants']['task']}
        cfg['predictor']['loss_fun'] = {k: v for k, v in cfg["predictor"]["loss_fun"].items() if k == cfg['constants']['task']}
        cfg['predictor']['metrics_on_progress_bar'] = {k: v for k, v in cfg['predictor']['metrics_on_progress_bar'].items() if k == cfg['constants']['task']}

    st = timeit.default_timer()

    # Initialize wandb only on first rank
    if os.environ.get("RANK", "0") == "0":
        # Disable wandb if the user is not logged in.
        wandb_cfg = cfg["constants"].get("wandb")
        if wandb_cfg is not None and wandb.login() is False:
            logger.info(
                "Not logged in to wandb - disabling wandb logging.\n"
                + "To enable wandb, run `wandb login` from the command line."
            )
            wandb.init(mode="disabled")
        elif wandb_cfg is not None:
            wandb_cfg.pop("save_dir", None)
            wandb.init(config=cfg, **wandb_cfg)
    else:
        wandb_cfg = None

    ## == Instantiate all required objects from their respective configs ==
    # Accelerator
    cfg, accelerator_type = load_accelerator(cfg)

    ## Data-module
    datamodule = load_datamodule(cfg, accelerator_type)
    datamodule.prepare_data()

    testing_only = cfg.get(TESTING_ONLY_CONFIG_KEY, False)

    if testing_only:
        # Load pre-trained model
        predictor = PredictorModule.load_pretrained_model(
            name_or_path=get_checkpoint_path(cfg), device=accelerator_type
        )

    else:
        ## Architecture
        model_class, model_kwargs = load_architecture(cfg, in_dims=datamodule.in_dims)

        ## Metrics
        metrics = load_metrics(cfg)

        # Note: these MUST be called after `cfg, accelerator = load_accelerator(cfg)`
        replicas = get_replication_factor(cfg)
        gradient_acc = get_gradient_accumulation_factor(cfg)
        micro_bs = get_training_batch_size(cfg)
        device_iterations = get_training_device_iterations(cfg)

        global_bs = replicas * gradient_acc * micro_bs * device_iterations

        ## Predictor
        predictor = load_predictor(
            config=cfg,
            model_class=model_class,
            model_kwargs=model_kwargs,
            metrics=metrics,
            task_levels=datamodule.get_task_levels(),
            accelerator_type=accelerator_type,
            featurization=datamodule.featurization,
            task_norms=datamodule.task_norms,
            replicas=replicas,
            gradient_acc=gradient_acc,
            global_bs=global_bs,
        )

        # Continual pre-training: load backbone weights from a previous checkpoint
        # into the freshly instantiated multitask network. Task heads in the new
        # config that don't exist in the checkpoint stay randomly initialized.
        if CONTINUAL_PRETRAINING_CONFIG_KEY in cfg:
            if FINETUNING_CONFIG_KEY in cfg:
                raise ValueError(
                    "Configs 'continual_pretraining' and 'finetuning' are mutually "
                    "exclusive. Continual pre-training keeps the model as a plain "
                    "FullGraphMultiTaskNetwork; finetuning wraps it in "
                    "FullGraphFinetuningNetwork."
                )
            cpt_cfg = cfg[CONTINUAL_PRETRAINING_CONFIG_KEY]
            load_pretrained_backbone(
                predictor=predictor,
                checkpoint_path=cpt_cfg["pretrained_checkpoint"],
                strict=cpt_cfg.get("strict", False),
                exclude_prefixes=set(cpt_cfg.get("exclude_prefixes", []) or []),
            )

    logger.info(predictor.model)
    logger.info(ModelSummary(predictor, max_depth=4))

    ## Trainer
    date_time_suffix = datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
    trainer = load_trainer(cfg, accelerator_type, date_time_suffix)

    if not testing_only:
        # Add the fine-tuning callback to trainer
        if FINETUNING_CONFIG_KEY in cfg:
            finetuning_training_kwargs = cfg["finetuning"]["training_kwargs"]
            trainer.callbacks.append(GraphFinetuning(**finetuning_training_kwargs))

        if wandb_cfg is not None:
            save_params_to_wandb(trainer.logger, cfg, predictor, datamodule, unresolved_config=unresolved_cfg)

        # Determine the max num nodes and edges in training and validation
        logger.info("Computing the maximum number of nodes and edges per graph")
        predictor.set_max_nodes_edges_per_graph(datamodule, stages=["train", "val"])

        # When resuming training from a checkpoint, we need to provide the path to the checkpoint in the config
        resume_ckpt_path = cfg["trainer"].get("resume_from_checkpoint", None)

        # Run the model training
        # torch.autograd.set_detect_anomaly(True)
        with SafeRun(name="TRAINING", raise_error=cfg["constants"]["raise_train_error"], verbose=True):
            trainer.fit(model=predictor, datamodule=datamodule, ckpt_path=resume_ckpt_path)

        # Save validation metrics - Base utility in case someone doesn't use a logger.
        results = trainer.callback_metrics
        results = {k: v.item() if torch.is_tensor(v) else v for k, v in results.items()}
        with fsspec.open(fs.join(output_dir, "val_results.yaml"), "w") as f:
            yaml.dump(results, f)

    # Determine the max num nodes and edges in testing
    predictor.set_max_nodes_edges_per_graph(datamodule, stages=["test"])

    # When checkpoints are logged during training, we can, e.g., use the best or last checkpoint for testing
    test_ckpt_path = None
    test_ckpt_name = cfg["trainer"].get("test_from_checkpoint", None)
    test_ckpt_dir = cfg["trainer"]["model_checkpoint"].get("dirpath", None)
    if test_ckpt_name is not None and test_ckpt_dir is not None:
        test_ckpt_path = os.path.join(test_ckpt_dir, test_ckpt_name)

    # Run the model testing
    with SafeRun(name="TESTING", raise_error=cfg["constants"]["raise_train_error"], verbose=True):
        trainer.test(model=predictor, datamodule=datamodule, ckpt_path=test_ckpt_path)

    logger.info("-" * 50)
    logger.info("Total compute time:", timeit.default_timer() - st)
    logger.info("-" * 50)

    save_checkpoint_to_wandb = cfg["trainer"].get("save_checkpoint_to_wandb")
    if save_checkpoint_to_wandb is True:
        # Save initial model state - and upload checkpoint to wandb
        if cfg["trainer"]["model_checkpoint"]["save_last"] is True:
            checkpoint_path = f"{cfg['trainer']['model_checkpoint']['dirpath']}/{cfg['trainer']['model_checkpoint']['filename']}-v1.ckpt"
            # Log the initial model checkpoint to wandb
            # wandb.save(checkpoint_path)
        wandb.finish()

    # Save test metrics - Base utility in case someone doesn't use a logger.
    results = trainer.callback_metrics
    results = {k: v.item() if torch.is_tensor(v) else v for k, v in results.items()}
    with fsspec.open(fs.join(output_dir, "test_results.yaml"), "w") as f:
        yaml.dump(results, f)

    # Save results to centralized CSV for easy notebook visualization
    try:
        _save_results_csv(results, cfg, output_dir)
    except Exception as e:
        logger.warning(f"Failed to save results CSV: {e}")

    # When part of of a hyper-parameter search, we are very specific about how we save our results
    # NOTE (cwognum): We also check if the we are in multi-run mode, as the sweeper is otherwise not active.
    if HYPER_PARAM_SEARCH_CONFIG_KEY in cfg and hydra_cfg.mode == RunMode.MULTIRUN:
        results = extract_main_metric_for_hparam_search(results, cfg[HYPER_PARAM_SEARCH_CONFIG_KEY])

    # Copy the current working directory to remote
    # By default, processes should just write results to Hydra's output directory.
    # However, this currently does not support remote storage, which is why we copy the results here if needed.
    # For more info, see also: https://github.com/facebookresearch/hydra/issues/993

    if dst_dir is not None:
        src_dir = hydra_cfg["runtime"]["output_dir"]
        dst_dir = fs.join(dst_dir, fs.get_basename(src_dir))
        fs.mkdir(dst_dir, exist_ok=True)
        fs.copy_dir(src_dir, dst_dir)

    return results


if __name__ == "__main__":
    cli()
