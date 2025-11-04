cd ../

# CUDA_VISIBLE_DEVICES=$1 graphium-train \
#     architecture=toymix \
#     tasks=toymix \
#     training=toymix \
#     model=gcn \
#     accelerator=gpu \
#     datamodule.args.processed_graph_data_path=/home/shpark/prj-kmelloddy/benchmark/data/graphium/toymix \
#     ++datamodule.args.task_specific_args.qm9.df_path=/home/shpark/prj-kmelloddy/benchmark/data/graphium/neurips2023/small-dataset/qm9.csv \
#     ++datamodule.args.task_specific_args.qm9.splits_path=/home/shpark/prj-kmelloddy/benchmark/data/graphium/neurips2023/small-dataset/qm9_random_splits.pt \
#     ++datamodule.args.task_specific_args.tox21.df_path=/home/shpark/prj-kmelloddy/benchmark/data/graphium/neurips2023/small-dataset/Tox21-7k-12-labels.csv \
#     ++datamodule.args.task_specific_args.tox21.splits_path=/home/shpark/prj-kmelloddy/benchmark/data/graphium/neurips2023/small-dataset/Tox21_random_splits.pt \
#     ++datamodule.args.task_specific_args.zinc.df_path=/home/shpark/prj-kmelloddy/benchmark/data/graphium/neurips2023/small-dataset/ZINC12k.csv \
#     ++datamodule.args.task_specific_args.zinc.splits_path=/home/shpark/prj-kmelloddy/benchmark/data/graphium/neurips2023/small-dataset/ZINC12k_random_splits.pt \


# TASK_LIST=('caco2_wang' 'hia_hou' 'pgp_broccatelli' 'bioavailability_ma' 'lipophilicity_astrazeneca' 'solubility_aqsoldb' 'bbb_martins' 'ppbr_az' 'vdss_lombardo' 'cyp2d6_veith' 'cyp3a4_veith' 'cyp2c9_veith' 'cyp2d6_substrate_carbonmangels' 'cyp3a4_substrate_carbonmangels' 'cyp2c9_substrate_carbonmangels' 'half_life_obach' 'clearance_microsome_az' 'clearance_hepatocyte_az' 'herg' 'ames' 'dili' 'ld50_zhu')
# TASK_LIST=('lipophilicity_astrazeneca' 'solubility_aqsoldb' 'ppbr_az' 'vdss_lombardo' 'half_life_obach' 'clearance_microsome_az' 'clearance_hepatocyte_az' 'ld50_zhu')
TASK_LIST=('ld50_zhu')
for task in "${TASK_LIST[@]}"; do
    echo $task
    CUDA_VISIBLE_DEVICES=$1 graphium-train \
        model=gcn \
        accelerator=gpu \
        finetuning.pretrained_model="./models_checkpoints/small-dataset/gcn/2025-11-03_12-24-02_20251103_122402/neurips2023_small_data_gcn_20251103_122402.ckpt" \
        +finetuning=admet \
        ++constants.task=$task \
        ++finetuning.task=$task \
        ++datamodule.args.tdc_benchmark_names=$task \
        ++finetuning.keep_modules_after_finetuning_module=null \
        ++constants.wandb.save_dir=null \
        ++constants.wandb.entity=eddy26 \
        ++constants.wandb.tags="['gcn', 'pretrained']" \
        ++constants.wandb.project=graphium
done