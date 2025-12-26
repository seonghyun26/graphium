cd ../

TASK_LIST=(
    # 'caco2_wang' 'hia_hou' 'pgp_broccatelli' 'bioavailability_ma'
    'lipophilicity_astrazeneca' 'solubility_aqsoldb' 
    'bbb_martins' 'ppbr_az' 'vdss_lombardo' 
    'cyp2d6_veith' 'cyp3a4_veith' 'cyp2c9_veith' 'cyp2c9_substrate_carbonmangels' 'cyp2d6_substrate_carbonmangels' 'cyp3a4_substrate_carbonmangels' 
    'half_life_obach' 'clearance_hepatocyte_az' 'clearance_microsome_az' 
    'ld50_zhu' 'herg' 'ames' 'dili' 
)
CKPT_LIST=(
    ./model/toymix/gcn/toymixrxrx3_500M.ckpt
)
HID_DIM=5120

for task in "${TASK_LIST[@]}"; do
    for ckpt in "${CKPT_LIST[@]}"; do
        echo $task with $ckpt

        CUDA_VISIBLE_DEVICES=$1 graphium-train \
            model=gcn \
            accelerator=gpu \
            tasks=admet \
            ++constants.seed=0 \
            ++constants.wandb.entity=eddy26 \
            ++constants.wandb.save_dir=null \
            ++constants.wandb.project=graphium \
            ++architecture.task_heads.${task}.hidden_dims=${HID_DIM} \
            ++constants.wandb.tags="['gcn','finetune','unfree_pretrained']" \
            ++constants.raise_train_error=False \
            ++constants.task=$task \
            ++finetuning.task=$task \
            ++datamodule.args.tdc_benchmark_names=$task \
            ++datamodule.args.num_workers=0 \
            +finetuning=admet \
            ++finetuning.pretrained_model=$ckpt \
            ++finetuning.unfreeze_pretrained_depth=16 \
            ++finetuning.epoch_unfreeze_all=40 \
            ++finetuning.finetuning_head.in_dim=${HID_DIM} \
            ++finetuning.finetuning_head.hidden_dims=${HID_DIM} \
            ++finetuning.finetuning_head.depth=4 \
            ++finetuning.new_out_dim=${HID_DIM} \
            ++trainer.model_checkpoint.save_last=False 
        sleep 1
    done
done
