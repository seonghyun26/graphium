cd ../

DEVICE=${1:-'0'}

TASK_LIST=(
    'caco2_wang' 'hia_hou' 'pgp_broccatelli' 'bioavailability_ma' 'lipophilicity_astrazeneca' 'solubility_aqsoldb' 
    'bbb_martins' 'ppbr_az' 'vdss_lombardo' 
    'cyp2d6_veith' 'cyp3a4_veith' 'cyp2c9_veith' 'cyp2c9_substrate_carbonmangels' 'cyp2d6_substrate_carbonmangels' 'cyp3a4_substrate_carbonmangels' 
    'half_life_obach' 'clearance_hepatocyte_az' 'clearance_microsome_az' 
    'ld50_zhu' 'herg' 'ames' 'dili' 
)
CKPT_LIST=(
    # ./model/largemix/gpspp/largemix_15M.ckpt
    ./model/largemix/gpspp/largemix_gpspp_500M.ckpt
)

for task in "${TASK_LIST[@]}"; do
    for ckpt in "${CKPT_LIST[@]}"; do
        echo $task with $ckpt

        CUDA_VISIBLE_DEVICES=$1 graphium-train \
            model=gpspp \
            accelerator=gpu \
            tasks=admet \
            ++constants.task=$task \
            ++finetuning.task=$task \
            ++datamodule.args.tdc_benchmark_names=$task \
            +finetuning=admet_gpspp \
            ++finetuning.pretrained_model=$ckpt \
            ++finetuning.unfreeze_pretrained_depth=4 \
            ++finetuning.epoch_unfreeze_all=20 \
            ++finetuning.finetuning_head.in_dim=256 \
            ++finetuning.finetuning_head.hidden_dims=256 \
            ++finetuning.finetuning_head.depth=4 \
            ++finetuning.new_out_dim=256 \
            ++finetuning.added_depth=4 \
            ++architecture.task_heads.${task}.hidden_dims=256 \
            ++constants.seed=0 \
            ++constants.wandb.entity=eddy26 \
            ++constants.wandb.save_dir=null \
            ++constants.wandb.project=graphium \
            ++constants.wandb.tags="['gpspp','finetune','largemix','unfree_pretrained']" \
            ++constants.raise_train_error=False \
            ++datamodule.args.num_workers=0 
            
        sleep 1
    done
done
