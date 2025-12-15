cd ../../

# Regression
# TASK_LIST=(
#     'caco2_wang'
#     # 'hia_hou' 'pgp_broccatelli' 'bioavailability_ma'
#     'lipophilicity_astrazeneca' 'solubility_aqsoldb' 
#     # 'bbb_martins' 
#     'ppbr_az' 'vdss_lombardo' 
#     # 'cyp2d6_veith' 'cyp3a4_veith' 'cyp2c9_veith' 'cyp2c9_substrate_carbonmangels' 'cyp2d6_substrate_carbonmangels' 'cyp3a4_substrate_carbonmangels' 
#     'half_life_obach' 'clearance_hepatocyte_az' 'clearance_microsome_az' 
#     'ld50_zhu'
#     # 'herg' 'ames' 'dili' 
# )
# Classification
# TASK_LIST=(
#     # 'caco2_wang'
#     'hia_hou' 'pgp_broccatelli' 'bioavailability_ma'
#     # 'lipophilicity_astrazeneca' 'solubility_aqsoldb' 
#     'bbb_martins' 
#     # 'ppbr_az' 'vdss_lombardo' 
#     'cyp2d6_veith' 'cyp3a4_veith' 'cyp2c9_veith' 'cyp2c9_substrate_carbonmangels' 'cyp2d6_substrate_carbonmangels' 'cyp3a4_substrate_carbonmangels' 
#     # 'half_life_obach' 'clearance_hepatocyte_az' 'clearance_microsome_az' 
#     # 'ld50_zhu'
#     'herg' 'ames' 'dili' 
# )
TASK_LIST=(
    'caco2_wang' 'hia_hou' 'pgp_broccatelli' 'bioavailability_ma' 'lipophilicity_astrazeneca' 'solubility_aqsoldb' 
    'bbb_martins' 'ppbr_az' 'vdss_lombardo' 
    'cyp2d6_veith' 'cyp3a4_veith' 'cyp2c9_veith' 'cyp2c9_substrate_carbonmangels' 'cyp2d6_substrate_carbonmangels' 'cyp3a4_substrate_carbonmangels' 
    'half_life_obach' 'clearance_hepatocyte_az' 'clearance_microsome_az' 
    'ld50_zhu' 'herg' 'ames' 'dili' 
)
CKPT_LIST=(
    # ./model/gcn/toymixrxrx3_small.ckpt
    # ./model/gcn/toymixrxrx3_medium.ckpt
    # ./model/gcn/toymixrxrx3_large.ckpt
    ./model/gcn/toymixrxrx3_ultra.ckpt
)

for task in "${TASK_LIST[@]}"; do
    for ckpt in "${CKPT_LIST[@]}"; do
        echo $task with $ckpt

        CUDA_VISIBLE_DEVICES=$1 graphium-train \
            model=gcn \
            accelerator=gpu \
            tasks=admet \
            ++constants.task=$task \
            ++finetuning.task=$task \
            ++datamodule.args.tdc_benchmark_names=$task \
            +finetuning=admet \
            ++finetuning.pretrained_model=$ckpt \
            ++finetuning.unfreeze_pretrained_depth=16 \
            ++constants.seed=0 \
            ++constants.wandb.entity=eddy26 \
            ++constants.wandb.save_dir=null \
            ++constants.wandb.project=graphium \
            ++constants.wandb.tags="['gcn','finetune','unfree_pretrained','toymixrxrx3']" \
            ++constants.raise_train_error=False 
            
        sleep 1
    done
done
