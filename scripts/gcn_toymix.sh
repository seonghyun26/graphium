cd ../

TASK_LIST=(
    'caco2_wang' 'hia_hou' 'pgp_broccatelli' 'bioavailability_ma' 'lipophilicity_astrazeneca' 'solubility_aqsoldb' 
    'bbb_martins' 'ppbr_az' 'vdss_lombardo' 
    'cyp2d6_veith' 'cyp3a4_veith' 'cyp2c9_veith' 'cyp2c9_substrate_carbonmangels' 'cyp2d6_substrate_carbonmangels' 'cyp3a4_substrate_carbonmangels' 
    'half_life_obach' 'clearance_hepatocyte_az' 'clearance_microsome_az' 
    'ld50_zhu' 'herg' 'ames' 'dili' 
)
for task in "${TASK_LIST[@]}"; do
    echo $task
    CUDA_VISIBLE_DEVICES=$1 graphium-train \
        model=gcn \
        accelerator=gpu \
        +finetuning=admet \
        finetuning.pretrained_model="./models_checkpoints/small-dataset/gcn/2025-11-03_12-24-02_20251103_122402/neurips2023_small_data_gcn_20251103_122402.ckpt" \
        ++constants.task=$task \
        ++finetuning.task=$task \
        ++datamodule.args.tdc_benchmark_names=$task \
        ++finetuning.keep_modules_after_finetuning_module=null \
        ++constants.wandb.save_dir=null \
        ++constants.wandb.entity=eddy26 \
        ++constants.wandb.tags="['gcn', 'pretrained']" \
        ++constants.wandb.project=graphium

    sleep 1
done

TASK_LIST=('caco2_wang' 'hia_hou' 'pgp_broccatelli' 'bioavailability_ma' 'lipophilicity_astrazeneca' 
    'solubility_aqsoldb' 'bbb_martins' 'ppbr_az' 'vdss_lombardo' 'cyp2d6_veith' 'cyp3a4_veith' 
    'cyp2c9_veith' 'cyp2d6_substrate_carbonmangels' 'cyp3a4_substrate_carbonmangels' 
    'cyp2c9_substrate_carbonmangels' 'half_life_obach' 'clearance_microsome_az' 
    'clearance_hepatocyte_az' 'herg' 'ames' 'dili' 'ld50_zhu')
GNN_HIDDEN_DIMS=(96 256 1024)
GNN_DEPTHS=(4 8 12)

for task in "${TASK_LIST[@]}"; do

    for GNN_HIDDEN_DIM in "${GNN_HIDDEN_DIMS[@]}"; do
        for GNN_DEPTH in "${GNN_DEPTHS[@]}"; do
            echo $task
            echo $GNN_HIDDEN_DIM
            echo $GNN_DEPTH
            sleep 1

            CUDA_VISIBLE_DEVICES=$1 graphium-train \
                model=gcn \
                accelerator=gpu \
                +finetuning=admet \
                finetuning.pretrained_model="./models_checkpoints/small-dataset/gcn/2025-11-03_12-24-02_20251103_122402/neurips2023_small_data_gcn_20251103_122402.ckpt" \
                ++finetuning.task=$task \
                ++finetuning.keep_modules_after_finetuning_module=null \
                ++constants.task=$task \
                ++datamodule.args.tdc_benchmark_names=$task \
                ++constants.seed=0 \
                ++constants.wandb.entity=eddy26 \
                ++constants.wandb.save_dir=null \
                ++constants.wandb.project=graphium \
                ++constants.wandb.tags="['gcn','pretrained']" \
                ++architecture.gnn.hidden_dims=$GNN_HIDDEN_DIM \
                ++architecture.gnn.depth=$GNN_DEPTH 
        done
    done
done