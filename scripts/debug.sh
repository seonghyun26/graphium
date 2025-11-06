cd ../

DEVICE=${1:-0}
echo "Using device: $DEVICE"

# TASK_LIST=('caco2_wang' 'lipophilicity_astrazeneca' 'solubility_aqsoldb' 'ppbr_az' 'vdss_lombardo' 'half_life_obach' 'clearance_microsome_az' 'clearance_hepatocyte_az' 'ld50_zhu')
TASK_LIST=(cyp2d6_veith cyp3a4_veith cyp2c9_veith cyp2d6_substrate_carbonmangels cyp3a4_substrate_carbonmangels cyp2c9_substrate_carbonmangels herg ames dili)

for task in "${TASK_LIST[@]}"; do
    echo $task
    CUDA_VISIBLE_DEVICES=$DEVICE graphium-train \
        model=gcn \
        accelerator=gpu \
        tasks=admet \
        ++constants.task=$task \
        ++datamodule.args.tdc_benchmark_names=$task \
        ++architecture.gnn.hidden_dims=256 \
        ++architecture.gnn.out_dim=256 \
        ++architecture.gnn.depth=8 \
        ++constants.seed=0 \
        ++constants.wandb.save_dir=null \
        ++constants.wandb.entity=eddy26 \
        ++constants.wandb.tags="['debug']" \
        ++constants.wandb.project=graphium
done