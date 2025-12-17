cd ../

DEVICE=${1:-'0'}

CUDA_VISIBLE_DEVICES=$DEVICE graphium-train \
    model=gpspp \
    accelerator=gpu \
    tasks=largemix \
    training=largemix \
    architecture=largemix \
    ++constants.seed=0 \
    ++constants.wandb.entity=eddy26 \
    ++constants.wandb.save_dir=null \
    ++constants.wandb.project=graphium \
    ++constants.wandb.tags="['gpspp','pretrain','pcqm']" \
    ++constants.norm=layer_norm 