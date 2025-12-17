cd ../../

DEVICE=${1:-'2'}

CUDA_VISIBLE_DEVICES=$DEVICE graphium-train \
    tasks=largemix_rxrx3 \
    model=gpspp \
    accelerator=gpu \
    training=largemix \
    architecture=largemix \
    ++constants.seed=0 \
    ++constants.wandb.entity=eddy26 \
    ++constants.wandb.save_dir=null \
    ++constants.wandb.project=graphium \
    ++constants.wandb.tags="['gpspp', 'pretrain']" \
    ++constants.norm=layer_norm \
    ++constants.data_dir='./data/graphium/neurips2023/large-dataset/' 