cd ../../

DEVICE=${1:-'0'}

CUDA_VISIBLE_DEVICES=$DEVICE graphium-train \
    model=gpspp \
    accelerator=gpu \
    tasks=largemix_debug \
    training=largemix \
    architecture=largemix \
    ++constants.seed=0 \
    ++constants.wandb.entity=eddy26 \
    ++constants.wandb.save_dir=null \
    ++constants.wandb.project=graphium \
    ++constants.wandb.tags="[gpspp,pretrain,debug]" \
    ++constants.norm=layer_norm \
    ++architecture.pre_nn.out_dim='768' \
    ++architecture.gnn.in_dim='768' 