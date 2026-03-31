cd ../

DEVICE=${1:-'0'}
# Deeper (32 layers) and narrower (512) GPS++ configuration
# Uses gpspp_deep model config with 512 dimensions and 32 layers

CUDA_VISIBLE_DEVICES=$DEVICE graphium-train \
    model=gpspp_deep \
    accelerator=gpu \
    tasks=largemix_rxrx3 \
    training=largemix \
    architecture=largemix \
    ++constants.seed=42 \
    ++constants.wandb.entity=eddy26 \
    ++constants.wandb.save_dir=null \
    ++constants.wandb.project=graphium \
    ++constants.wandb.tags="['gpspp','pretrain','largemix','rxrx3']" \
    ++datamodule.args.batch_size_training=100 \
    ++architecture.task_heads.rxrx3.hidden_dims=256 
