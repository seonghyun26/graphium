cd ../

DEVICE=${1:-'0'}
DIM=2048

CUDA_VISIBLE_DEVICES=$DEVICE graphium-train \
    model=gpspp \
    accelerator=gpu \
    tasks=largemix_rxrx3_dti \
    training=largemix \
    architecture=largemix \
    ++constants.seed=0 \
    ++constants.wandb.entity=eddy26 \
    ++constants.wandb.save_dir=null \
    ++constants.wandb.project=graphium \
    ++constants.wandb.tags="['gpspp','pretrain','largemix','rxrx3','tdcdti']" \
    ++datamodule.args.batch_size_training=100 \
    ++architecture.pre_nn.out_dim=$DIM \
    ++architecture.pre_nn.hidden_dims=$DIM \
    ++architecture.gnn.layer_kwargs.mpnn_kwargs.in_dim=$DIM \
    ++architecture.gnn.layer_kwargs.mpnn_kwargs.out_dim=$DIM \
    ++architecture.gnn.in_dim=$DIM \
    ++architecture.gnn.hidden_dims=$DIM \
    ++architecture.gnn.out_dim=$DIM