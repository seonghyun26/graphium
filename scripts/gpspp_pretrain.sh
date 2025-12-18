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
    ++constants.wandb.tags="['gpspp', 'pretrain','largemix']" \
    ++constants.norm=layer_norm \
    ++architecture.pre_nn.out_dim=128 \
    ++architecture.pre_nn.hidden_dims=128 \
    ++architecture.gnn.layer_kwargs.mpnn_kwargs.in_dim=128 \
    ++architecture.gnn.layer_kwargs.mpnn_kwargs.out_dim=128 \
    ++architecture.gnn.in_dim=128 \
    ++architecture.gnn.hidden_dims=128 \
    ++architecture.gnn.out_dim=128