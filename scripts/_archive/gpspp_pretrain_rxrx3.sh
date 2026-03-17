cd ../

DEVICE=${1:-'0'}
# Deeper (16 layers) and narrower (1024) GPS++ for similar total model size
# compared to the default 4-layer 2048-dim config
# 16 * 1024^2 ≈ 4 * 2048^2
DEPTH=16
DIM=1024

CUDA_VISIBLE_DEVICES=$DEVICE graphium-train \
    model=gpspp \
    accelerator=gpu \
    tasks=rxrx3 \
    training=rxrx3 \
    architecture=largemix \
    ++constants.seed=42 \
    ++constants.wandb.entity=eddy26 \
    ++constants.wandb.save_dir=null \
    ++constants.wandb.project=graphium \
    ++constants.wandb.tags="['gpspp','pretrain','rxrx3','cellimage']" \
    ++constants.norm=layer_norm \
    ++architecture.pre_nn.out_dim=$DIM \
    ++architecture.pre_nn.hidden_dims=$DIM \
    ++architecture.gnn.in_dim=$DIM \
    ++architecture.gnn.out_dim=$DIM \
    ++architecture.gnn.hidden_dims=$DIM \
    ++architecture.gnn.depth=$DEPTH \
    ++architecture.gnn.layer_kwargs.mpnn_kwargs.in_dim=$DIM \
    ++architecture.gnn.layer_kwargs.mpnn_kwargs.out_dim=$DIM \
    ++architecture.graph_output_nn.graph.hidden_dims=$DIM \
    ++architecture.task_heads.rxrx3.hidden_dims=256
