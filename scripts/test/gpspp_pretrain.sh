cd ../../

DEVICE=${1:-'0'}
DIM=2048

echo "Pretraining on toymix with hidden dim $DIM"
CUDA_VISIBLE_DEVICES=$DEVICE graphium-train \
    model=gpspp \
    accelerator=gpu \
    tasks=toymix \
    training=toymix \
    architecture=toymix \
    ++constants.seed=0 \
    ++constants.wandb.entity=eddy26 \
    ++constants.wandb.save_dir=null \
    ++constants.wandb.project=graphium \
    ++constants.wandb.tags="['gpspp','pretrain','toymix']" \
    ++constants.data_dir='/home/shpark/prj-molrepr/graphium/data/graphium/neurips2023/small-dataset' \
    ++constants.norm=layer_norm \
    ++architecture.pre_nn.hidden_dims=$DIM \
    ++architecture.pre_nn.out_dim=$DIM \
    ++architecture.pre_nn.out_dim=$DIM \
    ++architecture.pre_nn.hidden_dims=$DIM \
    ++architecture.gnn.layer_kwargs.mpnn_kwargs.in_dim=$DIM \
    ++architecture.gnn.layer_kwargs.mpnn_kwargs.out_dim=$DIM \
    ++architecture.gnn.in_dim=$DIM \
    ++architecture.gnn.hidden_dims=$DIM \
    ++architecture.gnn.out_dim=$DIM \
    ++datamodule.args.num_workers=0 
    # ++datamodule.args.batch_size_training=1024 