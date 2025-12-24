cd ../../

DEVICE=${1:-'0'}
DIM=2048

echo "Pretraining on toymix with hidden dim $DIM"
CUDA_VISIBLE_DEVICES=$DEVICE graphium-train \
    model=gpspp \
    accelerator=gpu \
    tasks=toymix_rxrx3 \
    training=toymix \
    architecture=toymix \
    ++constants.seed=0 \
    ++constants.data_dir='/home/shpark/prj-molrepr/graphium/data/graphium/neurips2023/small-dataset' \
    ++constants.norm=layer_norm \
    ++constants.wandb.tags="['gpspp','pretrain','toymix','rxrx3']" \
    ++architecture.pre_nn.hidden_dims=$DIM \
    ++architecture.pre_nn.out_dim=$DIM \
    ++architecture.gnn.layer_kwargs.mpnn_kwargs.in_dim=$DIM \
    ++architecture.gnn.layer_kwargs.mpnn_kwargs.out_dim=$DIM \
    ++architecture.gnn.in_dim=$DIM \
    ++architecture.gnn.hidden_dims=$DIM \
    ++architecture.gnn.out_dim=$DIM \
    ++datamodule.args.num_workers=0 
    # ++datamodule.args.batch_size_training=1024 