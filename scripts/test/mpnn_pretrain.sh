cd ../../

DEVICE=${1:-'0'}
HIDDEN_DIM_LIST=(128 400)

for hidden_dim in "${HIDDEN_DIM_LIST[@]}"; do
    echo "Hidden dimension: $hidden_dim"

    CUDA_VISIBLE_DEVICES=$DEVICE graphium-train \
        model=mpnn \
        accelerator=gpu \
        tasks=toymix_rxrx3 \
        training=toymix \
        architecture=toymix \
        ++constants.seed=0 \
        ++constants.wandb.entity=eddy26 \
        ++constants.wandb.save_dir=null \
        ++constants.wandb.project=graphium \
        ++constants.wandb.tags="['mpnn','pretrain','cell','toymix']" \
        ++constants.data_dir='/home/shpark/prj-molrepr/graphium/data/graphium/neurips2023/small-dataset' \
        ++constants.datacache_path='../datacache/rxrx3/' \
        ++architecture.pre_nn.hidden_dims=${hidden_dim} \
        ++architecture.pre_nn.out_dim=${hidden_dim} \
        ++architecture.pre_nn_edges.out_dim=${hidden_dim} \
        ++architecture.pre_nn_edges.hidden_dims=${hidden_dim} \
        ++architecture.gnn.depth=16 \
        ++architecture.gnn.in_dim=${hidden_dim} \
        ++architecture.gnn.out_dim=${hidden_dim} \
        ++constants.gnn_edge_dim=${hidden_dim} \
        ++architecture.gnn.hidden_dims=${hidden_dim} \
        ++architecture.graph_output_nn.graph.hidden_dims=${hidden_dim} \
        ++architecture.graph_output_nn.graph.out_dim=${hidden_dim} \
        ++architecture.task_heads.qm9.hidden_dims=${hidden_dim} \
        ++architecture.task_heads.tox21.hidden_dims=${hidden_dim} \
        ++architecture.task_heads.zinc.hidden_dims=${hidden_dim} 

    sleep 1
done