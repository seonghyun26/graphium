cd ../

PRETRAIN_DATASET=${1:-'toymix'}
DEVICE=${2:-'0'}
HIDDEN_DIM_LIST=(5120)
# HIDDEN_DIM_LIST=(46 210 696 2200)

echo "Pretraining on $PRETRAIN_DATASET"
for hidden_dim in "${HIDDEN_DIM_LIST[@]}"; do
    echo "Hidden dimension: $hidden_dim"

    if [ $PRETRAIN_DATASET == 'toymix' ]; then
        echo "Pretraining on toy dataset"
        CUDA_VISIBLE_DEVICES=$DEVICE graphium-train \
            model=gcn \
            accelerator=gpu \
            tasks=toymix \
            training=toymix \
            architecture=toymix \
            ++constants.seed=0 \
            ++constants.wandb.entity=eddy26 \
            ++constants.wandb.save_dir=null \
            ++constants.wandb.project=graphium \
            ++constants.wandb.tags="['gcn', 'pretrain']" \
            ++architecture.pre_nn.out_dim=${hidden_dim} \
            ++architecture.gnn.depth=16 \
            ++architecture.gnn.in_dim=${hidden_dim} \
            ++architecture.gnn.out_dim=${hidden_dim} \
            ++architecture.gnn.hidden_dims=${hidden_dim} \
            ++architecture.graph_output_nn.graph.hidden_dims=${hidden_dim} \
            ++architecture.graph_output_nn.graph.out_dim=${hidden_dim} \
            ++architecture.task_heads.qm9.hidden_dims=${hidden_dim} \
            ++architecture.task_heads.tox21.hidden_dims=${hidden_dim} \
            ++architecture.task_heads.zinc.hidden_dims=${hidden_dim} \
            ++datamodule.args.batch_size_training=1024 \
            ++trainer.model_checkpoint.save_last=False 
    
    elif [ $PRETRAIN_DATASET == 'largemix' ]; then
        echo "Pretraining on large dataset"
        CUDA_VISIBLE_DEVICES=$DEVICE graphium-train \
            model=gcn \
            accelerator=gpu \
            tasks=largemix \
            training=largemix \
            architecture=largemix \
            ++constants.seed=0 \
            ++constants.wandb.entity=eddy26 \
            ++constants.wandb.save_dir=null \
            ++constants.wandb.project=graphium \
            ++constants.wandb.tags="['gcn','pretrain','largemix']" \
            ++architecture.pre_nn.out_dim=${hidden_dim} \
            ++architecture.gnn.depth=16 \
            ++architecture.gnn.in_dim=${hidden_dim} \
            ++architecture.gnn.out_dim=${hidden_dim} \
            ++architecture.gnn.hidden_dims=${hidden_dim} \
            ++architecture.graph_output_nn.graph.hidden_dims=${hidden_dim} \
            ++architecture.graph_output_nn.graph.out_dim=${hidden_dim} \
            ++datamodule.args.batch_size_training=400 \
            ++trainer.model_checkpoint.save_last=False 
    
    elif [ $PRETRAIN_DATASET == 'largemix_rxrx3' ]; then
        echo "Pretraining on large+rxrx3 dataset"
        CUDA_VISIBLE_DEVICES=$DEVICE graphium-train \
            model=gcn \
            accelerator=gpu \
            tasks=largemix_rxrx3 \
            training=largemix \
            architecture=largemix \
            ++constants.seed=0 \
            ++constants.wandb.entity=eddy26 \
            ++constants.wandb.save_dir=null \
            ++constants.wandb.project=graphium \
            ++constants.wandb.tags="['gcn','pretrain','largemix','rxrx3']" \
            ++architecture.pre_nn.out_dim=${hidden_dim} \
            ++architecture.gnn.depth=16 \
            ++architecture.gnn.in_dim=${hidden_dim} \
            ++architecture.gnn.out_dim=${hidden_dim} \
            ++architecture.gnn.hidden_dims=${hidden_dim} \
            ++architecture.graph_output_nn.graph.hidden_dims=${hidden_dim} \
            ++architecture.graph_output_nn.graph.out_dim=${hidden_dim} \
            ++datamodule.args.batch_size_training=400 \
            ++trainer.model_checkpoint.save_last=False 

    else
        echo "Invalid dataset"
    fi

    sleep 1
done