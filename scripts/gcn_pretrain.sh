cd ../

PRETRAIN_DATASET=${1:-'toymix'}
DEVICE=${2:-'0'}
HIDDEN_DIM_LIST=(46 210 696)

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
            ++datamodule.args.task_specific_args.qm9.df_path=./data/graphium/neurips2023/small-dataset/qm9.csv \
            ++datamodule.args.task_specific_args.qm9.splits_path=./data/graphium/neurips2023/small-dataset/qm9_random_splits.pt \
            ++datamodule.args.task_specific_args.tox21.df_path=./data/graphium/neurips2023/small-dataset/Tox21-7k-12-labels.csv \
            ++datamodule.args.task_specific_args.tox21.splits_path=./data/graphium/neurips2023/small-dataset/Tox21_random_splits.pt \
            ++datamodule.args.task_specific_args.zinc.df_path=./data/graphium/neurips2023/small-dataset/ZINC12k.csv \
            ++datamodule.args.task_specific_args.zinc.splits_path=./data/graphium/neurips2023/small-dataset/ZINC12k_random_splits.pt
    
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
            ++constants.wandb.tags="['gcn', 'pretrain']" \
            ++architecture.pre_nn.out_dim=${hidden_dim} \
            ++architecture.gnn.depth=16 \
            ++architecture.gnn.in_dim=${hidden_dim} \
            ++architecture.gnn.out_dim=${hidden_dim} \
            ++architecture.gnn.hidden_dims=${hidden_dim} \
            ++architecture.graph_output_nn.graph.hidden_dims=${hidden_dim} \
            ++architecture.graph_output_nn.graph.out_dim=${hidden_dim} \
            ++architecture.task_heads.pcqm4m_g25_n4.hidden_dims=${hidden_dim} \
            ++architecture.task_heads.pcba_1328.hidden_dims=${hidden_dim} \
            ++architecture.task_heads.l1000_vcap.hidden_dims=${hidden_dim} \
            ++architecture.task_heads.l1000_mcf7.hidden_dims=${hidden_dim} \
            ++datamodule.args.task_specific_args.l1000_vcap.df_path=./data/graphium/neurips2023/large-dataset/LINCS_L1000_VCAP_0-2_th2.csv.gz \
            ++datamodule.args.task_specific_args.l1000_vcap.splits_path=./data/graphium/neurips2023/large-dataset/l1000_vcap_random_splits.pt \
            ++datamodule.args.task_specific_args.l1000_vcap.smiles_col='SMILES' \
            ++datamodule.args.task_specific_args.l1000_vcap.label_cols=geneID-* \
            ++datamodule.args.task_specific_args.l1000_vcap.task_level=graph \
            ++datamodule.args.task_specific_args.l1000_mcf7.df_path=./data/graphium/neurips2023/large-dataset/LINCS_L1000_MCF7_0-2_th2.csv.gz \
            ++datamodule.args.task_specific_args.l1000_mcf7.splits_path=./data/graphium/neurips2023/large-dataset/l1000_mcf7_random_splits.pt \
            ++datamodule.args.task_specific_args.l1000_mcf7.smiles_col='SMILES' \
            ++datamodule.args.task_specific_args.l1000_mcf7.label_cols=geneID-* \
            ++datamodule.args.task_specific_args.l1000_mcf7.task_level=graph \
            ++datamodule.args.task_specific_args.pcba_1328.df_path=./data/graphium/neurips2023/large-dataset/PCBA_1328_1564k.parquet \
            ++datamodule.args.task_specific_args.pcba_1328.splits_path=./data/graphium/neurips2023/large-dataset/pcba_1328_random_splits.pt \
            ++datamodule.args.task_specific_args.pcba_1328.smiles_col='SMILES' \
            ++datamodule.args.task_specific_args.pcba_1328.label_cols=assayID-* \
            ++datamodule.args.task_specific_args.pcba_1328.task_level=graph \
            ++datamodule.args.task_specific_args.pcqm4m_g25_n4.df_path=./data/graphium/neurips2023/large-dataset/PCQM4M_G25_N4.parquet \
            ++datamodule.args.task_specific_args.pcqm4m_g25_n4.splits_path=./data/graphium/neurips2023/large-dataset/pcqm4m_g25_n4_random_splits.pt \
            ++datamodule.args.task_specific_args.pcqm4m_g25_n4.smiles_col='ordered_smiles' \
            ++datamodule.args.task_specific_args.pcqm4m_g25_n4.label_cols='graph_*' \
            ++datamodule.args.task_specific_args.pcqm4m_g25_n4.task_level=graph
    
    else
        echo "Invalid dataset"
    fi

    sleep 1
done