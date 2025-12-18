cd ../

DEVICE=${1:-0}
TASK_LIST=('hia_hou')
# TASK_LIST=(
#     'caco2_wang' 'hia_hou' 'pgp_broccatelli' 'bioavailability_ma' 'lipophilicity_astrazeneca' 'solubility_aqsoldb' 
#     'bbb_martins' 'ppbr_az' 'vdss_lombardo' 
#     'cyp2d6_veith' 'cyp3a4_veith' 'cyp2c9_veith' 'cyp2c9_substrate_carbonmangels' 'cyp2d6_substrate_carbonmangels' 'cyp3a4_substrate_carbonmangels' 
#     'half_life_obach' 'clearance_hepatocyte_az' 'clearance_microsome_az' 
#     'ld50_zhu' 'herg' 'ames' 'dili' 
# )
HIDDEN_DIM=512


ADMET scratch
for task in "${TASK_LIST[@]}"; do
    echo "Training $task"
    sleep 1

    CUDA_VISIBLE_DEVICES=$DEVICE graphium-train \
        model=gcn \
        accelerator=gpu \
        tasks=admet \
        ++constants.task=$task \
        ++datamodule.args.tdc_benchmark_names=$task \
        ++constants.seed=0 \
        ++constants.wandb.entity=eddy26 \
        ++constants.wandb.save_dir=null \
        ++constants.wandb.project=graphium \
        ++constants.wandb.tags="['gcn', 'debug', 'scratch']" \
        ++trainer.trainer.max_epochs=10 \
        ++architecture.pre_nn.out_dim=$HIDDEN_DIM \
        ++architecture.gnn.depth=16 \
        ++architecture.gnn.in_dim=$HIDDEN_DIM \
        ++architecture.gnn.out_dim=$HIDDEN_DIM \
        ++architecture.gnn.hidden_dims=$HIDDEN_DIM \
        ++architecture.graph_output_nn.graph.hidden_dims=$HIDDEN_DIM \
        ++architecture.graph_output_nn.graph.out_dim=$HIDDEN_DIM \
        ++architecture.task_heads.${task}.hidden_dims=$HIDDEN_DIM 
done

# Pretraining largemix
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
    ++constants.wandb.tags="['gcn', 'debug', 'pretrain']" \
    ++trainer.trainer.max_epochs=10 \
    ++architecture.pre_nn.out_dim=$HIDDEN_DIM \
    ++architecture.gnn.depth=16 \
    ++architecture.gnn.in_dim=$HIDDEN_DIM \
    ++architecture.gnn.out_dim=$HIDDEN_DIM \
    ++architecture.gnn.hidden_dims=$HIDDEN_DIM \
    ++architecture.graph_output_nn.graph.hidden_dims=$HIDDEN_DIM \
    ++architecture.graph_output_nn.graph.out_dim=$HIDDEN_DIM \
    ++architecture.task_heads.qm9.hidden_dims=$HIDDEN_DIM \
    ++architecture.task_heads.tox21.hidden_dims=$HIDDEN_DIM \
    ++architecture.task_heads.zinc.hidden_dims=$HIDDEN_DIM \
    ++datamodule.args.task_specific_args.qm9.df_path=./data/graphium/neurips2023/small-dataset/qm9.csv \
    ++datamodule.args.task_specific_args.qm9.splits_path=./data/graphium/neurips2023/small-dataset/qm9_random_splits.pt \
    ++datamodule.args.task_specific_args.tox21.df_path=./data/graphium/neurips2023/small-dataset/Tox21-7k-12-labels.csv \
    ++datamodule.args.task_specific_args.tox21.splits_path=./data/graphium/neurips2023/small-dataset/Tox21_random_splits.pt \
    ++datamodule.args.task_specific_args.zinc.df_path=./data/graphium/neurips2023/small-dataset/ZINC12k.csv \
    ++datamodule.args.task_specific_args.zinc.splits_path=./data/graphium/neurips2023/small-dataset/ZINC12k_random_splits.pt \
    ++datamodule.args.task_specific_args.qm9.df_path=./data/graphium/neurips2023/small-dataset/qm9.csv \
    ++datamodule.args.task_specific_args.qm9.splits_path=./data/graphium/neurips2023/small-dataset/qm9_random_splits.pt \
    ++datamodule.args.task_specific_args.tox21.df_path=./data/graphium/neurips2023/small-dataset/Tox21-7k-12-labels.csv \
    ++datamodule.args.task_specific_args.tox21.splits_path=./data/graphium/neurips2023/small-dataset/Tox21_random_splits.pt \
    ++datamodule.args.task_specific_args.zinc.df_path=./data/graphium/neurips2023/small-dataset/ZINC12k.csv \
    ++datamodule.args.task_specific_args.zinc.splits_path=./data/graphium/neurips2023/small-dataset/ZINC12k_random_splits.pt \
    ++datamodule.args.task_specific_args.pcqm4m_g25_n4.df_path=./data/graphium/neurips2023/large-dataset/PCQM4M_G25_N4.parquet \
    ++datamodule.args.task_specific_args.pcqm4m_g25_n4.splits_path=./data/graphium/neurips2023/large-dataset/pcqm4m_g25_n4_random_splits.pt \
    ++datamodule.args.task_specific_args.pcba_1328.df_path=./data/graphium/neurips2023/large-dataset/PCBA_1328_1564k.parquet \
    ++datamodule.args.task_specific_args.pcba_1328.splits_path=./data/graphium/neurips2023/large-dataset/pcba_1328_random_splits.pt \
    ++datamodule.args.task_specific_args.l1000_vcap.df_path=./data/graphium/neurips2023/large-dataset/LINCS_L1000_VCAP_0-2_th2.csv \
    ++datamodule.args.task_specific_args.l1000_vcap.splits_path=./data/graphium/neurips2023/large-dataset/l1000_vcap_random_splits.pt \
    ++datamodule.args.task_specific_args.l1000_mcf7.df_path=./data/graphium/neurips2023/large-dataset/LINCS_L1000_MCF7_0-2_th2.csv \
    ++datamodule.args.task_specific_args.l1000_mcf7.splits_path=./data/graphium/neurips2023/large-dataset/l1000_mcf7_random_splits.pt \

