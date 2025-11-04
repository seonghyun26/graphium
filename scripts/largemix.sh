cd ../../

CUDA_VISIBLE_DEVICES=$1 graphium-train \
    architecture=largemix \
    tasks=largemix \
    training=largemix \
    model=$2 \
    accelerator=gpu \
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
    ++datamodule.args.processed_graph_data_path=./data/graphium/largemix
