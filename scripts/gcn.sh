cd ../

# graphium-train \
#     +finetuning=admet \
#     model=gcn \
#     accelerator=gpu \
#     finetuning.pretrained_model=dummy-pretrained-model \
#     datamodule.args.processed_graph_data_path=/home/shpark/prj-kmelloddy/benchmark/data/graphium/toymix \
#     ++datamodule.args.task_specific_args.qm9.df_path=/home/shpark/prj-kmelloddy/benchmark/data/graphium/neurips2023/small-dataset/qm9.csv \
#     ++datamodule.args.task_specific_args.qm9.splits_path=/home/shpark/prj-kmelloddy/benchmark/data/graphium/neurips2023/small-dataset/qm9_random_splits.pt \
#     ++datamodule.args.task_specific_args.tox21.df_path=/home/shpark/prj-kmelloddy/benchmark/data/graphium/neurips2023/small-dataset/Tox21-7k-12-labels.csv \
#     ++datamodule.args.task_specific_args.tox21.splits_path=/home/shpark/prj-kmelloddy/benchmark/data/graphium/neurips2023/small-dataset/Tox21_random_splits.pt \
#     ++datamodule.args.task_specific_args.zinc.df_path=/home/shpark/prj-kmelloddy/benchmark/data/graphium/neurips2023/small-dataset/ZINC12k.csv \
#     ++datamodule.args.task_specific_args.zinc.splits_path=/home/shpark/prj-kmelloddy/benchmark/data/graphium/neurips2023/small-dataset/ZINC12k_random_splits.pt

# graphium finetune admet \
#     --name admet \
#     model=gcn \
#     accelerator=gpu \
#     finetuning.pretrained_model=dummy-pretrained-model \

# # Regression tasks
# CUDA_VISIBLE_DEVICES=$1 graphium-train \
#     +finetuning=admet \
#     ++finetuning.task=['caco2_wang','lipophilicity_astrazeneca','solubility_aqsoldb','ppbr_az','half_life_obach','clearance_microsome_az','clearance_hepatocyte_az','ld50_zhu'] \
#     model=gcn \
#     accelerator=gpu \
#     ++constants.wandb.save_dir=null \
#     ++constants.wandb.entity=eddy26 \
#     ++constants.wandb.tags="['gcn']" \
#     ++constants.wandb.project=graphium 

# # Classification tasks
# CUDA_VISIBLE_DEVICES=$1 graphium-train \
#     +finetuning=admet \
#     ++finetuning.task=['hia_hou','pgp_broccatelli','bioavailability_ma','bbb_martins','cyp2d6_veith','cyp3a4_veith','cyp2c9_veith','cyp2d6_substrate_carbonmangels','cyp3a4_substrate_carbonmangels','cyp2c9_substrate_carbonmangels','herg','ames','dili'] \
#     model=gcn \
#     accelerator=gpu \
#     ++constants.wandb.save_dir=null \
#     ++constants.wandb.entity=eddy26 \
#     ++constants.wandb.tags="['gcn']" \
#     ++constants.wandb.project=graphium 


# TASK_LIST=('caco2_wang' 'lipophilicity_astrazeneca' 'solubility_aqsoldb' 'ppbr_az' 'vdss_lombardo' 'half_life_obach' 'clearance_microsome_az' 'clearance_hepatocyte_az' 'ld50_zhu')
TASK_LIST=('ld50_zhu')
for task in "${TASK_LIST[@]}"; do
    echo $task
    CUDA_VISIBLE_DEVICES=$1 graphium-train \
        model=gcn \
        accelerator=gpu \
        +finetuning=admet \
        ++constants.task=$task \
        ++finetuning.task=$task \
        ++datamodule.args.tdc_benchmark_names=$task \
        ++finetuning.keep_modules_after_finetuning_module=null \
        ++constants.wandb.save_dir=null \
        ++constants.wandb.entity=eddy26 \
        ++constants.wandb.tags="['gcn']" \
        ++constants.wandb.project=graphium
done