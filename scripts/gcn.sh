cd ../


TASK_LIST=('pgp_broccatelli' 'bioavailability_ma')
# TASK_LIST=(
#     'caco2_wang' 'hia_hou' 'pgp_broccatelli' 'bioavailability_ma' 'lipophilicity_astrazeneca' 'solubility_aqsoldb' 
#     'bbb_martins' 'ppbr_az' 'vdss_lombardo' 
#     'cyp2d6_veith' 'cyp3a4_veith' 'cyp2c9_veith' 'cyp2c9_substrate_carbonmangels' 'cyp2d6_substrate_carbonmangels' 'cyp3a4_substrate_carbonmangels' 
#     'half_life_obach' 'clearance_hepatocyte_az' 'clearance_microsome_az' 
#     'ld50_zhu' 'herg' 'ames' 'dili' 
# )
HIDDEN_DIM_LIST=(46 210 696)

for hidden_dim in "${HIDDEN_DIM_LIST[@]}"; do
    echo "Hidden dimension: $hidden_dim"

    for task in "${TASK_LIST[@]}"; do
        echo "Task: $task"
        sleep 1

        CUDA_VISIBLE_DEVICES=$1 graphium-train \
            model=gcn \
            accelerator=gpu \
            tasks=admet \
            ++constants.task=$task \
            ++datamodule.args.tdc_benchmark_names=$task \
            ++constants.seed=0 \
            ++constants.wandb.entity=eddy26 \
            ++constants.wandb.save_dir=null \
            ++constants.wandb.project=graphium \
            ++constants.wandb.tags="['gcn', 'scratch']" \
            ++architecture.pre_nn.out_dim=${hidden_dim} \
            ++trainer.trainer.precision=32 \
            ++architecture.gnn.depth=16 \
            ++architecture.gnn.in_dim=${hidden_dim} \
            ++architecture.gnn.out_dim=${hidden_dim} \
            ++architecture.gnn.hidden_dims=${hidden_dim} \
            ++architecture.graph_output_nn.graph.hidden_dims=${hidden_dim} \
            ++architecture.graph_output_nn.graph.out_dim=${hidden_dim} \
            ++architecture.task_heads.${task}.hidden_dims=${hidden_dim} 
    done
done