cd ../


# TASK_LIST=(
#     'caco2_wang'
#     # 'hia_hou' 'pgp_broccatelli' 'bioavailability_ma'
#     'lipophilicity_astrazeneca' 'solubility_aqsoldb' 
#     # 'bbb_martins' 
#     'ppbr_az' 'vdss_lombardo' 
#     # 'cyp2d6_veith' 'cyp3a4_veith' 'cyp2c9_veith' 'cyp2c9_substrate_carbonmangels' 'cyp2d6_substrate_carbonmangels' 'cyp3a4_substrate_carbonmangels' 
#     'half_life_obach' 'clearance_hepatocyte_az' 'clearance_microsome_az' 
#     'ld50_zhu'
#     # 'herg' 'ames' 'dili' 
# )
TASK_LIST=(
    # 'caco2_wang'
    'hia_hou' 'pgp_broccatelli' 'bioavailability_ma'
    # 'lipophilicity_astrazeneca' 'solubility_aqsoldb' 
    'bbb_martins' 
    # 'ppbr_az' 'vdss_lombardo' 
    'cyp2d6_veith' 'cyp3a4_veith' 'cyp2c9_veith' 'cyp2c9_substrate_carbonmangels' 'cyp2d6_substrate_carbonmangels' 'cyp3a4_substrate_carbonmangels' 
    # 'half_life_obach' 'clearance_hepatocyte_az' 'clearance_microsome_az' 
    # 'ld50_zhu'
    'herg' 'ames' 'dili' 
)
CKPT_LIST=(
    ./model/mpnn/toymix_small.ckpt
    # ./model/mpnn/toymix_medium.ckpt
    # ./model/mpnn/toymix_large.ckpt
)
HIDDEN_DIM_LIST=(40 128 400)

for task in "${TASK_LIST[@]}"; do
    for i in "${!CKPT_LIST[@]}"; do
    ckpt="${CKPT_LIST[$i]}"
    hidden_dim="${HIDDEN_DIM_LIST[$i]}"
        echo $task with $ckpt and hidden dimension $hidden_dim

        CUDA_VISIBLE_DEVICES=$1 graphium-train \
            model=mpnn \
            accelerator=gpu \
            tasks=admet \
            ++constants.task=$task \
            ++finetuning.task=$task \
            ++datamodule.args.tdc_benchmark_names=$task \
            +finetuning=admet \
            finetuning.pretrained_model=$ckpt \
            ++constants.seed=0 \
            ++constants.wandb.entity=eddy26 \
            ++constants.wandb.save_dir=null \
            ++constants.wandb.project=graphium \
            ++constants.wandb.tags="['mpnn','finetune']" \
            ++constants.raise_train_error=False \
            ++architecture.pre_nn.hidden_dims=${hidden_dim} \
            ++architecture.pre_nn.out_dim=${hidden_dim} \
            ++architecture.pre_nn_edges.out_dim=${hidden_dim} \
            ++architecture.pre_nn_edges.hidden_dims=${hidden_dim} \
            ++architecture.gnn.in_dim=${hidden_dim} \
            ++architecture.gnn.out_dim=${hidden_dim} \
            ++constants.gnn_dim=${hidden_dim} \
            ++constants.gnn_edge_dim=${hidden_dim} \
            ++architecture.gnn.hidden_dims=${hidden_dim} \
            ++architecture.graph_output_nn.graph.hidden_dims=${hidden_dim} \
            ++architecture.graph_output_nn.graph.out_dim=${hidden_dim} 
        sleep 1
    done
done
