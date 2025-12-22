cd ../


TASK_LIST=(
    'caco2_wang' 'hia_hou' 'pgp_broccatelli' 'bioavailability_ma' 'lipophilicity_astrazeneca' 'solubility_aqsoldb' 
    'bbb_martins' 'ppbr_az' 'vdss_lombardo' 
    'cyp2d6_veith' 'cyp3a4_veith' 'cyp2c9_veith' 'cyp2c9_substrate_carbonmangels' 'cyp2d6_substrate_carbonmangels' 'cyp3a4_substrate_carbonmangels' 
    'half_life_obach' 'clearance_hepatocyte_az' 'clearance_microsome_az' 
    'ld50_zhu' 'herg' 'ames' 'dili' 
)
HIDDEN_DIM_LIST=(320)
for hidden_dim in "${HIDDEN_DIM_LIST[@]}"; do
    echo "Hidden dimension: $hidden_dim"

    for task in "${TASK_LIST[@]}"; do
        echo "Task: $task"
        sleep 1

        CUDA_VISIBLE_DEVICES=$1 graphium-train \
            model=gpspp \
            accelerator=gpu \
            tasks=admet \
            ++constants.seed=0 \
            ++constants.wandb.entity=eddy26 \
            ++constants.wandb.save_dir=null \
            ++constants.wandb.project=graphium \
            ++constants.wandb.tags="['gpspp','admet','scratch']" \
            ++constants.norm='layer_norm' \
            ++constants.raise_train_error=False \
            ++constants.detect_anomaly=False \
            ++architecture.pre_nn.out_dim=${hidden_dim} \
            ++architecture.pre_nn.hidden_dims=${hidden_dim} \
            ++architecture.gnn.layer_kwargs.mpnn_kwargs.in_dim=${hidden_dim} \
            ++architecture.gnn.layer_kwargs.mpnn_kwargs.out_dim=${hidden_dim} \
            ++architecture.gnn.in_dim=${hidden_dim} \
            ++architecture.gnn.hidden_dims=${hidden_dim} \
            ++architecture.gnn.out_dim=${hidden_dim} \
            ++constants.task=$task \
            ++datamodule.args.tdc_benchmark_names=$task \
            ++architecture.task_heads.${task}.hidden_dims=${hidden_dim} \
            ++accelerator.float32_matmul_precision=high \
            ++architecture.gnn.layer_kwargs.precision=32 \
            ++trainer.trainer.precision=32 
    done
done

