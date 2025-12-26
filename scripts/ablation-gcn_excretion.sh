ACTIVATION_LIST=(
    ReLU
    Tanh
    ELU
    SELU
    GELU
    LeakyReLU
    Softplus
    SiLU
    None
)
hidden_dim=696
TASK_LIST=('half_life_obach' 'clearance_hepatocyte_az' 'clearance_microsome_az' )

for task in "${TASK_LIST[@]}"; do
    for activation in "${ACTIVATION_LIST[@]}"; do
        echo "Task $task with activation $activation"
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
            ++constants.wandb.tags="['gcn', 'scratch', 'ablation', 'activation']" \
            ++architecture.pre_nn.hidden_dims=${hidden_dim} \
            ++architecture.pre_nn.out_dim=${hidden_dim} \
            ++architecture.gnn.depth=16 \
            ++architecture.gnn.in_dim=${hidden_dim} \
            ++architecture.gnn.out_dim=${hidden_dim} \
            ++architecture.gnn.hidden_dims=${hidden_dim} \
            ++architecture.graph_output_nn.graph.hidden_dims=${hidden_dim} \
            ++architecture.graph_output_nn.graph.out_dim=${hidden_dim} \
            ++architecture.task_heads.${task}.in_dim=${hidden_dim} \
            ++architecture.task_heads.${task}.hidden_dims=${hidden_dim} \
            ++architecture.task_heads.${task}.activation=${activation} 
        
        sleep 1
    done
done