cd ../../

CKPT_LIST=(
    # ./model/gcn/toymixrxrx3_small.ckpt
    # ./model/gcn/toymixrxrx3_medium.ckpt
    # ./model/gcn/toymixrxrx3_large.ckpt
    # ./model/gcn/toymixrxrx3_ultra.ckpt
)

for task in "${TASK_LIST[@]}"; do
    for ckpt in "${CKPT_LIST[@]}"; do
        echo $task with $ckpt

        CUDA_VISIBLE_DEVICES=$1 graphium-train \
            model=gcn \
            accelerator=gpu \
            tasks=cell-rxrx3 \
            ++constants.task=cell-rxrx3 \
            +finetuning=cell-rxrx3\
            ++finetuning.task=cell-rxrx3 \
            ++finetuning.pretrained_model=$ckpt \
            ++finetuning.unfreeze_pretrained_depth=16 \
            ++constants.seed=0 \
            ++constants.wandb.entity=eddy26 \
            ++constants.wandb.save_dir=null \
            ++constants.wandb.project=graphium \
            ++constants.wandb.tags="['gcn','finetune','unfree_pretrained','toymixrxrx3']" \
            ++constants.raise_train_error=False 
            
        sleep 1
    done
done
