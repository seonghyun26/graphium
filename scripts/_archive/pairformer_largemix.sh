cd ../

DEVICE=${1:-'0'}
# Pairformer pre-training on LargeMix
# Uses the Pairformer architecture with dual single/pairwise tracks
# adapted from Boltz (https://github.com/jwohlwend/boltz)

CUDA_VISIBLE_DEVICES=$DEVICE graphium-train \
    model=pairformer \
    accelerator=gpu \
    tasks=largemix \
    training=largemix \
    architecture=largemix \
    ++constants.seed=42 \
    ++constants.wandb.entity=eddy26 \
    ++constants.wandb.save_dir=null \
    ++constants.wandb.project=graphium \
    ++constants.wandb.tags="['pairformer','pretrain','largemix']" \
    ++datamodule.args.batch_size_training=32
