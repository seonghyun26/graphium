cd ../../

# graphium data prepare --config-dir ./my_configs tasks=rxrx3 accelerator=gpu
# graphium data prepare tasks=rxrx3 accelerator=gpu

# graphium-train tasks=rxrx3 accelerator=gpu \
#   ++constants.seed=0 \
#   ++constants.wandb.entity=eddy26 \
#   ++constants.wandb.save_dir=null \
#   ++constants.wandb.project=graphium \
#   ++constants.wandb.tags=['debug','cell','pretrain'] \
#   ++constants.data_dir='/home/shpark/prj-molrepr/graphium/data/graphium/neurips2023/small-dataset'

graphium-train tasks=toymix_rxrx3 accelerator=gpu \
  ++constants.seed=0 \
  ++constants.wandb.entity=eddy26 \
  ++constants.wandb.save_dir=null \
  ++constants.wandb.project=graphium \
  ++constants.wandb.tags=['debug','cell','pretrain'] \
  ++constants.data_dir='/home/shpark/prj-molrepr/graphium/data/graphium/neurips2023/small-dataset' \
  ++constants.datacache_path='../datacache/rxrx3/'