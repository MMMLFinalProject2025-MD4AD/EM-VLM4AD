#!/bin/bash

# Setup environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29501
export WORLD_SIZE=2

# Number of GPUs to use (adjust as needed)
NUM_GPUS=1

# Launch training with torchrun (DDP)
torchrun --nproc_per_node=$NUM_GPUS --nnodes=1 --node_rank=0 \
    --master-addr=$MASTER_ADDR \
    --master-port=$MASTER_PORT \
    train_ddp.py \
    --batch-size 8 \
    --num-workers 0 \
    --checkpoint-frequency 10

