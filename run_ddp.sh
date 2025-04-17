#!/bin/bash

# Setup environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29502
export WORLD_SIZE=2

# Number of GPUs to use (adjust as needed)
NUM_GPUS=$1
MASK_IMG=$2
LORA=$3
FREEZE_LM=$4
LOAD_CHKPT=$5
LOAD_ORIG_FMT=$6
EPOCH=$7
CHKPT_FILE=$8
OUT_DIR=$9

# Conditionally enable --lora
if [ "$LORA" -eq 1 ]; then
    LORA_ARG="--lora"
else
    LORA_ARG=""
fi

# Set MASK_ARG only if MASK_IMG == 1
if [ "$MASK_IMG" -eq 1 ]; then
    MASK_ARG="--mask-img"
else
    MASK_ARG=""
fi

# Conditionally enable --freeze-lm
if [ "$FREEZE_LM" -eq 1 ]; then
    FREEZE_ARG="--freeze-lm"
else
    FREEZE_ARG=""
fi

if [ "$LOAD_CHKPT" -eq 1 ]; then
    LOAD_ARG="--load-checkpoint --checkpoint-file $CHKPT_FILE"
else
    LOAD_ARG=""
fi

if [ "$LOAD_ORIG_FMT" -eq 1 ]; then
    LOAD_ORIG_ARG="--load-orig-format"
else
    LOAD_ORIG_ARG=""
fi

# Launch training with torchrun (DDP)
torchrun --nproc_per_node=$NUM_GPUS --nnodes=1 --node_rank=0 \
    --master-addr=$MASTER_ADDR \
    --master-port=$MASTER_PORT \
    train_ddp.py \
    --batch-size 4 \
    $MASK_ARG \
    $LORA_ARG \
    $FREEZE_ARG \
    $LOAD_ARG \
    --output-dir $OUT_DIR \
    $LOAD_ORIG_ARG \
    --epochs $EPOCH


