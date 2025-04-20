#!/bin/bash

# Setup environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29502
export WORLD_SIZE=2

# Prevents CUDA from busy-waiting. Reduces resource waste and can help stabilize timeouts
export NCCL_BLOCKING_WAIT=1
# Allows distributed training to gracefully detect and handle errors (like one rank failing), instead of silently hanging.
export NCCL_ASYNC_ERROR_HANDLING=1
# Turns on debug logging for NCCL operations.
export NCCL_DEBUG=INFO
# Increases the timeout (in seconds) for NCCL collectives (default is 180s = 3 minutes).
export NCCL_TIMEOUT=1800

# Number of GPUs to use (adjust as needed)
NUM_GPUS=$1
FEAT=$2
MASK_IMG=$3
LORA=$4
FREEZE_LM=$5
LOAD_CHKPT=$6
LOAD_ORIG_FMT=$7
RESTART=$8
EPOCH=$9
CHK_FREQ=${10}
CHKPT_FILE=${11}
OUT_DIR=${12}

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

if [ "$RESTART" -eq 1 ]; then
    RESTART_ARG="--restart"
else
    RESTART_ARG=""
fi

if [ "$FEAT" -eq 1 ]; then
    FEAT_ARG="--feat bevfusion"
else
    FEAT_ARG="--feat image"
fi

# Launch training with torchrun (DDP)
torchrun --nproc_per_node=$NUM_GPUS --nnodes=1 --node_rank=0 \
    --master-addr=$MASTER_ADDR \
    --master-port=$MASTER_PORT \
    train_ddp.py \
    --batch-size 8 \
    --num-workers 0 \
    $MASK_ARG \
    $LORA_ARG \
    $FREEZE_ARG \
    $LOAD_ARG \
    --output-dir $OUT_DIR \
    $LOAD_ORIG_ARG \
    --epochs $EPOCH \
    $RESTART_ARG \
    $FEAT_ARG \
    --checkpoint-frequency $CHK_FREQ

#Example run: two GPUs, use bev, no masking img, no lora, freeze-lm, no using pretrained checkpoint, not original checkpoint from author, epoch starts from 0, total 2 epochs, check every 1000 steps, the pretrained checkpoint, output folder
#CUDA_VISIBLE_DEVICES=0,2 bash ./run_ddp.sh 2 1 0 0 1 0 0 1 2 1000 /data/patrick/mmml_saving/image_Q_pretrained/latest_model_saved.pth /data/patrick/mmml_saving/bev_Q_pretrained/

#Example run: two GPUs, use img, no masking img, use lora, no freeze-lm, use pretrained checkpoint, not original checkpoint from author, epoch starts from 0, total 2 epochs, check every 1000 steps, the pretrained checkpoint, output folder
#CUDA_VISIBLE_DEVICES=1,2 bash ./run_ddp.sh 2 0 0 1 0 1 0 1 2 1000 /data/patrick/mmml_saving/image_Q_pretrained/latest_model_saved.pth /data/patrick/mmml_saving/image_Q_finetuned/