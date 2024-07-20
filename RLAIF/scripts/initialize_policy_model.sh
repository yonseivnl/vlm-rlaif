#!/bin/bash

set -e
set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="$PWD:$PYTHONPATH"
export GPUS_PER_NODE=8
export OMP_NUM_THREADS=8

# MODEL CONFIG
VISION_TOWER=$1
VIDEO_PATH=$2
DATA_PATH=$3
BASE_MODEL_PATH=$4
POLICY_SAVE_PATH=$5

# TRAINING CONFIG
NUM_EPOCHS=1
LEARNING_RATE=1e-4
BATCH_SIZE=8
GRAD_ACCUMULATION=2

deepspeed \
    finetune_policy_init.py \
    --deepspeed scripts/zero2.json \
    --do_train \
    --do_eval \
    --seed 42 \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps $GRAD_ACCUMULATION \
    --model_name_or_path $BASE_MODEL_PATH \
    --image_folder $VIDEO_PATH \
    --vision_tower $VISION_TOWER \
    --learning_rate $LEARNING_RATE \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --freeze_mm_mlp_adapter True \
    --query_len 1280 \
    --response_len 768 \
    --dataset $DATA_PATH \
    --dataset_format "v1" \
    --eval_size 500 \
    --bits 16 \
    --lora_r 64 \
    --lora_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --output_dir $POLICY_SAVE_PATH \
    --num_train_epochs $NUM_EPOCHS \
    --group_by_length False \
    --evaluation_strategy "steps" \
    --eval_steps 50 \
    --save_strategy "steps" \
    --save_steps 1000000 \
    --save_total_limit 1 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --report_to "tensorboard" \
    --ddp_backend "nccl" \
    --bf16 True \
    --ddp_find_unused_parameters False \
    --resume_from_training True \
    --image_aspect_ratio 'pad'
