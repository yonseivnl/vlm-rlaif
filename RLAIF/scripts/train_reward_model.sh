#!/bin/bash

set -e
set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="$PWD:$PYTHONPATH"
export GPUS_PER_NODE=8
export OMP_NUM_THREADS=8

VISION_TOWER=$1
VIDEO_PATH=$2
PREFERENCE_DATA_PATH=$3
PREFERENCE_EVAL_DATA_PATH=$4
SFT_MODEL_PATH=$5
RM_SAVE_PATH=$6


# TRAINING CONFIG
NUM_EPOCHS=1
LEARNING_RATE=2e-5
BATCH_SIZE=2
GRAD_ACCUMULATION=1

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=$GPUS_PER_NODE \
    RLHF/finetune_lora_rm.py \
    --do_train \
    --do_eval \
    --seed 42 \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULATION \
    --model_name_or_path $SFT_MODEL_PATH \
    --image_folder $VIDEO_PATH/ \
    --vision_tower $VISION_TOWER \
    --learning_rate $LEARNING_RATE \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --freeze_mm_mlp_adapter True \
    --model_max_length 2048 \
    --query_len 1280 \
    --response_len 768 \
    --dataset_path $PREFERENCE_DATA_PATH \
    --eval_dataset_path $PREFERENCE_EVAL_DATA_PATH \
    --dataset_name "none" \
    --eval_dataset_name "none" \
    --eval_size 500 \
    --bits 16 \
    --lora_r 64 \
    --lora_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --output_dir $RM_SAVE_PATH \
    --num_train_epochs $NUM_EPOCHS \
    --group_by_length False \
    --evaluation_strategy "steps" \
    --eval_steps 400 \
    --save_strategy "steps" \
    --save_steps 400 \
    --save_total_limit 5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "constant_with_warmup" \
    --logging_steps 10 \
    --report_to "tensorboard" \
    --ddp_backend "nccl" \
    --bf16 True \
    --ddp_find_unused_parameters False \
    --resume_from_training True \
    --reward_prompt_file "./prompts/fact_rlaif_reward_prompt_video.txt" \
    --image_aspect_ratio 'pad'
