#!/bin/bash

set -e
set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MODEL_DIR="/dataset/llms/LLaVA_RLHF/LLaVA_Video-RLHF/pretrained"
export PYTHONPATH="$PWD:$PYTHONPATH"
export GPUS_PER_NODE=8
export OMP_NUM_THREADS=8
export TRANSFORMERS_OFFLINE=1

# ====================================== CHANGE HERE ======================================
VISION_TOWER=$1
VIDEO_PATH=$2
DATA_PATH=$3
SFT_MODEL_PATH=$4
POLICY_INIT_PATH=$5
RM_MODEL_PATH=$6
RLHF_SAVE_PATH=$7
CALC_RM_CKPT=$8

# TRAINING CONFIG
LEARNING_RATE=3e-5
KL_COEF=0.1
EPOCH=1
ROLLOUT_BATCH_SIZE=256
STEP_BATCH_SZIE=128
ROLLOUT_PER_DEVICE_BATCH_SIZE=16
REWARD_MODEL_PER_DEVICE_BATCH_SIZE=8
STEP_PER_DEVICE_BATCH_SIZE=8
NOPTEPOCHS=2

# FACT-RLHF CONFIG
INCOMPLETE_RESPONSE=-8.0
LENGTH_BONUS=-10.0
CORRECT_BONUS=2.0
# ==========================================================================================

# Get Largest RM PATH or use provided checkpoint
if [ "$CALC_RM_CKPT" = true ]; then
    largest_num=0
    RM_CKPT_NAME=""
    for directory in $RM_MODEL_PATH/checkpoint-*; do
        # Check if the entry is a directory
        if [ -d "$directory" ]; then
            # Extract the number from the directory name
            number=$(basename "$directory" | sed 's/[^0-9]*//g')
            # Compare the number with the largest number found so far
            if [ "$number" -gt "$largest_number" ]; then
                largest_number="$number"
                # largest_directory="$directory"
                RM_CKPT_NAME=$(basename "$directory")
            fi
        fi
    done
    RM_MODEL_PATH=$RM_MODEL_PATH/$RM_CKPT_NAME
else
    RM_MODEL_PATH=$RM_MODEL_PATH
fi


torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=$GPUS_PER_NODE \
    finetune_lora_ppo.py \
    --do_train \
    --seed 42 \
    --step_batch_size $STEP_BATCH_SZIE \
    --step_per_device_batch_size $STEP_PER_DEVICE_BATCH_SIZE \
    --rollout_batch_size $ROLLOUT_BATCH_SIZE \
    --rollout_per_device_batch_size $ROLLOUT_PER_DEVICE_BATCH_SIZE \
    --reward_model_per_device_batch_size $REWARD_MODEL_PER_DEVICE_BATCH_SIZE \
    --base_model_name $SFT_MODEL_PATH \
    --policy_model_name_or_path $POLICY_INIT_PATH \
    --reward_model_name_or_path $RM_MODEL_PATH \
    --learning_rate $LEARNING_RATE \
    --init_value_with_reward True \
    --warmup_steps 5 \
    --dataset_path $DATA_PATH \
    --train_splits "train" \
    --output_dir $RLHF_SAVE_PATH \
    --total_epochs $EPOCH \
    --group_by_length False \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 100000 \
    --weight_decay 0.0 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --ddp_backend "nccl" \
    --bf16 True \
    --penalty_reward_value $INCOMPLETE_RESPONSE \
    --length_bonus_score $LENGTH_BONUS \
    --correct_bonus_score $CORRECT_BONUS \
    --relative_stop_token_penalty True \
    --penalize_no_stop_token True \
    --ddp_find_unused_parameters False \
    --resume_from_training True \
    --kl_coef $KL_COEF \
    --max_grad_norm 1.0 \
    --whitening_async_stats "full_batch" \
    --clean_tokens_after_eos True \
    --temperature 1.0 \
    --whiten_rewards False \
    --model_max_length 2048 \
    --query_len 128 \
    --response_len 896 \
    --noptepochs $NOPTEPOCHS \
    --image_folder $VIDEO_PATH \
    --vision_tower different \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --freeze_mm_mlp_adapter True \
    --reward_prompt_file "./prompts/fact_rlaif_reward_prompt_video.txt" \
    --image_aspect_ratio 'pad'