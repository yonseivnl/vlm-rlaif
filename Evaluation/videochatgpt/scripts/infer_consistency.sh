NEW_PYPTH=$PWD/../..
NEW_PYPTH=$(builtin cd $NEW_PYPTH; pwd)
export PYTHONPATH=$PYTHONPATH:$NEW_PYPTH

MODEL_PATH=$1
MODEL_BASE=$2
OUTPUT_DIR=$3
TASKNAME=${4:-consistency}
VIDEOCHATGPT_EVAL_PATH=$5
FRAMES_PATH=$6
OUTPUT_DIR=$OUTPUT_DIR/$TASKNAME

GPU_IDS=( 0 1 2 3 4 5 6 7 )
SPLITS=( 0 1 2 3 4 5 6 7 )
N_SPLIT=${#GPU_IDS[@]}

for DEVICE_ID in ${GPU_IDS[@]}; do
    CUDA_VISIBLE_DEVICES=$DEVICE_ID \
    python3 evaluate/video_chatgpt/run_inference_benchmark_consistency.py \
    --model-path $MODEL_PATH \
    --model-base $MODEL_BASE \
    --frames_path $FRAMES_PATH \
    --gt_file $VIDEOCHATGPT_EVAL_PATH/$TASKNAME"_qa.json" \
    --output_dir $OUTPUT_DIR \
    --output_name $N_SPLIT"_${SPLITS[$DEVICE_ID]}" \
    --images \
    --num_frames 50 \
    --rlhf_ckpt \
    --chunks $N_SPLIT \
    --chunk_idx ${SPLITS[$DEVICE_ID]} \
    --resume \
    &
done
wait