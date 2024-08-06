NEW_PYPTH=$PWD/../..
NEW_PYPTH=$(builtin cd $NEW_PYPTH; pwd)
export PYTHONPATH=$PYTHONPATH:$NEW_PYPTH

# MODEL_NAME=$1
# CKPT_NAME=$2
# source scripts/model_paths/$MODEL_NAME
# TASKNAME=anet
MODEL_PATH=$1
MODEL_BASE=$2
OUTPUT_DIR=$3
TASKNAME=$4
ANNOT_PATH=$5
FRAMES_PATH=$6

PRED_DIR=$3/$TASKNAME
mkdir -p $PRED_DIR

# GPU_IDS=( 0 1 2 3 4 5 6 7 )
GPU_IDS=( 7 )
SPLITS=( 0 1 2 3 4 5 6 7 )
N_SPLIT=8

for DEVICE_ID in ${GPU_IDS[@]}; do
    CUDA_VISIBLE_DEVICES=$DEVICE_ID python3 Evaluation/zeroshotqa/qa_infer.py \
    --model-path $MODEL_PATH \
    --model-base $MODEL_BASE \
    --gt_file_qa $ANNOT_PATH \
    --chunks $N_SPLIT \
    --chunk_idx ${SPLITS[$DEVICE_ID]} \
    --output_dir $PRED_DIR \
    --output_name $N_SPLIT"_${SPLITS[$DEVICE_ID]}" \
    --images \
    --frames_path $FRAMES_PATH \
    --num_frames 50 \
    --resume \
    &
done
wait

