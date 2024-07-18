MODEL_PATH=$1
MODEL_BASE=$2
OUTPUT_DIR=$3
TASKNAME=$4
DATA_DIR=$5
FRAMES_PATH=$6

# Generic inference
bash Evaluation/videochatgpt/scripts/infer_general.sh \
    $MODEL_PATH \
    $MODEL_BASE \
    $OUTPUT_DIR \
    $TASKNAME \
    $DATA_DIR \
    $FRAMES_PATH
wait

bash Evaluation/videochatgpt/scripts/gpt_eval.sh $OUTPUT_DIR 3
wait