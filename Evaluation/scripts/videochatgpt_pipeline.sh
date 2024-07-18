NEW_PYPTH=$PWD/../..
NEW_PYPTH=$(builtin cd $NEW_PYPTH; pwd)
export PYTHONPATH=$PYTHONPATH:$NEW_PYPTH
DATA_PATH=playground/data

# ================== CHANGE HERE ==================
MODEL_PATH=SNUMPR/vlm_rlaif_video_llava_7b
MODEL_BASE=none
OUTPUT_DIR=results/vlm_rlaif_video_llava_7b
export cache_dir=./cache_dir
export API_KEY="YOUR OPENAI API KEY HERE"

TASKNAMES=( temporal )
TASKNAMES=( temporal )
DATA_DIR=/dataset/dcahn/llms/YuraLLM/playground/data/VideoChatGPT_Eval/original_data
FRAMES_PATH=/dataset/dcahn/llms/YuraLLM/playground/data/VideoChatGPT_Eval/Test_Videos
# ================== CHANGE HERE ==================
OUTPUT_DIR=$OUTPUT_DIR/videochatgpt

for TASKNAME in ${TASKNAMES[@]}; do
    bash Evaluation/videochatgpt/scripts/pipeline_$TASKNAME.sh \
        $MODEL_PATH \
        $MODEL_BASE \
        $OUTPUT_DIR \
        $TASKNAME \
        $DATA_DIR \
        $FRAMES_PATH
    wait
done