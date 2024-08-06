NEW_PYPTH=$PWD/../..
NEW_PYPTH=$(builtin cd $NEW_PYPTH; pwd)
export PYTHONPATH=$PYTHONPATH:$NEW_PYPTH
export cache_dir="cache_dir"

# ================== CHANGE HERE ==================
MODEL_PATH=SNUMPR/vlm_rlaif_video_llava_7b
MODEL_BASE=none
OUTPUT_DIR=results/vlm_rlaif_video_llava_7b
ANNOT_PATH=playground/data/eval_dataset/zeroshotqa/annotations
FRAMES_PATH="playground/data/eval_dataset/zeroshotqa/video_frames"
export API_KEY="YOUR OPENAI API KEY HERE"

TASKNAMES=( anet msrvtt msvd )
# ================== CHANGE HERE ==================

for TASKNAME in ${TASKNAMES[@]}; do
    bash Evaluation/zeroshotqa/scripts/zeroshotqa_infer.sh \
        $MODEL_PATH \
        $MODEL_BASE \
        $OUTPUT_DIR/zeroshotqa \
        $TASKNAME \
        $ANNOT_PATH/$TASKNAME"_qa.json" \
        $FRAMES_PATH/$TASKNAME \
        $CHUNKS
    wait
    bash Evaluation/zeroshotqa/scripts/zeroshotqa_eval.sh \
        $OUTPUT_DIR/zeroshotqa \
        $TASKNAME
    wait
done
