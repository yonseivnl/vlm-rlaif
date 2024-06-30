NEW_PYPTH=$PWD/../..
NEW_PYPTH=$(builtin cd $NEW_PYPTH; pwd)
export PYTHONPATH=$PYTHONPATH:$NEW_PYPTH
DATA_PATH=playground/data

# ================== CHANGE HERE ==================
MODEL_PATH=SNUMPR/vlm_rlaif_video_llava_7b
MODEL_BASE=none
OUTPUT_DIR=results/vlm_rlaif_video_llava_7b
FRAMES_PATH="playground/data/video_frames"

TASKNAMES=( anet msrvtt msvd tgif )
# ================== CHANGE HERE ==================

for TASKNAME in ${TASKNAMES[@]}; do
    bash Evaluation/zeroshotqa/scripts/zeroshotqa_infer.sh \
        $MODEL_PATH \
        $MODEL_BASE \
        $OUTPUT_DIR/zeroshotqa \
        $TASKNAME \
        $FRAMES_PATH/$TASKNAME \
        $CHUNKS
    wait
    bash Evaluation/zeroshotqa/scripts/zeroshotqa_eval.sh \
        $OUTPUT_DIR/zeroshotqa \
        $TASKNAME
    wait
done
