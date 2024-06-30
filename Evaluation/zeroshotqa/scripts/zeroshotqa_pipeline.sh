NEW_PYPTH=$PWD/../..
NEW_PYPTH=$(builtin cd $NEW_PYPTH; pwd)
export PYTHONPATH=$PYTHONPATH:$NEW_PYPTH
DATA_PATH=playground/data

# ================== CHANGE HERE ==================
MODEL_PATH=pretrained/final_models/Video_LLaVA_VLM_RLAIF_lora/adapter_model/lora_policy
MODEL_BASE=pretrained/final_models/Video_LLaVA_SFT_model

OUTPUT_DIR=results/VLM_RLAIF
FRAMES_PATH="playground/data/video_frames"

# TASKNAMES=( anet msrvtt msvd tgif )
TASKNAMES=( anet )
# ================== CHANGE HERE ==================

for TASKNAME in ${TASKNAMES[@]}; do
    bash Evaluation/zeroshotqa/scripts/zeroshotqa_infer.sh \
        $MODEL_PATH \
        $MODEL_BASE \
        $OUTPUT_DIR/zeroshotqa \
        $TASKNAME \
        $FRAMES_PATH/$TASKNAME
    wait
    bash Evaluation/zeroshotqa/scripts/zeroshotqa_eval.sh \
        $OUTPUT_DIR/zeroshotqa \
        $TASKNAME
    wait
done
