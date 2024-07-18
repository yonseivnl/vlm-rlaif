NUM_TASKS=10
API_KEY="OPENAI KEY HERE"

OUTPUT_DIR=$1
TASKID=$2

TASKNAMES=( correctness detailed_orientation context temporal consistency )
INFERNAMES=( generic generic generic temporal consistency )
TASKNAME="${TASKNAMES[$TASKID-1]}"
INFERFNAME="${INFERNAMES[$TASKID-1]}"

PRED_DIR=$OUTPUT_DIR/$INFERFNAME
PRED_PATH=$PRED_DIR/$INFERFNAME".json"

OUT_JSON=$OUTPUT_DIR/$INFERFNAME/gpt_$TASKNAME".json"
OUTPUT_DIR=$OUTPUT_DIR/$INFERFNAME/gpt_eval/$TASKNAME

if [ $INFERFNAME=generic ]; then
    python3 scripts/eval_script/combine_preds.py --pred_dir $PRED_DIR --infer_fname $INFERFNAME.json
fi

echo $PRED_PATH $OUTPUT_DIR
python3 Evaluation/videochatgpt/evaluate_benchmark_${TASKID}"_"$TASKNAME.py \
--pred_path $PRED_PATH \
--output_dir $OUTPUT_DIR \
--output_json $OUT_JSON \
--api_key $API_KEY \
--num_tasks $NUM_TASKS
