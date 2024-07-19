NUM_TASKS=10

PRED_DIR=$1
TASKNAME=$2

PRED_PATH=$PRED_DIR/$TASKNAME
OUT_JSON=$TASKNAME".json"

python3 Evaluation/combine_preds.py --pred_dir $PRED_PATH
PRED_PATH=$PRED_PATH/infer_all.json

echo $PRED_PATH
python3 Evaluation/zeroshotqa/gpt_eval.py \
--pred_path $PRED_PATH \
--output_json $OUT_JSON \
--api_key $API_KEY \
--num_tasks $NUM_TASKS
