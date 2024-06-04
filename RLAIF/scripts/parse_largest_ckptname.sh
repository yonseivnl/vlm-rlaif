
MODEL_DIR="/dataset/llms/LLaVA_RLHF/LLaVA_Video-RLHF/pretrained"
SFT_MODEL_NAME=llava-v1.5-7b-lora_w_lora_16_sftv2_short1632_and_then_long_rank32_alpha32_lr1e4
CKPTS_DIR=$SFT_MODEL_NAME"_allmodels"
RM_LORA_PATH=$CKPTS_DIR/RM_v2data


largest_number=0
largest_directory=""

echo $MODEL_DIR/$RM_LORA_PATH/
for directory in $MODEL_DIR/$RM_LORA_PATH/checkpoint-*; do
    # Check if the entry is a directory
    if [ -d "$directory" ]; then
        # Extract the number from the directory name
        # number=$(basename "$directory" | sed 's/checkpoint-//')
        number=$(basename "$directory" | sed 's/[^0-9]*//g')
        
        # Compare the number with the largest number found so far
        if [ "$number" -gt "$largest_number" ]; then
            largest_number="$number"
            # largest_directory="$directory"
            largest_directory=$(basename "$directory")
        fi
    fi
done


echo "Largest Directory: $largest_directory"