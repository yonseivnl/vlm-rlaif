import os
import argparse
from tqdm import tqdm
import argparse
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from transformers import TextStreamer

import os
import shutil
import json
from glob import glob
from Evaluation.infer_utils import get_chunk, load_json, save_json, load_frames, load_jsonl, save_jsonl


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)

    # Define the command-line arguments
    parser.add_argument('--images', action='store_true')
    parser.add_argument('--frames_path', help='Directory containing video files.', required=True)
    parser.add_argument('--num_frames', default=50, type=int)
    parser.add_argument('--gt_file', help='Path to the ground truth file.', required=True)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--output_dir', help='Directory to save the model results JSON.', required=True)
    parser.add_argument('--output_name', help='Name of the file for storing results JSON.', required=True)
    parser.add_argument("--device", type=str, default="cuda")
    
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    parser.add_argument("--rlhf_ckpt", action="store_true", default=False, help="Whether it is form RLHF checkpoint")
    parser.add_argument("--resume", action="store_true", default=False, help="Whether to resume inference")

    parser.add_argument("--chunks", type=int)
    parser.add_argument("--chunk_idx", type=int)

    return parser.parse_args()


def _load_model(args):
    model_name = get_model_name_from_path(args.model_path)
    if args.rlhf_ckpt:
        model_name = args.model_path # FIXME naive solution
        if not os.path.exists(os.path.join(args.model_path, "config.json")):
            shutil.copy(os.path.join(args.model_base, "config.json"), os.path.join(args.model_path, "config.json")) # Copy SFT model's config -> to RLHF folder
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, 
                                                                           model_name, args.load_8bit, args.load_4bit, 
                                                                           device=args.device, is_rlhf_checkpoint=args.rlhf_ckpt)

    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    return model, tokenizer, image_processor, context_len, args.conv_mode


def run_inference(args):
    """
    Run inference on a set of video files using the provided model.

    Args:
        args: Command-line arguments.
    """
    LOG_INTERVAL = 50
    # Initialize the model
    model, tokenizer, image_processor, context_len, args.conv_mode = _load_model(args)
    conv_mode = args.conv_mode
    
    # Load the ground truth file
    gt_contents_all = load_json(args.gt_file)
    gt_contents = get_chunk(gt_contents_all, args.chunks, args.chunk_idx)
    output_file_path = os.path.join(args.output_dir, f"{args.output_name}.jsonl")

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.path.exists(output_file_path) and args.resume:
        output_list = load_jsonl(output_file_path) # Initialize ouptuts with previously predicted
    else:
        output_list = []
    prev_pred_ids = [d['id'] for d in output_list]


    # Iterate over each sample in the ground truth file
    for idx, sample in tqdm(enumerate(gt_contents), total=len(gt_contents)):
        video_name = sample['video_name']
        pid = video_name + "_" + str(idx + args.start_idx)
        if pid in prev_pred_ids:
            continue
        sample_set = sample
        if 'id' in sample_set: sample_set['oid'] = sample_set['id']
        sample_set['id'] = pid
        question = sample['Q']
        answer = sample["A"]

        img_full_path = os.path.join(args.frames_path, video_name)
        full_vidframes_list = glob(img_full_path + '/*')
        full_vidframes_list.sort()
        images = load_frames(full_vidframes_list, args.num_frames)
        image_tensor = process_images(images, image_processor, args)
        if args.images:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
            image_tensor = image_tensor.unsqueeze(0)  
        elif type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        
        sample_set = {k: v for k, v in sample_set.items()}
        sample_set["images"] = img_full_path
        sample_set["question"] = question
        sample_set["answer"] = answer
        
        # Get conversation
        conv = conv_templates[args.conv_mode].copy()
        if images is not None:
            # first message
            if model.config.mm_use_im_start_end:
                question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
            else:
                question = DEFAULT_IMAGE_TOKEN + '\n' + question
            conv.append_message(conv.roles[0], question)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], question)
            # conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        sample_set[f'pred'] = outputs

        output_list.append(sample_set)
        if idx % LOG_INTERVAL == 0:
            save_jsonl(output_list, output_file_path)

    # Save the output list to a JSON file
    save_jsonl(output_list, output_file_path)


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)