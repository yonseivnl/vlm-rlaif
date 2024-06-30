from PIL import Image
from io import BytesIO
import base64

import torch
from transformers import StoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from transformers import TextStreamer
from os.path import join


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]
    
    
from PIL import Image
from io import BytesIO
import base64

import torch
from transformers import StoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from transformers import TextStreamer
from os.path import join


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


def get_independent_context_n_captions(vidframes_path_list, model, image_processor, tokenizer, args, num_captions=3, cond='', agg_type='concat'):

    vidframes_path_list = vidframes_path_list[1:-1]
    num_captions = int(num_captions)
    
    if num_captions == 1:
        img_files = [vidframes_path_list[randint(0, len(vidframes_path_list)-1)]]
    elif num_captions >= len(vidframes_path_list):
        img_files = vidframes_path_list
    else:
        L = len(vidframes_path_list)
        if num_captions <= 0 or num_captions > L:
            return "Invalid value of n. Please provide a valid number."
        
        max_gap = (L - 1) // (num_captions - 1)  # Calculate the maximum gap between indices
        img_files = [vidframes_path_list[i * max_gap] for i in range(num_captions)]
    

    outputs = []
    
    # for idx, fname in enumerate([first_img_file, last_img_file]):
    for idx, fname in enumerate(img_files):
        # image load
        image = load_image(fname)
        # Similar operation in model_worker.py
        image_tensor = process_images([image], image_processor, args)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        
        conv = conv_templates[args.conv_mode].copy()
        
        tmp_query = 'Considering the following question, \'' + cond + '\', describe this image concisely and shortly.'
        
        question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + tmp_query
        conv.append_message(conv.roles[0], question)
        image = None
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
        
        output = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        outputs.append(output)
        
    outputs = ' '.join(outputs)
    outputs = outputs.replace('</s>', '') 
        
    if agg_type == 'summ':
        conv = conv_templates[args.conv_mode].copy()
        tmp_query = 'Summarize the following sentences, ' + outputs
        
        question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + tmp_query
        conv.append_message(conv.roles[0], question)
        image = None
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
        # outputs.append(output)
        
    outputs = outputs.replace('</s>', '')    
    
    return outputs


def get_independent_context_captions(vidframes_path_list, model, image_processor, tokenizer, args, num_captions=3, cond='', agg_type='concat'):
    
    first_img_file = vidframes_path_list[1]
    tmp_frm_idx = max(0, int(len(vidframes_path_list)/2))
    tmp_frm_idx = min(len(vidframes_path_list)-1, int(len(vidframes_path_list)/2))
    center_img_file = vidframes_path_list[tmp_frm_idx]
    last_img_file = vidframes_path_list[-2]
    
    outputs = []
    
    for idx, fname in enumerate([first_img_file, center_img_file, last_img_file]):
        # image load
        image = load_image(fname)
        # Similar operation in model_worker.py
        image_tensor = process_images([image], image_processor, args)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        
        conv = conv_templates[args.conv_mode].copy()
        
        tmp_query = 'Considering the following question, \'' + cond + '\', describe this image concisely and shortly.'
        
        question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + tmp_query
        conv.append_message(conv.roles[0], question)
        image = None
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
        
        output = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        outputs.append(output)
        
    outputs = ' '.join(outputs)
    outputs = outputs.replace('</s>', '') 
        
    if agg_type == 'summ':
        conv = conv_templates[args.conv_mode].copy()
        tmp_query = 'Summarize the following sentences, ' + outputs
        
        question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + tmp_query
        conv.append_message(conv.roles[0], question)
        image = None
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
        # outputs.append(output)
        
    outputs = outputs.replace('</s>', '')    
    
    return outputs


def get_dependent_context_captions(vidframes_path_list, model, image_processor, tokenizer, args, num_captions=3, cond='', agg_type='concat'):
    
    first_img_file = vidframes_path_list[1]
    tmp_frm_idx = max(0, int(len(vidframes_path_list)/2))
    tmp_frm_idx = min(len(vidframes_path_list)-1, int(len(vidframes_path_list)/2))
    center_img_file = vidframes_path_list[tmp_frm_idx]
    last_img_file = vidframes_path_list[-2]
    
    outputs = []
    
    for idx, fname in enumerate([first_img_file, center_img_file, last_img_file]):
        # image load
        image = load_image(fname)
        # Similar operation in model_worker.py
        image_tensor = process_images([image], image_processor, args)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        
        conv = conv_templates[args.conv_mode].copy()
        
        if idx == 0:
            tmp_query = 'Considering the following question, \'' + cond + '\', describe this image concisely and shortly.'
        else:
            tmp_query = 'Considering the following question and sentences\'' + cond + ' and ' + output + '\', describe this image concisely and shortly.'
        
        tmp_query = tmp_query.replace('</s>', '')
        question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + tmp_query
        conv.append_message(conv.roles[0], question)
        image = None
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
        
        output = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        output.replace('</s>', '')
        outputs.append(output)
        
    outputs = ' '.join(outputs)
    outputs = outputs.replace('</s>', '') 
        
    if agg_type == 'summ':
        conv = conv_templates[args.conv_mode].copy()
        tmp_query = 'Summarize the following sentences, ' + outputs
        
        # question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + tmp_query
        conv.append_message(conv.roles[0], tmp_query)
        image = None
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
                do_sample=True,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
            
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        # outputs.append(output)
        
    outputs = outputs.replace('</s>', '')    
    
    return outputs


def get_dependent_context_n_captions(vidframes_path_list, model, image_processor, tokenizer, args, num_captions=3, cond='', agg_type='concat'):
    
    vidframes_path_list = vidframes_path_list[1:-1]
    num_captions = int(num_captions)
    
    if num_captions == 1:
        img_files = [vidframes_path_list[randint(0, len(vidframes_path_list)-1)]]
    elif num_captions >= len(vidframes_path_list):
        img_files = vidframes_path_list
    else:
        L = len(vidframes_path_list)
        if num_captions <= 0 or num_captions > L:
            return "Invalid value of n. Please provide a valid number."
        
        max_gap = (L - 1) // (num_captions - 1)  # Calculate the maximum gap between indices
        img_files = [vidframes_path_list[i * max_gap] for i in range(num_captions)]
    
    
    # first_img_file = vidframes_path_list[1]
    # tmp_frm_idx = max(0, int(len(vidframes_path_list)/2))
    # tmp_frm_idx = min(len(vidframes_path_list)-1, int(len(vidframes_path_list)/2))
    # center_img_file = vidframes_path_list[tmp_frm_idx]
    # last_img_file = vidframes_path_list[-2]
    
    outputs = []
    
    for idx, fname in enumerate([first_img_file, center_img_file, last_img_file]):
        # image load
        image = load_image(fname)
        # Similar operation in model_worker.py
        image_tensor = process_images([image], image_processor, args)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        
        conv = conv_templates[args.conv_mode].copy()
        
        if idx == 0:
            tmp_query = 'Considering the following question, \'' + cond + '\', describe this image concisely and shortly.'
        else:
            tmp_query = 'Considering the following question and sentences\'' + cond + ' and ' + output + '\', describe this image concisely and shortly.'
        
        tmp_query = tmp_query.replace('</s>', '')
        question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + tmp_query
        conv.append_message(conv.roles[0], question)
        image = None
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
        
        output = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        output.replace('</s>', '')
        outputs.append(output)
        
    outputs = ' '.join(outputs)
    outputs = outputs.replace('</s>', '') 
        
    if agg_type == 'summ':
        conv = conv_templates[args.conv_mode].copy()
        tmp_query = 'Summarize the following sentences, ' + outputs
        
        # question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + tmp_query
        conv.append_message(conv.roles[0], tmp_query)
        image = None
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
                do_sample=True,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
            
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        # outputs.append(output)
        
    outputs = outputs.replace('</s>', '')    
    
    return outputs


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False