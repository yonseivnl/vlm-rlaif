import os
import math
import json
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import torch
from torchvision.transforms import Compose, Lambda, ToTensor
from torchvision.transforms.functional import to_pil_image


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(l) for l in f]
    
def save_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f)

def save_jsonl(data, file_path):
    with open(file_path, 'w') as f:
        for d in data:
            f.write(json.dumps(d) + '\n')

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def load_frames(frame_names, num_frames=None):
    frame_names.sort()
    # sample frames
    if num_frames is not None and len(frame_names) != num_frames:
        duration = len(frame_names)
        frame_id_array = np.linspace(0, duration-1, num_frames, dtype=int)
        frame_id_list = frame_id_array.tolist()
    else:
        frame_id_list = range(num_frames)

    results = []
    for frame_idx in frame_id_list:
        frame_name = frame_names[frame_idx]
        results.append(load_image(frame_name))

    return results


def load_video_into_frames(
        video_path,
        video_decode_backend='opencv',
        num_frames=8,
        return_tensor=False,
):
    print("VIDEO PATH !!!", video_path)
    if video_decode_backend == 'decord':
        import decord
        from decord import VideoReader, cpu
        decord.bridge.set_bridge('torch')
        decord_vr = VideoReader(video_path, ctx=cpu(0))
        duration = len(decord_vr)
        frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)
        video_data = decord_vr.get_batch(frame_id_list)
        if return_tensor:
            video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
        else:
            video_data = [to_pil_image(f) for f in video_data]
    elif video_decode_backend == 'frames':
        frames = load_frames([os.path.join(video_path, imname) 
                              for imname in os.listdir(video_path)],
                             num_frames=num_frames)
        video_data = frames
        if return_tensor:
            to_tensor = ToTensor()
            video_data = torch.stack([to_tensor(_) for _ in frames]).permute(1, 0, 2, 3) # (T, C, H, W) -> (C, T, H, W)
    elif video_decode_backend == 'opencv':
        import cv2
        cv2_vr = cv2.VideoCapture(video_path)
        duration = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)
        # frame_id_list = np.linspace(0, duration-5, num_frames, dtype=int)

        video_data = []
        for frame_idx in frame_id_list:
            cv2_vr.set(1, frame_idx)
            ret, frame = cv2_vr.read()
            if not ret:
                raise ValueError(f'video error at {video_path}')
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if return_tensor:
                video_data.append(torch.from_numpy(frame).permute(2, 0, 1))
            else:
                video_data.append(Image.fromarray(frame))
        cv2_vr.release()
        if return_tensor:
            video_data = torch.stack(video_data, dim=1)
    else:
        raise NameError(f'video_decode_backend should specify in (pytorchvideo, decord, opencv, frames) but got {video_decode_backend}')
    return video_data
