# Preparing Training & Evaluation Dataset
## 🗃️ Training Dataset
- **Note** Our Dataset is built upon four sources of datasets.
    1. [Video-ChatGPT Video Instruction Dataset](https://github.com/mbzuai-oryx/Video-ChatGPT/blob/main/docs/train_video_chatgpt.md)
        - ActivityNet, WebVid videos
        - 100K instructions
    2. [Video Localized Narratives Dataset](https://github.com/google/video-localized-narratives/blob/main/data_preparation.md)
    3. [How2QA](https://github.com/ych133/How2R-and-How2QA)
    4. [NextQA](https://doc-doc.github.io/docs/nextqa.html)
    5. [WebVid](https://github.com/m-bain/webvid)

&nbsp;
- 📜 **Instructions**: Download all of our video instructions from 🤗 [SNUMPR/vlm_rlaif_datasets](https://huggingface.co/SNUMPR/vlm_rlaif_datasets)
    | Dataset Usage | Filename | Source of Videos |
    |----------|---------------|---------------|
    | SFT (short) | SFT_short.json | All |
    | SFT (long) | SFT_long.json | All |
    | Preference dataset (RM) | RM_13b_v1_dataset_39k.json | ANet |
    | PPO init | PPO_init.json | ANet |
    | RLAIF | RL_data.json | ANet |

&nbsp;

- 🎥 **Videos**: Download source videos following the instructions below, and then extract 50 frames per each video to train the model. 

    1. **Video-ChatGPT Instruction Dataset - ActivityNet videos**:
        - **Frames** (🤗  [SNUMPR/vlm_rlaif_train_anet_frames](https://huggingface.co/datasets/SNUMPR/vlm_rlaif_train_anet_frames)): Our version of preprocessed videos, extracted 50 frames per each video 
        - [Videos](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EatOpE7j68tLm2XAd0u6b8ABGGdVAwLMN6rqlDGM_DwhVA?e=90WIuW): video as mp4 format from original paper
    2. Video Localized Narratives Dataset
        - See [download instructions](https://github.com/google/video-localized-narratives/blob/main/data_preparation.md) for the original dataset
        - Download source video from four datasets; OoPs, OVIS, kinetics400 (UVO), kinetics
        - Extract 50 frames per each video into `OOPs_50frames`, `OVIS_50frames`, `kinetics400_50frames`, `kinetics_50frames`
    3. How2QA
        - See [download instructions](https://github.com/ych133/How2R-and-How2QA) to download videos
        - Extract 50 frames per each video into `how2qa_50frames`
    4. NeXTQA
        - Download [Google Drive](https://drive.google.com/file/d/1jTcRCrVHS66ckOUfWRb-rXdzJ52XAWQH/view) link provided by original authors to download video files
        - Extract 50 frames per each video into `nextqa_50frames`

    5. WebVid
        - Follow the official [WebVid dataset](https://github.com/m-bain/webvid) README to download the videos.
        - Extract 50 frames per each video into `webvid_50frames`


```Shell
# 📁  Training data folder structure
TRAIN_DATA_ROOT # (playground/data/train_dataset in default)
├── instructions
└── videos
    ├── anet_vidchatgpt_50frames
    ├── OOPs_50frames
    ├── OVIS_50frames
    ├── kinetics400_50frames
    ├── kinetics_50frames
    ├── how2qa_50frames
    ├── nextqa_50frames
    └── webvid_50frames
```

```plain text
// Example structure
{
    'id': 'sampleid',
    'src_data': 'original data source',
    'conversations': [
        {'role': 'human', 'value': ''},
        {'role': 'gpt', 'value': ''}
    ]
    'images': [
        'video_dir/image_01.jpg',
        'video_dir/image_02.jpg',
            ...
    ]
}

```
&nbsp;
&nbsp;

## 🗃️ Evaluation Dataset


```Shell
# 📁  Evaluation folder structure
EVAL_DATA_ROOT # (playground/data/eval_dataset in default)
├── zeroshotqa
│   ├── annotations
│   └── frames
│       ├── anet
│       ├── msvd
│       └── msrvtt
└── videogenerativebench
    ├── annotations
    └── frames
```
### Zero-shot QA

- 🤗 [**SNUMPR/vlm_rlaif_eval_datasets**](https://huggingface.co/datasets/SNUMPR/vlm_rlaif_eval_datasets/tree/main/zeroshotqa) Download our preprocessed zero-shot QA benchmark from this link.
- For original videos and test split, follow instructions from [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA/blob/main/TRAIN_AND_VALIDATE.md) to download the Zero-shot QA dataset.


### Video Generative Benchmark
- Download evaluation dataset & videos for zero-shot question answering from [Video-ChatGPT Qualitative Evaluation](https://github.com/mbzuai-oryx/Video-ChatGPT/blob/main/quantitative_evaluation/README.md).
    - [Videos](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EatOpE7j68tLm2XAd0u6b8ABGGdVAwLMN6rqlDGM_DwhVA?e=90WIuW),  [Descriptions](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hanoona_bangalath_mbzuai_ac_ae/EYqblLdszspJkayPvVIm5s0BCvl0m6q6B-ipmrNg-pqn6A?e=QFzc1U) 
    - Extract 50 frames per each videos
