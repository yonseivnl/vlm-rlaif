# 🤖 VLM-RLAIF (ACL 2024)
> [**Tuning Large Multimodal Models for Videos using Reinforcement Learning from AI Feedback**](https://dcahn12.github.io/projects/RLAIF/),            
[Daechul Ahn](https://dcahn12.github.io)<sup>1,3</sup>,
[Yura Choi](https://yuuraa.github.io)<sup>1,3</sup>,
[Youngjae Yu](https://yj-yu.github.io/home/)<sup>1</sup>, 
[Dongyeop Kang](https://dykang.github.io)<sup>2</sup>,
[Jonghyun Choi](https://ppolon.github.io)<sup>3,&dagger;</sup><br>
<sup>1</sup>Yonsei University,
<sup>2</sup>University of Minnesota,
<sup>3</sup>Seoul National University<br>
<sup>&dagger;</sup>Corresponding Author<br>
[ACL 2024](https://2024.aclweb.org) (To appear)

[![model-checkpoint](https://img.shields.io/badge/Model-RLAIF-blue)](https://huggingface.co/SNUMPR/vlm_rlaif_video_llava_7b)
[![model-checkpoint-sft](https://img.shields.io/badge/Model-SFT-blue)](https://huggingface.co/SNUMPR/vlm_rlaif_video_llava_7b)
[![paper](https://img.shields.io/badge/Paper-Arxiv-green)](https://arxiv.org/abs/2402.03746)

&nbsp;

## 📣 News 
- [Jul 16, 2024]  🎙️ **VLM-RLAIF** has been selected for ***oral*** presentation at **ACL 2024**! See you in Bangkok 🇹🇭
- [Jun 16, 2024] 🔥 Our next work on aligning large video multimodal model, **i-SRT**🚄, is now available on [[arXiv](https://arxiv.org/pdf/2406.11280v1), [code](https://github.com/snumprlab/SRT)]
- [May 31, 2024] 🥳 **VLM-RLAIF** is accepted to **ACL 2024** !

&nbsp;


## 👀 Overview
<!-- <details>
    <summary><b>Abstract</b></summary>
    <i>Recent advancements in large language models have influenced the development of video large multimodal models (VLMMs). Previous approaches for VLMMs involve Supervised Fine-Tuning (SFT) with instruction-tuned datasets, integrating LLM with visual encoders, and additional learnable parameters. Here, aligning video with text, and vice versa, remains a challenge, primarily due to the insufficient quality and quantity of multimodal instruction-tune data compared to that of text-only. This discrepancy often results in alignments that poorly ground the video content. To address this, we present a novel alignment strategy that employs a multimodal AI system equipped with Reinforcement Learning from AI Feedback (RLAIF), providing self-preference feedback to refine itself and facilitating the alignment of video and text modalities. Our approach uniquely integrates detailed video descriptions as context into a multimodal AI system during preference feedback generation to enrich the understanding of video content, a process we call context-aware reward modeling. Empirical evaluations on various video benchmarks demonstrate that our VLM-RLAIF outperforms existing approaches, including the SFT model.</i>
</details> -->
> **Abstract:** *Recent advancements in large language models have influenced the development of video large multimodal models (VLMMs). Previous approaches for VLMMs involve Supervised Fine-Tuning (SFT) with instruction-tuned datasets, integrating LLM with visual encoders, and additional learnable parameters. Here, aligning video with text, and vice versa, remains a challenge, primarily due to the insufficient quality and quantity of multimodal instruction-tune data compared to that of text-only. This discrepancy often results in alignments that poorly ground the video content. To address this, we present a novel alignment strategy that employs a multimodal AI system equipped with Reinforcement Learning from AI Feedback (RLAIF), providing self-preference feedback to refine itself and facilitating the alignment of video and text modalities. Our approach uniquely integrates detailed video descriptions as context into a multimodal AI system during preference feedback generation to enrich the understanding of video content, a process we call context-aware reward modeling. Empirical evaluations on various video benchmarks demonstrate that our VLM-RLAIF outperforms existing approaches, including the SFT model.*

<div align="center">
    <img src="assets/images/rlaif_feedback_teaser.png" alt="" width="98%">
<p>Pipeline of VLM-RLAIF</p>
</div>


## 🗃️ Dataset and Checkpoints
| Model | Size | Checkpoint | corr. | detail. | context | temp. | const. |
|----------|----------|-----------|---|---|---|---|---|
| RLAIF | 7B | [SNUMPR/vlm_rlaif_video_llava_7b](https://huggingface.co/SNUMPR/vlm_rlaif_video_llava_7b)| 3.63 | 3.25 | 4.00 | 3.23 | 3.32 |
| SFT | 7B | [SNUMPR/vlm_sft_video_llava_7b](https://huggingface.co/SNUMPR/vlm_sft_video_llava_7b) | 2.79 | 2.82 | 3.37 | 2.28 | 2.49 |

&nbsp;

## 📊 Evaluation
- Prepare evaluation dataset
    - download evaluation dataset & videos for zero-shot question answering from [VideoChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT.git).
- Run evaluation
    ```bash
        bash Evaluation/zeroshotqa/scripts/zeroshotqa_pipeline.sh
    ```
&nbsp;


## 💻 Training w/ RLAIF
- Refer to the [RLAIF](./RLAIF) folder to train reward model, policy model, and do PPO

&nbsp;

## 🔧 Data Generation
**Available Soon**
&nbsp;



## 📚 Citation
```
@inproceedings{ahnCYKC24,
      author    = {Daechul Ahn and Yura Choi and Youngjae Yu and Dongyeop Kang and Jonghyun Choi},
      title     = {Tuning Large Multimodal Models for Videos using Reinforcement Learning from AI Feedback},
      booktitle = {ACL},
      year      = {2024}
}
```
&nbsp;

## License
GNU GENERAL PUBLIC LICENSE
&nbsp;

## Acknowledgement
- [LLaVA](https://github.com/haotian-liu/LLaVA.git)
- [LLaVA-RLHF](https://github.com/llava-rlhf/LLaVA-RLHF.git)
- [VideoChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT.git)
- [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA.git)
