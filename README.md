# VLM-RLAIF
> [**Tuning Large Multimodal Models for Videos using Reinforcement Learning from AI Feedback**](https://arxiv.org/pdf/2402.03746),            
[Daechul Ahn](https://dcahn12.github.io)<sup>1</sup>,
Yura Choi<sup>1</sup>,
[Youngjae Yu](https://yj-yu.github.io/home/)<sup>1</sup>, 
[Dongyeop Kang](https://dykang.github.io)<sup>2</sup>,
[Jonghyun Choi](https://ppolon.github.io)<sup>1,&dagger;</sup><br>
<sup>1</sup>Yonsei University,
<sup>2</sup>University of Minnesota<br>
<sup>&dagger;</sup>Corresponding Author<br>
[ACL 2024](https://2024.aclweb.org)

> **Abstract:** *Recent advancements in large language models have influenced the development of video large multimodal models (VLMMs). Previous approaches for VLMMs involve Supervised Fine-Tuning (SFT) with instruction-tuned datasets, integrating LLM with visual encoders, and additional learnable parameters. Here, aligning video with text, and vice versa, remains a challenge, primarily due to the insufficient quality and quantity of multimodal instruction-tune data compared to that of text-only. This discrepancy often results in alignments that poorly ground the video content. To address this, we present a novel alignment strategy that employs a multimodal AI system equipped with Reinforcement Learning from AI Feedback (RLAIF), providing self-preference feedback to refine itself and facilitating the alignment of video and text modalities. Our approach uniquely integrates detailed video descriptions as context into a multimodal AI system during preference feedback generation to enrich the understanding of video content, a process we call context-aware reward modeling. Empirical evaluations on various video benchmarks demonstrate that our VLM-RLAIF outperforms existing approaches, including the SFT model.*

## Usage
**Available Soon!**

## Citation
```
@inproceedings{ahnCYKC24,
      author    = {Daechul Ahn and Yura Choi and Youngjae Yu and Dongyeop Kang and Jonghyun Choi},
      title     = {Tuning Large Multimodal Models for Videos using Reinforcement Learning from AI Feedback},
      booktitle = {ACL},
      year      = {2024}
}
```
