# RL from AI Feedback

This RLAIF codebase is mainly adapted from the from the [LLaVA-RLHF](https://github.com/llava-rlhf/LLaVA-RLHF.git), which is adapted from the [SALMON](https://github.com/Edward-Sun/SALMON) codebase.

## 0. Setup

Please refer to [`llava_setup`](../llava_setup) for instructions on how to set up the customized llava package.

Additionally, you **should** run the following command to make sure the versions of some essential packages are correct:

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install deepspeed==0.9.3
pip install peft==0.4.0
pip install transformers==4.31.0
pip install bitsandbytes==0.41.0
pip install datasets
```

## 1. Training the Reward Model

**Note**: For both 7b and 13b policy models, we use the same 13b reward model.


## 2. Initialize the Policy Model


## 3. Training the RL Model with PPO

