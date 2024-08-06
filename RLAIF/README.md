# RL from AI Feedback

This RLAIF codebase is mainly adapted from the from the [LLaVA-RLHF](https://github.com/llava-rlhf/LLaVA-RLHF.git), which is adapted from the [SALMON](https://github.com/Edward-Sun/SALMON) codebase.

## 0. Setup

Please refer to [`llava_setup`](../llava_setup) for instructions on how to set up the customized llava package.

Additionally, you **should** run the following command to make sure the versions of some essential packages are correct:

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install deepspeed==0.9.5
pip install peft==0.4.0
pip install transformers==4.34.0
pip install bitsandbytes==0.41.0
pip install datasets
```

We use following **SFT checkpoints** at huggingface to initialize RM and Policy model. Follow process 1. and 2. to train the whole PPO process from SFT
- [SNUMPR/vlm_sft_video_llava_13b](https://huggingface.co/SNUMPR/vlm_sft_video_llava_13b) -> initialize RM
- [SNUMPR/vlm_sft_video_llava_7b](https://huggingface.co/SNUMPR/vlm_sft_video_llava_7b) -> initialize Policy model

Or, you can use our lora weights of trained RM and Policy model, and directly progress to step 3.
- [SNUMPR/vlm_rm_13b_lora](https://huggingface.co/SNUMPR/vlm_rm_13b_lora)
- [SNUMPR/vlm_policy_init_7b_lora](https://huggingface.co/SNUMPR/vlm_policy_init_7b_lora)
<!-- - [SNUMPR/vlm_rm_video_llava_7b_lora](https://huggingface.co/SNUMPR/vlm_rm_video_llava_7b_lora) -->


## 1. Train the Reward Model
**Note**: For both 7b and 13b policy models, we use the same 13b reward model.
```bash
bash RLAIF/scripts/train_reward_model.sh \
		openai/clip-vit-large-patch14-336 \ # vision tower init
		dataset/videos \ # path to videos
		dataset/RM_13b_v1_dataset_39k.json \ # training pref data
		dataset/RM_13b_v1_dataset_39k.json \ # validation pref data
		SNUMPR/vlm_sft_video_llava_13b \ # sft model
		checkpoints/Video_LLaVA_RM_13b_lora \ # path to save trained reward model
```


## 2. Initialize the Policy Model
```bash
bash RLAIF/scripts/initialize_policy_model.sh \
		openai/clip-vit-large-patch14-336 \ # vision tower init
		dataset/videos \ # path to videos
		dataset/PPO_init.json \ # training data
		SNUMPR/vlm_sft_video_llava_7b \ # sft model
		checkpoints/Video_LLaVA_Policy_Init_7b_lora \ # path to save policy model init
```

## 3. Train the RL Model with PPO
#### Using your trained model
```bash
bash RLAIF/scripts/train_rl_model.sh \
		openai/clip-vit-large-patch14-336 \ # vision tower init
		dataset/videos \ # path to videos
		dataset/RL_data.json \ # training data
		SNUMPR/vlm_sft_video_llava_7b \ # sft model
		checkpoints/Video_LLaVA_Policy_Init_7b_lora \ # path to trained policy model lora
		checkpoints/Video_LLaVA_RM_13b_lora \ # path to trained reward model lora
		checkpoints/Video_LLaVA_RLAIF_7b \ # path to save ppo model
		True \ # use latest checkpoint of reward model
```

#### Using provided RM and policy init
1. Download lora weights for RM & initialized Policy model  
	```bash
	cd checkpoints
	git clone https://huggingface.co/SNUMPR/vlm_policy_init_7b_lora # Clone policy init lora
	git clone https://huggingface.co/SNUMPR/vlm_rm_video_llava_13b_lora  # Clone reward model lora
	```
2. Train w/ PPO
	```bash
	bash RLAIF/scripts/train_rl_model.sh \
			openai/clip-vit-large-patch14-336 \ # vision tower init
			dataset/videos \ # path to videos
			dataset/RL_data.json \ # training data
			SNUMPR/vlm_sft_video_llava_7b \ # sft model
			checkpoints/vlm_policy_init_7b_lora \ # path to trained policy model lora
			checkpoints/vlm_rm_video_llava_13b_lora \ # path to trained reward model lora
			checkpoints/Video_LLaVA_RLAIF_7b \ # path to save ppo model
			False \ # use provided checkpoint of reward model
	```