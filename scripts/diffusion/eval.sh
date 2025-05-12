python train_text_diffusion.py \
--eval_test \
--resume_dir saved_diff_models/roc/2025-05-09_16-45-16 \
--sampling_timesteps 250 \
--num_samples 1000 \
--wandb_name roc_ddim \
--sampler ddpm \
--sampling_schedule cosine \
--dataset_name roc \
--output_dir ./eval_results \
# Need to update resume_dir to the correct path