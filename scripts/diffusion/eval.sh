python train_text_diffusion.py \
--eval_test \
--resume_dir saved_diff_models/nq/2025-06-13_15-51-25 \
--sampling_timesteps 250 \
--num_samples 100000000 \
--wandb_name nq_ddim \
--sampler ddpm \
--sampling_schedule cosine \
--output_dir eval_results/nq \
# Need to update resume_dir to the correct path