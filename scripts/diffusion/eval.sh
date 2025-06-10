python train_text_diffusion.py \
--eval_test \
--resume_dir saved_diff_models/commongen/2025-06-02_16-22-08 \
--sampling_timesteps 250 \
--num_samples 100000 \
--wandb_name commongen_ddim \
--sampler ddpm \
--sampling_schedule cosine \
--dataset_name commongen \
--output_dir ./eval_results/commongen_v2 \
# Need to update resume_dir to the correct path