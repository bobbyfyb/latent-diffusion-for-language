python train_text_diffusion.py \
--eval_test \
--resume_dir saved_diff_models/commongen/2025-09-23_11-16-22 \
--dataset_name commongen \
--sampling_timesteps 250 \
--num_samples 100000000 \
--wandb_name commongen_ddim \
--sampler ddpm \
--sampling_schedule cosine \
--output_dir eval_results/commongen_num_candidates_1 \
--seq2seq_candidates 1 \
# Need to update resume_dir to the correct path