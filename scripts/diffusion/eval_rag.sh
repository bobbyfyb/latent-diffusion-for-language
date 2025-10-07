DATASET_NAME='dimongen'
TOP_K=5
NUM_CANDIDATES=1

python train_text_diffusion.py \
--eval_test \
--resume_dir saved_diff_models/dimongen/2025-10-04_15-30-11 \
--dataset_name ${DATASET_NAME} \
--sampling_timesteps 250 \
--num_samples 100000000 \
--wandb_name ${DATASET_NAME}_rag_top5_ddim \
--sampler ddpm \
--sampling_schedule cosine \
--output_dir eval_results/${DATASET_NAME}_rag_top${TOP_K}_num_candidates_${NUM_CANDIDATES} \
--seq2seq_candidates 1 \
--is_rag \
--top_k 5 \
# Need to update resume_dir to the correct path