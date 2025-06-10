RES_DIR=eval_results/commongen/test/guide2.0
PRED_PATH=$RES_DIR/_seq2seq_test_samples.json
SAVE_PATH=$RES_DIR
EVAL_SPLIT='test'

python evaluation/nlg_eval.py \
--pred_path=$PRED_PATH \
--save_path=$SAVE_PATH \
--data_name='common_gen' \
--eval_split=$EVAL_SPLIT \