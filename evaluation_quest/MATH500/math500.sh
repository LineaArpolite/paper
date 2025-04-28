MODELPATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
PRED_DIR=results/math500_pred/dsqwen
EVAL_DIR=results/math500_eval/dsqwen

# MODELPATH="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# PRED_DIR=results/math500_pred/dsllama
# EVAL_DIR=results/math500_eval/dsllama

budget=2048

PRED_DIR="${PRED_DIR}_${budget}"
EVAL_DIR="${EVAL_DIR}_${budget}"
mkdir -p $PRED_DIR
mkdir -p $EVAL_DIR

python math500_pred.py \
    --model_name_or_path $MODELPATH \
    --task_name "math500" --dataset_start_idx 0 --dataset_end_idx 500 \
    --quest --token_budget $budget --chunk_size 16 \
    --pred_dir $PRED_DIR


python math500_eval_new.py \
    --model_name_or_path $MODELPATH \
    --task_name "math500" --dataset_start_idx 0 --dataset_end_idx 500 \
    --pred_dir $PRED_DIR --eval_dir $EVAL_DIR


#卡 6 7
#CUDA_VISIBLE_DEVICES=6,7 bash math500.sh