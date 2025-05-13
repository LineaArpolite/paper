# MODELPATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# PRED_DIR=results/math500_pred/dsqwen
# EVAL_DIR=results/math500_eval/dsqwen

MODELPATH="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
PRED_DIR=results/math500_pred/dsllama
EVAL_DIR=results/math500_eval/dsllama





python math500_eval.py \
    --model_name_or_path $MODELPATH \
    --task_name "math500" --dataset_start_idx 0 --dataset_end_idx 500 \
    --pred_dir $PRED_DIR --eval_dir $EVAL_DIR

#使用 CUDA_VISIBLE_DEVICES=1 bash math500eval.sh