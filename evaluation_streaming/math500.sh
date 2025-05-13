MODELPATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
PRED_DIR=results/math500_pred/dsqwen
EVAL_DIR=results/math500_eval/dsqwen

# MODELPATH="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# PRED_DIR=results/math500_pred/dsllama
# eval_dir=results/math500_eval/dsllama


begin_sink_size=16
local_window_size=1024


PRED_DIR="${PRED_DIR}_${begin_sink_size}_${local_window_size}"
EVAL_DIR="${EVAL_DIR}_${begin_sink_size}_${local_window_size}"
mkdir -p $PRED_DIR
mkdir -p $EVAL_DIR

python math500_pred.py \
    --model_name_or_path $MODELPATH \
    --cache_type "sink" --local_window_size $local_window_size --begin_sink_size $begin_sink_size \
    --task_name "math500" --dataset_start_idx 0 --dataset_end_idx 500\
    --pred_dir $PRED_DIR

python math500_eval.py \
    --model_name_or_path $MODELPATH \
    --task_name "math500" --dataset_start_idx 0 --dataset_end_idx 500\
    --pred_dir $PRED_DIR --eval_dir $EVAL_DIR

# 卡6 CUDA_VISIBLE_DEVICES=6 bash math500.sh