# MODELPATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# PRED_DIR=results/math500_pred/dsqwen
# EVAL_DIR=results/math500_eval/dsqwen

for local_window_size in 512 2048
do
    MODELPATH="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    PRED_DIR=results/math500_pred/dsllama
    EVAL_DIR=results/math500_eval/dsllama


    begin_sink_size=16
    local_window_size=512


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
done
#  CUDA_VISIBLE_DEVICES=7 bash math5002temp.sh