

LOG_FILE="dsqwen_run_PG19.log"

# 添加时间戳和CUDA信息
echo "===================== SCRIPT START =====================" >> "$LOG_FILE"
echo "[`date`] Running A.sh" >> "$LOG_FILE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >> "$LOG_FILE"
echo "--------------------------------------------------------" >> "$LOG_FILE"

# 把当前脚本内容也记录到日志里
cat "$0" >> "$LOG_FILE"
echo "--------------------- SCRIPT END -----------------------" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"


python dsqwen_run.py \
    --model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" \
    --task_name "pg19"  \
    --sink_cache --local_window_size 2000 --begin_sink_size 64 \
    --dataset_end_idx 10 \
    --length_thresh 2 \
    --warmup_len 100 \
    --replace_k \
    --replace_v \
    --save_name  "Qwen2.5-Math-1.5B-Instruct_block4_kv"\
    --save_path "results/pg19/dsqwen" \
    | tee -a "$LOG_FILE"


#task="pg19"/"math500"/
#写成cache_type  /home/ubuntu/yibo.han/o1_decoding/inspect_same_token_kv_replacement.py
