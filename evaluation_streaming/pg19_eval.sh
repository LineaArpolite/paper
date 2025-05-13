begin_sink_size=16
local_window_size=512


#要改最好这两个都改
# MODELPATH="meta-llama/Llama-3.1-8B-Instruct"
# OUTPUT_DIR=results/pg19_eval/Llama3

MODELPATH="Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR=results/pg19_eval/Qwen2


FULL_OUTPUT_DIR="${OUTPUT_DIR}_${begin_sink_size}_${local_window_size}"
mkdir -p $FULL_OUTPUT_DIR


python pg19_eval.py \
    --model_name_or_path $MODELPATH \
    --output_dir $FULL_OUTPUT_DIR \
    --sink_cache --local_window_size $local_window_size --begin_sink_size $begin_sink_size 