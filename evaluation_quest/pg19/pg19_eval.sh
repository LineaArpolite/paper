budget=512

# MODELPATH="Qwen/Qwen2.5-7B-Instruct"
# OUTPUT_DIR=results/pg19_eval/Qwen2


MODELPATH="meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR=results/pg19_eval/Llama3

FULL_OUTPUT_DIR="${OUTPUT_DIR}_${budget}"
mkdir -p $FULL_OUTPUT_DIR



python ppl_eval_lidong.py \
    --model_name_or_path $MODELPATH \
    --output_dir $FULL_OUTPUT_DIR \
    --num_eval_tokens 30000 \
    --quest --token_budget $budget --chunk_size 16 