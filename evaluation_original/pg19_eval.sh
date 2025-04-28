
####################################################################
# 三要素：评测方法+模型+算法
# 默认评测30000个token，同quest
# 换模型请更改MODELPATH和OUTPUT_DIR
####################################################################



# MODELPATH="meta-llama/Llama-3.1-8B-Instruct"
# OUTPUT_DIR=results/pg19_eval/Llama3

MODELPATH="Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR=results/pg19_eval/Qwen2


mkdir -p $OUTPUT_DIR


python pg19_eval.py \
    --model_name_or_path $MODELPATH \
    --output_dir $OUTPUT_DIR \
    | tee -a pg19_eval.log
