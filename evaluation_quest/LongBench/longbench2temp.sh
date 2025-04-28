
# model="Qwen/Qwen2.5-7B-Instruct" #一个模型对应一个文件夹，里面有各种dataset - budget.pkl
model="meta-llama/Llama-3.1-8B-Instruct"

for task in "narrativeqa" #"qasper" "hotpotqa" "multifieldqa_en" "gov_report" "triviaqa"
do
    for budget in 512 1024 2048 4096
    do
        python longbench_pred.py \
            --model_name_or_path $model --task $task \
            --quest --token_budget $budget --chunk_size 16 
    done
done

python -u longbench_eval.py --model_name_or_path $model
