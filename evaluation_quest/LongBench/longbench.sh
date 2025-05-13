

for max_length in 15500 
do
    #for model in "meta-llama/Llama-3.1-8B-Instruct"
    for model in "Qwen/Qwen2.5-7B-Instruct"
    do
        # for task in "narrativeqa" "qasper" "hotpotqa" "multifieldqa_en" "gov_report" "triviaqa"
        for task in  "gov_report"
        do
            #for budget in 512 1024 2048 4096
            for budget in 1024
            do
                python longbench_pred.py \
                    --model_name_or_path $model --task $task \
                    --quest --token_budget $budget --chunk_size 16 \
                    --max_length $max_length
            done
        done
        python -u longbench_eval.py --model_name_or_path $model
    done
done
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash longbench.sh