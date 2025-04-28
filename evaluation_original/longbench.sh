
model="Qwen/Qwen2.5-7B-Instruct" #一个模型对应一个文件夹


# for task in "qasper" "narrativeqa" "hotpotqa" "multifieldqa_en" "gov_report" "triviaqa"
# do
#     python longbench_pred.py \
#         --model_name_or_path $model --task $task
# done

python -u longbench_eval.py --model_name_or_path $model
