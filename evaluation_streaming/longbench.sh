begin_sink_size=16
local_window_size=512

#model="Qwen/Qwen2.5-7B-Instruct" #一个模型对应一个文件夹
model="meta-llama/Llama-3.1-8B-Instruct"


# for task in "qasper" "narrativeqa" "hotpotqa" "multifieldqa_en" "gov_report" "triviaqa"
# do
#     python longbench_pred.py \
#         --model_name_or_path $model --task $task \
#         --sink_cache --local_window_size $local_window_size --begin_sink_size $begin_sink_size

# done

python  longbench_eval.py --model_name_or_path $model \
    --local_window_size $local_window_size --begin_sink_size $begin_sink_size 
