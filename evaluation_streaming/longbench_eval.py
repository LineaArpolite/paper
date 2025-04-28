import os
import json
import argparse
import numpy as np
import pickle
from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)
from transformers import AutoTokenizer
    
#添加注释 3/28晚上
#字典里的value是个函数
#longbench.sh里使用的数据集"qasper" "narrativeqa" "hotpotqa" "multifieldqa_en" "gov_report" "triviaqa"
dataset2metric = {
    "narrativeqa": qa_f1_score,#
    "qasper": qa_f1_score,#
    "multifieldqa_en": qa_f1_score,#
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,#
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,#
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,#
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default=None)

    #sink cache
    parser.add_argument("--sink_cache", action='store_true')
    parser.add_argument("--local_window_size", type=int, default=2000)
    parser.add_argument("--begin_sink_size", type=int, default=64)

    return parser.parse_args(args)

#得分函数dataset_name是字符串=数据集名称, predictions对应所有sample对应的自回归生成的字符串结果的列表, answers是精简的标准答案（字符串）的列表, all_classes是什么不清楚
def scorer(dataset_name, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):#zip(列表1, 列表2)是返回一个元组的列表
        score = 0.
        for ground_truth in ground_truths:#注意.jsonl里的answer:[]这里是个列表（表示标准答案可能不止一个），这一行取出列表中元素（即标准答案的字符串）
            score = max(score, dataset2metric[dataset_name](prediction, ground_truth, all_classes=all_classes))#表现最好的标准答案
        total_score += score
    return round(100 * total_score / len(predictions), 2)#返回百分比XX%中的XX，且保留两位小数

def preprocess_strs(strs,dataset_name):
    processed_strs=[]
    for str in strs:
        if dataset_name in ["triviaqa"]:
            str = str.lstrip('\n').split('\n')[0]#lstrip('\n')是去除开头的换行符
        else:
            pass
        processed_strs.append(str)
    return processed_strs

def load_token_list(file_path):
    with open(file_path, "rb") as f:
        token_lists = pickle.load(f)
    return token_lists

def load_answersETC(dataset):
    answers=[]
    all_classes=[]
    length=[]

    from datasets import (
        load_dataset
    )
    
    data = load_dataset("THUDM/LongBench", dataset, split="test")#只用"THUDM/LongBench"的测试集

    for json_obj in data:
        answers.append(json_obj["answers"])#其实json_obj["answers"]是个列表，里面存放了一个/多个标准答案的字符串
        all_classes.append(json_obj["all_classes"])
        length.append(json_obj["length"])

    return answers,all_classes,length
#传入arg.model这个参数
#arg.model="longchat-v1.5-7b-32k"/"Qwen2.5-7B-Instruct"
if __name__ == '__main__':
    args = parse_args()
    scores = dict()
    
    model_name=args.model_name_or_path
    model_name_simple = model_name.split("/")[-1] 

    path = f"results/longbench/{model_name_simple}_{args.begin_sink_size}_{args.local_window_size}/"

    
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    
    
    all_files = os.listdir(path)#比如pre/longchat-v1.5-7b-32k/
    print("Evaluating on:", all_files)
    for filename in all_files:#比如qasper-512.jsonl、qasper-full.jsonl。一次score的计算规模对应 特定数据集的特定budget的所有sample的结果
        if not filename.endswith("pkl"):
            continue

        #获取数据集名称
        dataset_name = filename.split('-')[0]

        #获取token id列表的列表
        file_path = os.path.join(path, filename)
        preds=load_token_list(file_path)

        #转换成字符串列表（跳过特殊字符）
        pred_strs = tokenizer.batch_decode(preds, skip_special_tokens=True)

        #是否需要一些处理
        pred_strs = preprocess_strs(pred_strs,dataset_name)

        answers,all_classes,lengths=load_answersETC(dataset_name)

        # print("预测结果(前五个)---------\n",pred_strs[:5])#打印前5个预测结果，便于debug
        # print("标准答案(前五个)---------\n",answers[:5])
        

        score = scorer(dataset_name, pred_strs, answers, all_classes)

        scores[filename] = score


    out_path = f"results/longbench/{model_name_simple}_{args.begin_sink_size}_{args.local_window_size}/eval.json"
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

# pred存储的结果要像它一样pre/{model_name_simple}/{dataset}-{budget}.pkl或者pre_e/{model_name_simple}/{dataset}-{budget}.pkl