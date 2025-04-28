import os
import json
import argparse
import numpy as np

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
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

#添加注释 李东 4/1下午
#得分函数dataset是字符串=数据集名称, predictions对应所有sample对应的自回归生成的字符串结果的列表, answers是精简的标准答案（字符串）的列表, all_classes是什么不清楚
def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):#zip(列表1, 列表2)是返回一个元组的列表
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]#lstrip('\n')是去除开头的换行符
        for ground_truth in ground_truths:#注意.jsonl里的answer:[]这里多了个列表符号，这一行取出列表中元素（即标准答案的字符串）
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)#返回百分比XX%中的XX，且保留两位小数

#添加注释 李东 3/28下午
#arg.model="longchat-v1.5-7b-32k"/"Qwen2.5-7B-Instruct"
if __name__ == '__main__':
    args = parse_args()
    scores = dict()
    if args.e:
        path = f"pred_e/{args.model}/"
    else:
        path = f"pred/{args.model}/"
    all_files = os.listdir(path)#比如pre/longchat-v1.5-7b-32k/
    print("Evaluating on:", all_files)
    for filename in all_files:#比如qasper-512.jsonl、qasper-full.jsonl。一次score的计算规模对应 特定数据集的特定budget的所有sample的结果
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, lengths = [], [], []
        dataset = filename.split('-')[0]#获取数据集名称
        with open(f"{path}{filename}", "r", encoding="utf-8") as f:#把所有行的数据打包为predictions, answers, lengths三个列表
            for line in f:#一行代表一个sample
                data = json.loads(line)
                predictions.append(data["pred"])#自回归decode生成的字符串（AI的答案）
                answers.append(data["answers"])#数据集中自带的标准的精简答案字符串
                all_classes = data["all_classes"]#不知道，evaluation/LongBench/pred/longchat-v1.5-7b-32k文件夹下面的.jsonl里的all_classes全是NULL
                if "length" in data:
                    lengths.append(data["length"])#比token数少一半，可能是单词数
        if args.e:
            score = scorer_e(dataset, predictions, answers, lengths, all_classes)
        else:
            score = scorer(dataset, predictions, answers, all_classes)
        scores[filename] = score
    if args.e:
        out_path = f"pred_e/{args.model}/result.json"
    else:
        out_path = f"pred/{args.model}/result.json"
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
