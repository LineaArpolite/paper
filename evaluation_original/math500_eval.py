import os
import argparse
import datasets
import pickle
import json
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from math_equivalence import is_equiv
import re
import numpy as np
from collections import Counter
import string
import time




def extract_answer(output, mode='gen'):
    extracted_text = ''
    if mode == 'codegen':
        # Extract the code between ```python and ```
        pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(pattern, output, re.DOTALL | re.IGNORECASE)
        if matches:
            extracted_text = matches[-1].strip()  # Take the last match
    elif mode == 'infogen':
        # Extract content after **Final Information** or **Modified Reasoning Steps**
        pattern_info = "\n**Final Information**"
        pattern_step = "\n**Modified Reasoning Steps**"
        if pattern_info in output:
            extracted_text = output.split(pattern_info)[-1].replace("\n","").strip("```").strip()
        elif pattern_step in output:
            extracted_text = output.split(pattern_step)[-1].strip("```").strip()
        else:
            extracted_text = "No helpful information found."
    else:
        # Existing extraction logic for 'gen' and 'choose' modes
        pattern = r'\\boxed\{(.*)\}'
        matches = re.findall(pattern, output)
        if matches:
            extracted_text = matches[-1]  # Take the last match
            if mode in ['choose', 'qa']:
                # Handle 'choose' mode
                inner_pattern = r'\\text\{(.*)\}'
                inner_matches = re.findall(inner_pattern, extracted_text)
                if inner_matches:
                    extracted_text = inner_matches[-1]  # Take the last match
                extracted_text = extracted_text.strip("()")
    return extracted_text


def normalize_answer(text):
    text = text.lower()
    text = " ".join(text.strip().split())
    return text

def normalize_answer_qa(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.strip().split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def evaluate_predictions(output, labeled_answer, mode='gen'):
    final_metric = {"is_valid_answer": False, "acc": 0, "em": 0, "f1": 0, 'math_equal': 0}
    pred_answer = extract_answer(output, mode=mode)
    if pred_answer != '':
        final_metric["is_valid_answer"] = True

    if mode == 'qa':
        normalized_pred_answer = normalize_answer_qa(pred_answer)
        for answer in labeled_answer:
            normalized_ground_truth = normalize_answer_qa(answer)
            em = int(normalized_pred_answer == normalized_ground_truth)
            acc = int(normalized_ground_truth in normalized_pred_answer)

            prediction_tokens = normalized_pred_answer.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ["em", "acc", "f1"]:
                final_metric[k] = max(eval(k), final_metric[k])

    else:
        normalized_pred_answer = normalize_answer(pred_answer)
        normalized_ground_truth = normalize_answer(labeled_answer)

        em = int(normalized_pred_answer == normalized_ground_truth)
        acc = int(normalized_ground_truth in normalized_pred_answer)
    
        prediction_tokens = normalized_pred_answer.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            f1 = 0
        else:
            precision = 1.0 * num_same / len(prediction_tokens) if len(prediction_tokens) > 0 else 0
            recall = 1.0 * num_same / len(ground_truth_tokens) if len(ground_truth_tokens) > 0 else 0
            if (precision + recall) == 0:
                f1 = 0
            else:
                f1 = (2 * precision * recall) / (precision + recall)

        final_metric["em"] = em
        final_metric["acc"] = acc
        final_metric["f1"] = f1

        final_metric["math_equal"] = is_equiv(normalized_pred_answer, normalized_ground_truth)

    # print(em, acc, f1, normalized_pred_answer, '|', normalized_ground_truth)
    return final_metric, pred_answer


def run_evaluation(filtered_data, input_list, output_list, dataset_name, output_dir, total_time, split, apply_backoff=False, model_path=None, tokens_load_path=None):
    # Existing evaluation for other datasets
    avg_em, avg_acc, avg_f1, avg_math = [], [], [], []
    num_valid_answer = 0

    # If the dataset is GPQA, track metrics per domain
    domain_metrics = {}

    for item, input_prompt, result in zip(filtered_data, input_list, output_list):
        
        
        
        if type(result) == str:
            item['Output'] = result
        else:
            item['Output'] = result.outputs[0].text
        if dataset_name in ['gpqa', 'medmcqa']:
            labeled_answer = item["Correct Choice"]
            # labeled_choice_answer = item["Correct Answer"]
            mode = 'choose'
        elif dataset_name in ['MATH', 'math500', 'aime', 'amc']:
            labeled_answer = item["answer"]
            mode = 'gen'
        elif dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki']:
            labeled_answer = item["answer"]
            mode = 'qa'
        elif dataset_name in ['pubhealth']:
            labeled_answer = item["answer"]
            mode = 'choose'
        else:
            raise ValueError(f"Unknown dataset_name: {dataset_name}")

        metric, pred_answer = evaluate_predictions(output=item['Output'], labeled_answer=labeled_answer, mode=mode)
        item['Pred_Answer'] = pred_answer
        item['Metrics'] = metric
        item['Question'] = input_prompt

        # Determine the validity of the predicted answer
        my_method_valid = (pred_answer != '' and not (mode == 'choose' and dataset_name == 'gpqa' and len(pred_answer) > 1))

        avg_em.append(metric['em'])
        avg_acc.append(metric['acc'])
        avg_f1.append(metric['f1'])
        avg_math.append(metric['math_equal'])

        if my_method_valid:
            num_valid_answer += 1

        # If the dataset is GPQA, attempt to track metrics per domain
        if dataset_name == 'gpqa':
            domain = item.get("High-level domain", "Unknown")
            if domain not in domain_metrics:
                domain_metrics[domain] = {'em': [], 'acc': [], 'f1': [], 'math_equal': [], 'num_valid_answer': 0, 'total_num': 0}
            domain_metrics[domain]['total_num'] += 1
            domain_metrics[domain]['em'].append(metric['em'])
            domain_metrics[domain]['acc'].append(metric['acc'])
            domain_metrics[domain]['f1'].append(metric['f1'])
            domain_metrics[domain]['math_equal'].append(metric['math_equal'])
            if my_method_valid:
                domain_metrics[domain]['num_valid_answer'] += 1

    t = time.localtime()
    result_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.json'
    metrics_json_name = f'{split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.metrics.json'

    # Compute overall metrics
    overall_results = {
        'model_path': model_path,
        'tokens_load_path': tokens_load_path,
        'em': np.mean(avg_em) if len(avg_em) > 0 else 0.0,
        'acc': np.mean(avg_acc) if len(avg_acc) > 0 else 0.0,
        'f1': np.mean(avg_f1) if len(avg_f1) > 0 else 0.0,
        'math_equal': np.mean(avg_math) if len(avg_em) > 0 else 0.0,
        'num_valid_answer': f'{num_valid_answer} of {len(input_list)}',
        'query_latency': f'{(total_time / len(input_list) * 1000):.0f} ms',
    }

    # If the dataset is GPQA, output average metrics per domain
    domain_avg_metrics = {}
    if dataset_name == 'gpqa':
        for dm, m in domain_metrics.items():
            domain_avg_metrics[dm] = {
                'em': np.mean(m['em']) if len(m['em']) > 0 else 0,
                'acc': np.mean(m['acc']) if len(m['acc']) > 0 else 0,
                'f1': np.mean(m['f1']) if len(m['f1']) > 0 else 0,
                'math_equal': np.mean(m['math_equal']) if len(m['math_equal']) > 0 else 0,
                'num_valid_answer': f'{m["num_valid_answer"]} of {m["total_num"]}'
            }

    # 保存总体和分domain的指标
    final_metrics = {'overall': overall_results}
    if dataset_name == 'gpqa':
        final_metrics['per_domain'] = domain_avg_metrics

    t = time.localtime()
    result_json_name = f'{dataset_name}_{split}-{t.tm_mon:02}.{t.tm_mday:02}.{t.tm_hour:02}:{t.tm_min:02}:{t.tm_sec:02}.json'
    metrics_json_name = f'{dataset_name}_{split}-{t.tm_mon:02}.{t.tm_mday:02}.{t.tm_hour:02}:{t.tm_min:02}:{t.tm_sec:02}.metrics.json'
    if apply_backoff:
        result_json_name = output_dir
        metrics_json_name = output_dir.replace('.json', '.metrics.backoff.json')

    # Save prediction results and metrics
    with open(os.path.join(output_dir, result_json_name), mode='w', encoding='utf-8') as json_file:
        json.dump(filtered_data, json_file, indent=4, ensure_ascii=False)

    with open(os.path.join(output_dir, metrics_json_name), mode='w', encoding='utf-8') as json_file:
        json.dump(final_metrics, json_file, indent=4, ensure_ascii=False)



def parse_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-32B-Instruct")#请输入全称
    parser.add_argument("--max_tokens", type=int, default=8192) # math500任务中用到了
    parser.add_argument("--atten_sparsity", type=float, default=0.9)

    # dataset
    parser.add_argument("--task_name", type=str, default="math500")
    parser.add_argument("--dataset_name", type=str, default="qasper")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--dataset_start_idx", type=int, default=0)
    parser.add_argument("--dataset_end_idx", type=int, default=None)

    # load
    parser.add_argument("--load_token_path", type=str)
    
    # KVcache
    parser.add_argument("--cache_type", type=str,choices=["sink","dynamic","none"], default="none")
    
    parser.add_argument("--local_window_size", type=int, default=512)
    parser.add_argument("--begin_sink_size", type=int, default=16)
    
    
    parser.add_argument("--pred_dir", type=str,required=True)
    parser.add_argument("--eval_dir", type=str,required=True)
    

    args = parser.parse_args()
    return args

def math500_eval(model_name,dataset_name,pred_dir,eval_dir, split="test", dataset_start_idx=0, dataset_end_idx=None):#answer.pkl，再从本地 加载 Question字符串
    
    model_name_simple = model_name.split("/")[-1]
    
    if dataset_name == 'math500':
        data_path = f'./data/MATH500/{split}.json'
    elif dataset_name == 'MATH':
        data_path = f'./data/MATH/{split}.json'
    elif dataset_name.lower() == 'amc':
        data_path = f'./data/AMC/amc_2022_2023.json'
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    # Load main output data
    with open(data_path, mode='r', encoding='utf-8') as file:#读取的是json文件
        data = json.load(file)
    if dataset_end_idx == None:
        dataset_end_idx = len(data)
        
    filtered_data=data[dataset_start_idx:min(len(data),dataset_end_idx)]
    input_list = [item['Question'] for item in data[dataset_start_idx:min(len(data),dataset_end_idx)]]
    

    output_list = []
    #要评测[{dataset_start_idx},{dataset_end_idx})，就必须有这个名字的jsonl文件
    pred_path=os.path.join(pred_dir, f"{model_name_simple}-[{dataset_start_idx},{dataset_end_idx}).jsonl")
    
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():  # 跳过空行
                continue
            data = json.loads(line)
            output_list.append(data["output_text"])
        

    # Estimate total_time (if available). Here, set to 0 as a placeholder.
    total_time = 0  # Modify if timing information is available

    # Run evaluation
    os.makedirs(eval_dir, exist_ok=True)
    run_evaluation(
        filtered_data=filtered_data,
        input_list=input_list,
        output_list=output_list,
        dataset_name=dataset_name,
        output_dir=eval_dir,
        total_time=total_time,#0
        split=split,#default="test"
        model_path=model_name,#模型全称
        tokens_load_path=None
    )

    print(f"Evaluation completed. Metrics saved to {eval_dir}")



@torch.no_grad()
def main():
    args = parse_args()
    
    model_name = args.model_name_or_path



    args.dataset_name = "math500"
    
    
    os.makedirs(args.eval_dir, exist_ok=True)

        
    math500_eval(model_name, args.task_name, pred_dir=args.pred_dir,eval_dir=args.eval_dir ,dataset_start_idx=args.dataset_start_idx,dataset_end_idx=args.dataset_end_idx)

if __name__ == "__main__":
    main()

