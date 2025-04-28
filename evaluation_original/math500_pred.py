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


import random
import numpy as np
import torch
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)




@torch.no_grad()
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


def run_math500_evaluation(filtered_data, input_list, output_list, dataset_name, output_dir, total_time, split, apply_backoff=False, model_path=None, tokens_load_path=None):
    """
    请将
    filtered_data：喂给模型的sample对应的原始sample的原始数据列表
    input_list：喂给模型的问题字符串列表
    output_list：模型推理的回答字符串列表
    三者的id对齐
    """
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


def load_token_lists(file_path):
    with open(file_path, "rb") as f:
        token_lists = pickle.load(f)
    return token_lists

def get_prompt_list(tokenizer,dataset_name,dataset_split="test", dataset_start_idx=0, dataset_end_idx=None):
    
    if dataset_name in ["MATH", "amc", "math500"]:
        if dataset_name == "MATH":
            dataset_path = "hendrycks/competition_math"
        elif dataset_name == "amc":
            dataset_path = "AI-MO/aimo-validation-amc"
            dataset_split = "train"
        elif dataset_name == "math500":
            dataset_path = "HuggingFaceH4/MATH-500"
            dataset_split = "test"
        dataset = datasets.load_dataset(dataset_path, trust_remote_code=True)
        dataset_split = dataset[dataset_split]

        question_list = [data['problem'] for data in dataset_split]
        # system_prompt = "You are a helpful and harmless assistant. You should think step-by-step."
        system_prompt = "Please answer the following math question. You should think step-by-step.\nYou should provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n"

    elif dataset_name in ["qasper", "narrativeqa", "hotpotqa", "multifieldqa_en", "gov_report", "triviaqa"]:
        dataset = datasets.load_dataset("THUDM/LongBench", dataset_name, split="test")
        with open ("config/dataset2prompt.json") as prompt_file:
            dataset2prompt = json.load(prompt_file)
        prompt_format = dataset2prompt[dataset_name]
        question_list = [prompt_format.format(**json_obj) for json_obj in dataset]
        system_prompt = "Please follow the instructions below."

    elif dataset_name == "MATHQA":
        dataset = datasets.load_dataset("math_qa", split='test')
        question_list = [data['question'] for data in dataset]
        system_prompt = "You are a helpful and harmless assistant. Please answer the following math question. You should think step-by-step."

    elif dataset_name == "humaneval":
        dataset = datasets.load_dataset("openai/openai_humaneval", split='test')
        question_list = [data['prompt'] for data in dataset]
        system_prompt = "You are a helpful and harmless assistant. Please complete the code. You should think step-by-step."


    elif dataset_name == "govreport":
        dataset = datasets.load_dataset("ccdv/govreport-summarization", split='test')
        question_list = [data['report'] for data in dataset]
        system_prompt = "You are a helpful and harmless assistant. Please summarize the text as detailed as possible."
        
    else:
        raise NotImplementedError


    def complete_system_prompt(tokenizer, system_prompt, prompt):
        if tokenizer.chat_template:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            text = system_prompt + "\n" + prompt # 字符串拼接
        return text
        
    prompt_list = [complete_system_prompt(tokenizer, system_prompt, q) for q in question_list]
    if dataset_end_idx == None:
        dataset_end_idx = len(prompt_list)
    prompt_list = prompt_list[dataset_start_idx:dataset_end_idx]
    
    return prompt_list

def parse_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument("--model_name_or_path", type=str, required=True)#请输入全称
    parser.add_argument("--max_tokens", type=int, default=8192) 

    # dataset
    parser.add_argument("--task_name", type=str, default="math500")
    parser.add_argument("--dataset_name", type=str, default="qasper")
    parser.add_argument("--dataset_split", type=str, default="test")
    parser.add_argument("--dataset_start_idx", type=int, default=0)
    parser.add_argument("--dataset_end_idx", type=int, default=None)

    # KVcache
    parser.add_argument("--cache_type", type=str,choices=["sink","dynamic","none"], default="none")
    
    parser.add_argument("--local_window_size", type=int, default=512)
    parser.add_argument("--begin_sink_size", type=int, default=16)
    
    
    parser.add_argument("--pred_dir", type=str,required=True)#跑math500时用的

    parser.add_argument("--save_name", type=str, default="")

    args = parser.parse_args()
    return args

def math500_eval(model_name,dataset_name,tokens_load_path ,save_dir, split="test", dataset_start_idx=0, dataset_end_idx=None):#answer.pkl，再从本地 加载 Question字符串
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
        token_lists = load_token_lists(tokens_load_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        for tlist in token_lists:
            _output_text = tokenizer.decode(tlist)
            output_list.append(_output_text)
        print("Load batch size:", len(output_list))

        

        # Estimate total_time (if available). Here, set to 0 as a placeholder.
        total_time = 0  # Modify if timing information is available

        # Run evaluation
        os.makedirs(save_dir, exist_ok=True)
        run_math500_evaluation(
            filtered_data=filtered_data,
            input_list=input_list,
            output_list=output_list,
            dataset_name=dataset_name,
            output_dir=save_dir,
            total_time=total_time,#0
            split=split,#default="test"
            model_path=model_name,#模型全称
            tokens_load_path=tokens_load_path #answer.pkl的地址
        )

        print(f"Evaluation completed. Metrics saved to {save_dir}")



@torch.no_grad()
def main():
    seed_everything(42)
    
    args = parse_args()
    
    model_name = args.model_name_or_path
    model_name_simple = model_name.split("/")[-1]
    device="cuda"#"cpu"
    device_map="auto"
   
    
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, torch_dtype=torch.bfloat16)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    print(model)

    args.dataset_name = "math500"
    prompt_list = get_prompt_list(tokenizer, args.dataset_name,dataset_end_idx=args.dataset_end_idx)#从官网的json文件 加载 question字符串列表 

    if args.dataset_end_idx==None:
        args.dataset_end_idx=len(prompt_list)
        
    save_dir = args.pred_dir
    os.makedirs(save_dir, exist_ok=True)
    
    

    
    save_path = os.path.join(save_dir, f"{model_name_simple}-[{args.dataset_start_idx},{args.dataset_end_idx}).jsonl")#文件地址=文件夹 + {model_name_simple}
    
    print("--------------------------save_path:", save_path)


    generation_config = dict(
        temperature=0.7,
        top_p=0.8,
        max_new_tokens=args.max_tokens
    )


    output_tokens_list = []
    
    
    for prompt_idx, prompt in enumerate(tqdm(prompt_list)):
        
        if prompt_idx<args.dataset_start_idx or prompt_idx >=args.dataset_end_idx:
            continue
        
        past_key_values=None
        _input = tokenizer(prompt, return_tensors="pt", padding=True)
        _input_ids = _input["input_ids"]
        
        _attention_mask = _input['attention_mask']

        try:
            _output = model.generate( # 封装好的 generate(),内部不断的自回归。直接使用model是生成一个
                input_ids = _input_ids.to(device), 
                attention_mask = _attention_mask.to(device),
                past_key_values = past_key_values,
                **generation_config)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                del _input_ids, _attention_mask
                torch.cuda.empty_cache() 
                print("OOM!", flush=True)
                continue
            else:
                raise

            
        
        output_text = tokenizer.batch_decode(_output)[0]

        _output_id_list = _output[0].tolist()
        output_tokens_list.append(_output_id_list)

        
        sample_output={
            "id":prompt_idx,
            "output_text":output_text,
            "output_token_num":len(_output_id_list),
        }

        with open(save_path, "a", encoding="utf-8") as f:
            json.dump(sample_output, f, ensure_ascii=False)
            f.write("\n")

    print("---------------------pred finish,save_path:", save_path)



if __name__ == "__main__":
    main()

