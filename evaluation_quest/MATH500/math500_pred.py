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




import os
import sys
# 获取当前文件的父目录，并加入到 sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from quest_attention_lidong import enable_quest_attention_eval

import os
import torch
import random
import numpy as np
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


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

    #Quest
    parser.add_argument("--token_budget", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--quest", action="store_true", help="Enable Quest Attention")

    #Sink
    parser.add_argument("--cache_type", type=str,choices=["sink","dynamic","none"], default="none")
    parser.add_argument("--local_window_size", type=int, default=512)
    parser.add_argument("--begin_sink_size", type=int, default=16)
    
    
    parser.add_argument("--pred_dir", type=str,required=True)

    parser.add_argument("--save_name", type=str, default="")

    args = parser.parse_args()
    return args



@torch.no_grad()
def main():
    seed_everything(42)
    
    args = parse_args()
    
    model_name = args.model_name_or_path
    model_name_simple = model_name.split("/")[-1]
    device="cuda"#"cpu"
    device_map="auto"
   
    
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, torch_dtype=torch.bfloat16)
    if args.quest:
        enable_quest_attention_eval(model, args)
    model = model.eval()
   
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
    

    generation_config = dict(
        temperature=0.7,
        top_p=0.8,
        max_new_tokens=args.max_tokens
    )


    output_tokens_list = []
    
    
    for prompt_idx, prompt in enumerate(tqdm(prompt_list)):


        if prompt_idx<args.dataset_start_idx or prompt_idx >=args.dataset_end_idx:
            continue
        


        if args.cache_type=="sink":
            KV_cache = SinkCache(args.local_window_size,args.begin_sink_size)
        elif args.cache_type=="dynamic":
            KV_cache = DynamicCache()
        else:
            KV_cache=None

            

        _input = tokenizer(prompt, return_tensors="pt", padding=True)
        _input_ids = _input["input_ids"]
        
        _attention_mask = _input['attention_mask']

        try:
            _output = model.generate( # 封装好的 generate(),内部不断的自回归。直接使用model是生成一个
                input_ids = _input_ids.to(device), 
                attention_mask = _attention_mask.to(device),
                past_key_values = KV_cache,
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

    print("---------------------pred completed,save_path:", save_path)



if __name__ == "__main__":
    main()

