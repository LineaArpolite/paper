import argparse
import datasets
import json
import pickle

def compute_lps(pattern):
    m = len(pattern)
    lps = [0] * m
    length = 0  
    i = 1

    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    return lps


def find_max_and_index(nums):
    if not nums:  # Check if the list is empty
        return None, None
    max_val = max(nums)  # Find the maximum value
    max_index = nums.index(max_val)  # Find the index of the maximum value
    return max_val, max_index


def suffix_pattern_matching(sequence, len_thresh):
    sequence_reverse = sequence[::-1]
    subsequence = sequence_reverse

    cnt = 0
    matched_pairs = [] # list of (replace_len, replace_idx_start, matching_idx_start)
    while len(subsequence) > len_thresh:
        lps = compute_lps(subsequence)
        _maxlen, _max_idx = find_max_and_index(lps)
        
        if _maxlen < len_thresh:
            subsequence = subsequence[1:]
        else:
            cnt += _maxlen
            matched_pairs.append((_maxlen, len(subsequence)-_maxlen, len(subsequence)-1-_max_idx))


            subsequence = subsequence[_maxlen:]
    return cnt, matched_pairs


def load_token_lists(file_path):
    with open(file_path, "rb") as f:
        token_lists = pickle.load(f)
    return token_lists


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
    
    
    parser.add_argument("--save_dir", type=str,required=True)#跑math500时用的

    parser.add_argument("--save_name", type=str, default="")

    args = parser.parse_args()
    return args





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

