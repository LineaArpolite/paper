import os
from datasets import load_dataset
import torch
import json
import pickle
from transformers import (
    AutoTokenizer,
    AutoConfig,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
)
from tqdm import tqdm
import numpy as np
import random
import argparse

from transformers.cache_utils import DynamicCache,SinkCache

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        choices=[
            "meta-llama/Llama-3.1-8B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        ],
    )

    parser.add_argument("--task", type=str, help="task name", required=True)

    #sink cache
    parser.add_argument("--sink_cache", action='store_true')
    parser.add_argument("--local_window_size", type=int, default=2000)
    parser.add_argument("--begin_sink_size", type=int, default=64)


    # #quest
    # parser.add_argument("--token_budget", type=int, default=None)
    # parser.add_argument("--chunk_size", type=int, default=None)
    # parser.add_argument("--quest", action="store_true", help="Enable Quest Attention")
    
    return parser.parse_args(args)#ArgumentParser.parse_args() 是 argparse 标准库中自带的函数，不是自己定义的 parse_args 函数。


#添加注释 李东 3/28
def get_pred(
    model,
    tokenizer,
    data,#load_dataset得到的数据集
    max_length,
    max_gen,
    prompt_format,#里面包含 {} 占位符
    dataset,#数据集的名称，单个字符串
    device,
    model_name,
    args
):
    preds = []
    for i, json_obj in enumerate(tqdm(data)):#json_obj 是个字典（上下文kv+问题kv+答案kv），对应一个问题。tqdm：是一个 Python 进度条库，用于在循环中显示进度。这里 tqdm(data) 让 data 在迭代时显示进度条。enumerate为数据自动生成索引赋给i
        # if i>=20:
        #     break 
        prompt = prompt_format.format(**json_obj) # 使用 ** 解包字典json_obj，把字符串prompt_format里占位符{}里 用对应json_obj里key-value对 替换成value值，生成最终的 prompt+question 字符串
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer (
            prompt, truncation=False, return_tensors="pt" # truncation=False 表示不自动截断输入（默认可能会根据模型最大长度自动截断）。return_tensors="pt" 表示返回的是 PyTorch tensor 格式。最终返回2Dtensor
        ).input_ids[0]#tokenized_prompt.input_ids[0]是prompt的token_id的1Dtensor
        if "chatglm3" in model_name:#"chatglm3"中add_special_tokens=False
            tokenized_prompt = tokenizer(
                prompt, truncation=False, return_tensors="pt", add_special_tokens=False
            ).input_ids[0]
        if len(tokenized_prompt) > max_length:#若tokenized_prompt的长度大于模型的最大处理长度max_length，则又token_id便会token
            half = int(max_length / 2)
            prompt = tokenizer.decode(
                tokenized_prompt[:half], skip_special_tokens=True
            ) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)# "+"是字符串拼接操作
        
        
        # split the prompt and question (simulate decoding in the question stage)
        if dataset in ["qasper", "hotpotqa"]:
            q_pos = prompt.rfind("Question:")
        elif dataset in ["multifieldqa_en", "gov_report"]:
            q_pos = prompt.rfind("Now,")
        elif dataset in ["triviaqa"]:
            q_pos = prompt.rfind("Answer the question")
        elif dataset in ["narrativeqa"]:
            q_pos = prompt.rfind("Do not provide")
        else:
            q_pos = -1

        # max simulation length is 100
        q_pos = max(len(prompt) - 100, q_pos)#经过统计，question的长度都不会超过100（甚至50），所以这里设置了一个最小值len(prompt)-100，防止q_pos=-1，导致后续的question截取不到（若q_pos=-1则强行截取后100个作为question）

        if q_pos != None:
            question = prompt[q_pos:]
            prompt = prompt[:q_pos]
        else:
            raise ValueError(f"Cannot find the question position in the prompt for dataset {dataset}. ")

        
        # input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        #tokenizer("字符串", truncation=false不进行截断，return_tensors="pt"返回torch.Tensor格式的数据).to("cuda")将Tensor移动到GPU计算，
        p_input = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda") #prompt部分：字符串对shape（1，n_prompt）两层tensor的映射{'input_ids':tensor[[]],'atteneion_mask':tensor([[]])}
        q_input = tokenizer(question, truncation=False, return_tensors="pt").to("cuda")  #question对应的input，（1，n_question）维tensor，同上
        
        q_input.input_ids = q_input.input_ids[:, 1:]#关于q_input的每个batch去掉<s>token

        context_length = p_input.input_ids.shape[-1] + q_input.input_ids.shape[-1]

        from SinkCacheWrapper import SinkCacheWrapper

        if args.sink_cache:
            KV_cache = SinkCacheWrapper(args.local_window_size, args.begin_sink_size).to("cuda")
        else:
            KV_cache = None

        # 在 forward/generate 里照常传入：
        # model(input_ids=..., past_key_values=KV_cache, ...)

        

        with torch.no_grad():
            output = model(
                input_ids=p_input.input_ids,
                past_key_values=KV_cache,
                use_cache=True,
            )
            # 关于prompt部分，一口气喂
            # output也是个字典，
            # {
            #     'logits':(batch_size, seq_len ,vocab_size)维度的tensor , 代表概率分布
            #     'past_key_value':
            # }
            KV_cache = output.past_key_values
            #past_key_values是个元组（不可改变），维度是 layer ，2，batch_size, num_heads, seq_len, head_dim
            # (
            #     (tensor([[[...]]]), tensor([[[...]]])),  # 第 1 层 (key, value)
            #     (tensor([[[...]]]), tensor([[[...]]])),  # 第 2 层 (key, value)
            #     ...
            #     (tensor([[[...]]]), tensor([[[...]]]))   # 第 12 层 (key, value)
            # )
            for input_id in q_input.input_ids[0]: # 关于question部分，逐个token喂
                output = model(
                    input_ids=input_id.unsqueeze(0).unsqueeze(0),
                    past_key_values=KV_cache,
                    use_cache=True,
                )
                KV_cache = output.past_key_values

            #'logits':(batch_size, seq_len ,vocab_size)维度的tensor , 代表概率分布。[:, -1, :]表示取最后一个token的概率分布,(batch_size, seq_len ,vocab_size)->(batch_size ,vocab_size)
            # argmax(dim=-1)表示只取第-1个维度（vocab_size）即中最大值，(batch_size,vocab_size)->(batch_size)
            # unsqueeze(1)：(batch_size)->(batch_size,1)
            pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_token_ids = [pred_token_idx.item()] #.item()将单个元素的tensor转换成python数值，存入generated_content列表
            for _ in range(max_gen - 1):#关于answer部分，逐个token喂，且是autogressive
                outputs = model(
                    input_ids=pred_token_idx,
                    past_key_values=KV_cache,
                    use_cache=True,
                )

                KV_cache = outputs.past_key_values
                pred_token_idx = (
                    outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                )
                generated_token_ids += [pred_token_idx.item()]#列表拼接操作
                if pred_token_idx.item() == tokenizer.eos_token_id:
                    break

        # 可以通过tokenizer.all_special_tokens查看会跳过哪些token
        # pred = tokenizer.decode(generated_content, skip_special_tokens=True)
        
        preds.append(generated_token_ids)
    return preds #返回一个数据集中所有sample的pred，是token_id列表的列表


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(model_name, device):
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = model.eval()

    return model, tokenizer

#添加注释 李东 3/28下午
if __name__ == "__main__":
    seed_everything(42)#42 被很多开发者和数据科学家用作 默认的随机种子
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model_name_or_path
    # define your model
    model, tokenizer = load_model_and_tokenizer(
        model_name, device 
    )

    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    max_length = model2maxlen[model_name]#几千几万，模型的最大输出长度

    
    datasets_name = [args.task]#arg.task="qasper"/"narrativeqa"/"hotpotqa"/"multifieldqa_en"/"gov_report"/"triviaqa"，但是每次执行只会执行一个任务，所以这里的args.task只会是一个字符串一次python执行只完成一个task
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

    # predict on each dataset
    if not os.path.exists("results/longbench"):
        os.makedirs("results/longbench")

    model_name_simple=model_name.split("/")[-1]
    for dataset_name in datasets_name:#dataset_name= "qasper"/"narrativeqa"/"hotpotqa"/"multifieldqa_en"/"gov_report"/"triviaqa"
        
        data = load_dataset("THUDM/LongBench", dataset_name, split="test")#只用"THUDM/LongBench"的测试集
        if not os.path.exists(f"results/longbench/{model_name_simple}_{args.begin_sink_size}_{args.local_window_size}"):
            os.makedirs(f"results/longbench/{model_name_simple}_{args.begin_sink_size}_{args.local_window_size}")

        out_path = f"results/longbench/{model_name_simple}_{args.begin_sink_size}_{args.local_window_size}/{dataset_name}-full.pkl"#没用quest则是full

        prompt_format = dataset2prompt[dataset_name]#短提示词
        max_gen = dataset2maxlen[dataset_name]#几十一百
        preds = get_pred(
            model,
            tokenizer,
            data,#load_dataset得到的数据
            max_length,#几千几万：模型的最大处理长度
            max_gen,#几十一百：数据集的最大生成长度（最长的回答长度）
            prompt_format,#短提示词
            dataset_name,
            device,
            model_name,#全称字符串
            args=args
        )
        with open(out_path, "wb") as f:
            pickle.dump(preds, f)#不需要像jsonl一样 f.write("\n")

