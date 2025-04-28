#无需修改

import torch
from tqdm import tqdm
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
import numpy as np
import random

import argparse
from argparse import ArgumentParser

import os
import sys
# 获取当前文件的父目录，并加入到 sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from quest_attention_lidong import enable_quest_attention_eval

device = "cuda"

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str)
parser.add_argument("--fixed-length", type=int)
parser.add_argument("--max-tokens", type=int, default=8192)
parser.add_argument("--min-tokens", type=int, default=256)
parser.add_argument("--tokens-step", type=int)
parser.add_argument("--length-step", type=int, default=128)
parser.add_argument("--iterations", type=int, default=20)
parser.add_argument("--output_dir", type=str)

parser.add_argument("--num_eval_tokens", type=int, default=None)

parser.add_argument("--quest", action="store_true", help="Enable quest attention")
parser.add_argument("--token_budget", type=int, default=1024)
parser.add_argument("--chunk_size", type=int, default=16)

#添加注释 李东 3/28早上
def load(model_name_or_path):
    print(f"Loading model from {model_name_or_path} ...")
    # however, tensor parallel for running falcon will occur bugs
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None: #设置tokenizer.pad_token_id 
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    model.eval()#调用 model.eval() 将模型设置为评估模式，禁用 dropout 和 batch normalization

    return model, tokenizer



args = parser.parse_args()

data = load_dataset("emozilla/pg19-test", split="test")#官方的函数，只需要测试集test

model, tokenizer = load(args.model_name_or_path)#"lmsys/longchat-7b-v1.5-32k"或者"Qwen/Qwen2"
#打印data数据集
# Dataset({
#     features: ['short_book_title', 'publication_date', 'url', 'text'],
#     num_rows: 100 #有100个样本
# })
# 可以直接print(data[0])查看第一个数据样本，会自动上面四个key+value拼接输出
# 也可以print(data[0]['text'])只查看文本内容，注意这个data[0]['text']中有四万个单词（加上标点符号可能五万了）
nlls = []
loss_fn = CrossEntropyLoss(reduction="none")#计算交叉熵损失
past_key_values = None

#对LlamaAttention或MistralAttention自注意力子层修改forword函数，并传入一些命令行参数和层数
if args.quest:
    print("Enable quest attention")
    enable_quest_attention_eval(model, args)

os.makedirs(args.output_dir, exist_ok=True)
f = open(f"{args.output_dir}/log.txt", "w")

num_eval_tokens = 0
for text in data["text"][:1]:#data["text"][:1]是个只有一个元素的列表，就相当于[data[0]['text']]
    encodings = tokenizer(text, return_tensors="pt")

    print(encodings.input_ids[:, :10])#input_ids的第一维dim0=1，输出：tensor([[    1,  6850, 29889, 17687, 13309, 17435,    13,    13,    13, 29923]])

    seq_len = encodings.input_ids.size(1)
    print(f"seq_len: {seq_len}")#输出65134
    pbar = tqdm(range(0, seq_len - 1))#扣去了一个token，最后一个token

    for idx in pbar:
        input_ids = encodings.input_ids[:, idx : idx + 1].to(device)
        

        token_id = encodings.input_ids[0, idx].item()  # 转成整数
        token_str = tokenizer.decode(token_id)
        print("token:",token_str)


        with torch.no_grad():
            outputs = model(
                input_ids,#逐个token的喂
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits.view(-1, model.config.vocab_size)
            print("logits",logits)# nan
            #outputs.logits是个3Dtensor=(batch_size, seq_len, vocab_size)，表示对各个单词的预测概率。.view即.reshape，即合并dim0和dim1（注意bsz=1，单词逐个decode：seq_len=1）
            past_key_values = outputs.past_key_values
            label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)#label是个1Dtensor，label.shape=[bsz]=[1]，对应一个label索引位置为1.其余位置为0的概率分布
            print("label",label)
            neg_log_likelihood = loss_fn(logits, label)#计算交叉熵及最终的平均交叉熵（非负数，无上限，越小表示预测越准确）


        print("neg_log_likelihood:",neg_log_likelihood)
        nlls.append(neg_log_likelihood)
        pbar.set_description(
            f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"#会在终端内显示 单个交叉损失函数（即 负对数似然），但不会print输出
        )
        print(neg_log_likelihood.item(), file=f, flush=True)
        num_eval_tokens += 1
        if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
            break
    if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
        break

f.close()
#求平均，print输出，写进日志
ppl = torch.exp(torch.stack(nlls).mean())
print(ppl.item())
with open(f"{args.output_dir}/ppl.txt", "w") as f:
    f.write(f"{ppl.item()}\n")
