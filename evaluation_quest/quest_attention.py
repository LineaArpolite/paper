import math
import numpy as np
from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.cuda.amp import autocast

import types

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)

from transformers.cache_utils import DynamicCache

from transformers.models.mistral.modeling_mistral import MistralAttention
#添加注释 李东 3/27晚上
def local_heavy_hitter_mask(attn_weights, token_budget, chunk_size):#注意attn_weight中每chunk_size个元素是相同的
    # attn_weights (BS, head, query, keys)

    # expend attn_weights to be divisible by chunk_size
    seq_length = attn_weights.shape[-1]
    padding_length = chunk_size - ((seq_length - 1) % chunk_size + 1)
    attn_weights = torch.cat(
        [
            attn_weights,
            torch.ones(
                (
                    attn_weights.shape[0],
                    attn_weights.shape[1],
                    attn_weights.shape[2],
                    padding_length,
                ),
                device=attn_weights.device,
            )
            * torch.tensor(torch.finfo(attn_weights.dtype).min),
        ],
        dim=-1,
    )

    # chunk attn_weights into chunk_size tokens
    chunk_attn_weights = attn_weights.reshape(
        attn_weights.shape[0],
        attn_weights.shape[1],
        attn_weights.shape[2],
        attn_weights.shape[3] // chunk_size,
        chunk_size,
    ).amax(dim=-1)#[bsz,num_head,1,seq_len/chunk] 懵了 不是每chunk_size个元素是相同的吗？为什么还要求最大值

    _, topk = chunk_attn_weights.topk(#决定要哪几个chunk（topk为索引），最多不超过总个数，最少要3个
        k=min(max(3, token_budget // chunk_size), chunk_attn_weights.size(-1)), dim=-1
    )#top_k.shape:(batch_size, num_heads, q_len=1, k)表示索引位置
    # repeat topk chunk_size times and recover the original indexes (* chunk_size + arange(chunk_size))
    topk = topk.unsqueeze(-1).repeat(
        1, 1, 1, 1, chunk_size#每个索引位置都复制chunk_size次
    ) * chunk_size + torch.arange(chunk_size, device=topk.device) # 索引块号*块大小 + 块内位置 = 最终的位置
    topk = topk.reshape(topk.shape[0], topk.shape[1], topk.shape[2], -1)#dim=-2索引块号 dim=-1块内位置
    mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
    mask_bottom.scatter_(-1, topk, True)#按照top_k中对应位置为Ture

    # remove the padding
    mask_bottom = mask_bottom[:, :, :, :seq_length]

    return mask_bottom #返回[bsz,num_head,q_len=1,seq_len]，其中1or0，1是成块出现的，表示想要的元素

#添加注释 李东 3/27晚上
def forward(
    self,#，self 就指的是当前的LlamaAttention 或 MistralAttention类的实例
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,#计算位置编码需要
    past_key_value: Optional[Tuple[torch.Tensor]] = None,#这些self.变量都是只对应一层的
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    if q_len > 1 or self.layer_id < 2: #q_len>1表示是在prefill阶段（decode阶段q_len=1），离用户最近的两层的layer_id=0和1
        return self.flash_forward(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            **kwargs,
        )

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)#.view()调整tensor的shape为目标shape
        .transpose(1, 2)#转换成(bsz, self.num_heads, q_len=1, self.head_dim)便于每个头独立，加速并行
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)#注意这里的dim=2和上面不同，说明可能会repeat_kv
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        .transpose(1, 2)
    )
    
    # New cache format。Qwen2和longchat都是进入的if分支
    if isinstance(past_key_value, DynamicCache):#假如使用DynamicCache类来存储past_key_value。即 past_key_value 是一个 DynamicCache 对象
        kv_seq_len = past_key_value.get_seq_length()#DynamicCache自行管理序列的长度（包括新的 key 和 value），不需要手动在计算 kv_seq_len 时加上新的 key 或 value 的长度。
    # Legacy cache format
    else:#使用传统的旧缓存格式，即 past_key_value 是一个元组，包含了之前的 key 和 value 向量
        kv_seq_len = key_states.shape[-2]#等于1
        if past_key_value is not None:
            assert isinstance(past_key_value, tuple)
            kv_seq_len += past_key_value[0].shape[-2]
    
    cos, sin = self.rotary_emb(value_states, position_ids.to(value_states.device))
    query_states, key_states = apply_rotary_pos_emb( #官方函数，将query_states和key_states加上cos和sin
        query_states, key_states, cos, sin, position_ids
    )
    # [bsz, nh, t, hd]

    # New cache format
    if isinstance(past_key_value, DynamicCache):
        if use_cache:
            key_states, value_states = past_key_value.update(key_states, value_states, layer_idx=self.layer_idx)
    # Legacy cache format
    else:
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)#在kvcache=past_key_value[0]后拼接key_states，其中维度[bsz, self.num_heads, q_len=1, self.head_dim]，保持q_len拼接，其他维度不变
            value_states = torch.cat([past_key_value[1], value_states], dim=2)#同理，让value_states表示所有的v_cache
        past_key_value = (key_states, value_states) if use_cache else None

    key_states = repeat_kv(key_states, self.num_key_value_groups)#repeat_kv后self.num_key_value_heads变成self.num_heads
    value_states = repeat_kv(value_states, self.num_key_value_groups)
    
    #注意，这里query_states.shape=(bsz, self.num_heads, q_len=1, self.head_dim),
    #       key_states.shape=(bsz, self.num_heads, seq_len=总长度, self.head_dim)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(#.transpose(2, 3)是为了在self.head_dim对齐，可以执行乘法
        self.head_dim
    )#返回(bsz, self.num_heads, q_len=1,q_len=总长度)的tensor

    sign = (query_states > 0) + (~(query_states > 0)) * -1 #相当于对query_states的每一个元素求符号函数，>0的函数值=1，<=0的函数值-1，返回相同形状的tensor
    max_key = key_states * sign #相当于把query_states的符号剥夺 乘给key_states，即sign(q)*v
    postive_query = query_states * sign #相当于对query_states的符号剥夺：q*v=|q|*(sign(q)*v)

    # expend max_key to be divisible by chunk_size
    seq_length = max_key.shape[-2]
    padding_length = self.chunk_size - ((seq_length - 1) % self.chunk_size + 1)
    max_key = torch.cat( #padding扩充的数为大负数
        [
            max_key,
            torch.ones(
                (max_key.shape[0], max_key.shape[1], padding_length, max_key.shape[3]),
                device=max_key.device,
            )
            * torch.tensor(torch.finfo(max_key.dtype).min),
        ],
        dim=-2,
    )
    #每self.chunk_size个求最大值
    # chunk max_key into chunk_size tokens
    chunk_max_key = max_key.reshape( #max_key是sign(q)*v
        max_key.shape[0],
        max_key.shape[1],
        max_key.shape[2] // self.chunk_size,
        self.chunk_size,
        max_key.shape[3],
    ).amax(dim=-2) #先从4Dtensor拆分成5D，在dim=-2维度求最大值变成4D：[bsz, self.num_heads, (总长度seq_len+padding)/chunk_size, self.head_dim]

    # duplicate chunk_max_key chunk_size times
    chunk_max_key = chunk_max_key.unsqueeze(-2).repeat(1, 1, 1, self.chunk_size, 1)#在dim=-2出生成1，[a,b,c,d]变[a,b,c,1,d],再各个位置复制对应的倍数，abcd不复制，1复制chunk_size次
    # reshape chunk_max_key to the original shape
    chunk_max_key = chunk_max_key.reshape(
        chunk_max_key.shape[0], chunk_max_key.shape[1], -1, chunk_max_key.shape[-1]#合并dim=2和dim=3
    )[:, :, :seq_length, :]#裁减成seq_len长度

    quantized_weight = torch.matmul(
        postive_query.float(),#[bsz, self.num_heads, q_len=1, self.head_dim]
        chunk_max_key.transpose(2, 3),#转置后[bsz, self.num_heads,  self.head_dim, seq_len=总长度]
    )#[bsz, self.num_heads, q_len=1,seq_len=总长度]，但是结果中每chunk_size个结果是相同的

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len): #q_len=1
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )
    """
    ValueError: Attention weights should be of size (1, 28, 1, 0), but is torch.Size([1, 28, 1, 1])
    """
    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len): #q_len=1
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask #attn_weights=真正的q*K，attention_mask中元素为0or超大负数
        attn_weights = torch.max(#超大负数变统一的大负数
            attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
        )
        quantized_weight = quantized_weight + attention_mask
        quantized_weight = torch.max(
            quantized_weight, torch.tensor(torch.finfo(quantized_weight.dtype).min)
        )#[bsz, self.num_heads, q_len=1,seq_len=总长度]，但是结果中每chunk_size个结果是相同的，为注意力值or大负数

    token_budget = min(kv_seq_len, self.token_budget)

    attn_weights_for_selection = quantized_weight

    if token_budget > 0:
        mask_bottom = local_heavy_hitter_mask(
            attn_weights_for_selection, token_budget, self.chunk_size
        )  # Default: No padding applied to input
    else:
        mask_bottom = torch.zeros_like(attn_weights_for_selection, dtype=torch.bool)

    mask_bottom = torch.tril(mask_bottom, diagonal=position_ids[0][0].item())# q_len=1还还需要三角化吗？
    #注意attn_weights_for_selection中每chunk_size个元素是相同的，但attn_weights元素是不同的
    attn_weights[~mask_bottom] = torch.tensor(torch.finfo(attn_weights.dtype).min)# 将mask_bottom中为False的位置，在attn_weights中变成大负数

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    ) #[bsz,num_heads,q_len=1,seq_len]
    attn_output = torch.matmul(attn_weights, value_states)#[bsz,num_heads,q_len=1,seq_len]和[bsz,num_heads,seq_len,head_dim]相乘

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)#变成(bsz, q_len, self.num_heads, self.head_dim)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)# 多个头的输出拼接concat

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


global layer_id
layer_id = 32

# 添加注释 李东 3/26晚上
# model=AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B-Instruct',....)
# arg.m/arg.model=meta-llama/Llama-3.1-8B-Instruct
# arg.iterations=100
# arg.fixed-length=10000
# arg.quest=true
# arg.token_budget=512
# arg.chunk_size=16
# arg.output-file=results/Llama-3.1-8B-Instruct-quest-512.jsonl
# arg.max-tokens=8192
# arg.min-tokens=256
# arg.length-step=128
# arg.iterations=20

def enable_quest_attention_eval(model, args):
    for name, module in reversed(model._modules.items()):#先从离用户远的子模块（layer_id=31）开始，【假设整个model是个树，用户在左，结点先右再左最后本身】
        if len(list(module.children())) > 0:
            enable_quest_attention_eval(
                module,
                args,
            )

        global layer_id
        #LlamaAttention, MistralAttention是官方类，只检查这两种类，都是自注意力子层的类
        if isinstance(module, (LlamaAttention, MistralAttention)):#if isinstance(对象/变量a,类/类型或其元组B) 如果对象/变量a的类/类型等于B或为元组之一，则为True。
            # For longchat model
            layer_id -= 1
            model._modules[name].layer_id = layer_id
            model._modules[name].flash_forward = model._modules[name].forward #哪怕没有事先定义函数flash_forward，也能替换函数。将模型原有的forward函数改名为flash_forward
            model._modules[name].forward = types.MethodType(
                forward, model._modules[name]
            )

            model._modules[name].token_budget = args.token_budget
            model._modules[name].chunk_size = args.chunk_size
