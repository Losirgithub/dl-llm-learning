"""L3.3 实验:体验预训练模型。

用 HuggingFace 加载真正的 GPT-2 和 BERT，对比 L3.2 你手写的玩具 GPT。
感受"预训练 + 规模"的力量。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 用国内镜像下载 HuggingFace 模型(必须在 import transformers 之前设)
# huggingface.co 在国内不稳定，hf-mirror.com 是国内镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.utils import set_seed  # noqa: E402


# ===========================================================================
# 1. BPE Tokenizer 体验:看 GPT-2 怎么切词
# ===========================================================================
def show_tokenization() -> None:
    """展示 BPE 子词 tokenization。"""
    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    examples = [
        "hello world",
        "unhappiness",
        "The quick brown fox jumps over the lazy dog.",
        "tokenization is important",
        "antidisestablishmentarianism",  # 超长罕见词，看怎么拆
    ]

    print("=" * 60)
    print("BPE Tokenization 展示(GPT-2)")
    print("=" * 60)
    for text in examples:
        token_ids = tokenizer.encode(text)
        tokens = [tokenizer.decode([tid]) for tid in token_ids]
        print(f"\n原文: {text}")
        print(f"tokens ({len(tokens)} 个): {tokens}")
        print(f"token 数: {len(tokens)}, 字符数: {len(text)}")


# ===========================================================================
# TODO 1: 用 GPT-2 生成文本
# ===========================================================================
def generate_with_gpt2() -> None:
    """用真正的 GPT-2 生成文本，对比 L3.2 的玩具 GPT。"""
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    print("\n" + "=" * 60)
    print("GPT-2 生成(对比 L3.2 玩具 GPT)")
    print("=" * 60)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GPT-2 参数量: {n_params:,}  (对比 L3.2 你的玩具 GPT: ~50 万)")
    print(f"L3.2 玩具 GPT 训练数据: ~600 字符莎士比亚片段")
    print(f"GPT-2 预训练数据: ~40GB 网页文本(WebText)")

    # TODO 1: 用 GPT-2 生成文本
    # 提示:
    #   1. prompt = "To be, or not to be"  (和 L3.2 一样的开头,对比效果)
    #   2. input_ids = tokenizer.encode(prompt, return_tensors="pt")
    #   3. output = model.generate(
    #          input_ids,
    #          max_new_tokens=100,
    #          do_sample=True,        # 启用采样(不是贪心)
    #          temperature=0.7,       # 温度,和 L3.2 一样
    #          top_p=0.9,             # top-p 采样,限制在概率前 90% 的词里采
    #          pad_token_id=tokenizer.eos_token_id,
    #      )
    #   4. generated = tokenizer.decode(output[0], skip_special_tokens=True)
    #   5. print(generated)
    prompt = "To be, or not to be"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")  # 每个token对应的词表id
    # 自回归生成的封装。内部就是L3.2 写的 generate 循环(预测一个 -> 拼回去 -> 重复),但 HuggingFace 帮你封装好了,支持多种采样策略。
    output = model.generate(
        input_ids,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated)
    # raise NotImplementedError("TODO 1: 用 GPT-2 生成文本")


# ===========================================================================
# TODO 2: 用 BERT 做 MLM(遮词预测)
# ===========================================================================
def bert_mlm() -> None:
    """用 BERT 做掩码语言模型:遮住一个词，看 BERT 预测 top5。"""
    from transformers import BertForMaskedLM, BertTokenizer

    print("\n" + "=" * 60)
    print("BERT MLM(遮词预测)")
    print("=" * 60)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    model.eval()

    # TODO 2: 遮词预测
    # 提示:
    #   1. text = "The capital of France is [MASK]."  (遮住 Paris)
    #      或自己造句: "I want to [MASK] a book about deep learning."
    #   2. input_ids = tokenizer.encode(text, return_tensors="pt")
    #   3. mask_token_index = (input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    #      找到 [MASK] 的位置
    #   4. with torch.no_grad():
    #          logits = model(input_ids).logits  # (1, seq, vocab)
    #      mask_logits = logits[0, mask_token_index, :]  # (1, vocab) MASK 位置的预测
    #   5. top5 = torch.topk(mask_logits, 5, dim=-1)  # 取概率最高的 5 个
    #   6. for tid in top5.indices[0]:
    #          print(f"  {tokenizer.decode([tid])}")
    text = "What the [MASK] are you doing?"
    input_ids = tokenizer.encode(text, return_tensors="pt")  # input_ids 形状 (1, seq)
    # (input_ids=to.mask_token_id)[0]去掉batch维->(seq,)，nonzero返回非零元素的索引,因为是astuple，所以返回的是元组，
    # as_tuple=True返回元组形式，取第一个元素就是mask位置的索引
    mask_token_index = (input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    # 上下文管理器，块内推理部分，不需要梯度计算
    with torch.no_grad():
        logits = model(input_ids).logits
    # model(input_ids):前向传播。BERT 把整个序列过一遍,对每个位置都输出"这个位置可能是什么词"的预测。
    # .logits:BERT 的输出是一个对象(MaskedLMOutput),
    # .logits 取出里面的预测张量。形状 (1, seq, vocab_size):
    mask_logits = logits[0, mask_token_index, :]
    top5 = torch.topk(mask_logits, 5, dim=-1)
    print(f"原句: {text}")
    print("BERT 预测 top5:")
    for tid in top5.indices[0]:
        print(f"  {tokenizer.decode([tid])}")
    # raise NotImplementedError("TODO 2: BERT 遮词预测")


def main() -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")

    show_tokenization()
    generate_with_gpt2()
    bert_mlm()

    print("\n" + "=" * 60)
    print("对比总结")
    print("=" * 60)
    print("L3.2 玩具 GPT: 50 万参数 + 600 字符 -> 生成重复退化文本")
    print("GPT-2: 1.24 亿参数 + 40GB 预训练 -> 生成流畅英文")
    print("BERT: 1.1 亿参数 + 16GB 预训练 -> 遮词预测准确")
    print("\n这就是'预训练 + 规模'的力量--架构一样，数据和规模决定能力。")


if __name__ == "__main__":
    main()
