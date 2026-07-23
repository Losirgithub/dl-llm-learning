"""L4.1 实验:加载 Qwen 大模型并对话 + 4-bit 量化。

用 HuggingFace 加载真正的开源 LLM(Qwen2.5-Instruct)，和它对话，
再体验 4-bit 量化把 7B 模型塞进 8GB 显存。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 用国内镜像(必须在 import transformers 之前)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 把 HF 模型缓存改到 E 盘(C 盘空间不足,默认在 C:\Users\...\.cache\huggingface)
os.environ["HF_HOME"] = "E:/hf_cache"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.utils import set_seed  # noqa: E402


# ===========================================================================
# 1. 加载 1.5B 模型并对话(基础版)
# ===========================================================================
def chat_with_qwen_1_5b() -> None:
    """加载 Qwen2.5-1.5B-Instruct，单轮对话。"""
    print("=" * 60)
    print("1. Qwen2.5-1.5B-Instruct 对话")
    print("=" * 60)

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    print(f"加载模型: {model_name} ...")

    # 加载模型和 tokenizer
    # torch_dtype="auto":自动用 fp16/bf16 省显存
    # device_map="auto":自动放到 GPU
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto", device_map="auto")
    print(f"模型设备: {next(model.parameters()).device}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # TODO 1: 构造对话消息 + 应用 chat template
    # 提示:
    #   1. messages = [{"role": "user", "content": "用三句话解释什么是 Transformer"}]
    #      也可以自己换问题
    #   2. text = tokenizer.apply_chat_template(
    #          messages, tokenize=False, add_generation_prompt=True
    #      )
    #      add_generation_prompt=True 在末尾加"该 assistant 说了"的提示
    #   3. model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    #      注意 .to(model.device) 把输入搬到模型所在设备
    messages = [{"role": "user", "content": "你是什么模型？"}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    # raise NotImplementedError("TODO 1: 构造消息 + chat template")

    # TODO 2: 生成回复 + 解码
    # 提示:
    #   1. generated_ids = model.generate(
    #          model_inputs.input_ids,    # 输入
    #          max_new_tokens=512,        # 最多生成 512 个新 token
    #          do_sample=True,            # 采样模式
    #          temperature=0.7,           # 温度
    #          top_p=0.8,                 # 核采样
    #      )
    #   2. generated_ids = [
    #          output_ids[len(input_ids):]
    #          for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    #      ]
    #      去掉 prompt 部分，只留新生成的
    #   3. response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    #   4. print(response)
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
    )
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
    # raise NotImplementedError("TODO 2: 生成 + 解码")


# ===========================================================================
# 2. 4-bit 量化:把 7B 模型塞进 8GB 显存
# ===========================================================================
def chat_with_qwen_7b_4bit() -> None:
    """用 4-bit 量化加载 Qwen2.5-7B-Instruct，8GB 显存也能跑。"""
    print("\n" + "=" * 60)
    print("2. Qwen2.5-7B-Instruct (4-bit 量化)")
    print("=" * 60)

    from transformers import BitsAndBytesConfig

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"加载模型: {model_name} (4-bit) ...")

    # 4-bit 量化配置
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
    )

    # 看显存占用
    if torch.cuda.is_available():
        mem_gb = torch.cuda.memory_allocated() / 1024**3
        print(f"4-bit 7B 模型显存占用: {mem_gb:.2f} GB  (fp16 本要 14GB)")
        print(f"对比: 3070 显存 8GB,4-bit 让 7B 也能跑!")

    # 同样的对话流程(这次直接给完整代码,你体会 4-bit 和 fp16 用法一样)
    messages = [{"role": "user", "content": "写一首关于深度学习的四行诗"}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            inputs.input_ids, max_new_tokens=200, do_sample=True, temperature=0.8, top_p=0.8
        )
    response = tokenizer.decode(output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
    print(f"\n7B 模型回复:\n{response}")


def main() -> None:
    set_seed(42)
    print(f"使用设备: {'cuda' if torch.cuda.is_available() else 'cpu'}\n")

    chat_with_qwen_1_5b()

    # 4-bit 7B 需要装 bitsandbytes,且下载较大(~15GB)
    # 如果 3070 显存紧张或不想下大模型,可以注释掉这行
    # try:
    #     chat_with_qwen_7b_4bit()
    # except Exception as e:
    #     print(f"\n[跳过 7B 4-bit 实验] {e}")
    #     print("如果没装 bitsandbytes: pip install bitsandbytes")


if __name__ == "__main__":
    main()
