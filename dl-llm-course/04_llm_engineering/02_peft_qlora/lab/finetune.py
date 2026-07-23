"""L4.2 实验:QLoRA 微调 Qwen2.5-0.5B-Instruct。

在 3070(8GB)上用 QLoRA 微调一个小模型，观察微调效果。
用 PEFT 库 + TRL 的 SFTTrainer。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 国内镜像 + 缓存到 E 盘(必须在 import transformers 之前)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "E:/hf_cache"
os.environ["HF_HUB_OFFLINE"] = "1"  # 强制离线模式,只用已缓存的模型(L4.1 已下 1.5B)

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.utils import set_seed  # noqa: E402


def load_model_4bit(model_name: str):
    """加载 4-bit 量化的模型(QLoRA 的 Q)。"""
    print(f"加载 4-bit 模型: {model_name}")

    # 4-bit 量化配置(QLoRA 的核心)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # 4-bit NormalFloat(QLoRA 论文核心创新1)
        bnb_4bit_compute_dtype=torch.bfloat16,  # 计算时用 bf16
        bnb_4bit_use_double_quant=True,  # 双重量化(核心创新2)
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def prepare_dataset():
    """准备一份小型指令微调数据集。

    任务:固定格式输出--不管输入什么,回答必须以【结果】开头、--AI小深结尾。
    这是基线模型默认不会做的,微调效果会很明显。
    """
    examples = [
        {"input": "苹果", "output": "【结果】苹果是一种水果--AI小深"},
        {"input": "深度学习", "output": "【结果】深度学习是机器学习的分支--AI小深"},
        {"input": "今天天气好", "output": "【结果】今日天气晴朗--AI小深"},
        {"input": "猫", "output": "【结果】猫是常见的宠物--AI小深"},
        {"input": "量子计算", "output": "【结果】量子计算利用量子力学原理--AI小深"},
        {"input": "篮球", "output": "【结果】篮球是一项团队运动--AI小深"},
        {"input": "太阳", "output": "【结果】太阳是太阳系的恒星--AI小深"},
        {"input": "Python", "output": "【结果】Python是流行的编程语言--AI小深"},
        {"input": "长城", "output": "【结果】长城是中国古代建筑--AI小深"},
        {"input": "咖啡", "output": "【结果】咖啡是一种提神饮品--AI小深"},
        {"input": "人工智能", "output": "【结果】人工智能是模拟人类智能的技术--AI小深"},
        {"input": "海洋", "output": "【结果】海洋覆盖地球大部分表面--AI小深"},
    ]

    # 转成模型能学的格式(text 字段)
    data = []
    for ex in examples:
        text = (
            f"<|im_start|>user\n请解释: {ex['input']}<|im_end|>\n"
            f"<|im_start|>assistant\n{ex['output']}<|im_end|>"
        )
        data.append({"text": text})

    return Dataset.from_list(data)


def main() -> None:
    set_seed(42)
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"  # 用 L4.1 已缓存的模型,避免重新下载

    # 1. 加载 4-bit 模型
    model, tokenizer = load_model_4bit(model_name)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    if torch.cuda.is_available():
        print(f"4-bit 加载后显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # 2. 配置 LoRA(QLoRA 的 LoRA)
    # TODO 1: 配置 LoRA 参数
    # 提示:
    lora_config = LoraConfig(
        r=16,  # 秩,加大到 16(原来 8 容量不够学新格式)
        lora_alpha=32,  # 缩放因子,常用 r 的 2 倍
        target_modules=["q_proj", "v_proj"],  # 给 attention 的 Q/V 加 LoRA
        lora_dropout=0.05,  # LoRA 层的 dropout
        task_type="CAUSAL_LM",  # 任务类型:因果语言模型
    )
    # raise NotImplementedError("TODO 1: 配置 LoraConfig")

    model = get_peft_model(model, lora_config)  # noqa: F821
    model.print_trainable_parameters()  # 打印可训练参数(应该只有 ~0.5%)

    # 3. 准备数据
    dataset = prepare_dataset()
    print(f"数据集大小: {len(dataset)}")

    # 4. 训练配置 + 训练
    training_args = SFTConfig(
        output_dir="./output",
        num_train_epochs=40,  # 多轮,让模型记住格式(原来5轮不够)
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=5e-4,   # LoRA 学习率可以大点
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        max_length=128,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # 5. 微调前测试(基线)
    test_prompts = ["区块链", "黑洞", "太极拳"]

    print("\n" + "=" * 60)
    print("微调前测试(基线)")
    print("=" * 60)
    test_model(model, tokenizer, test_prompts)

    # 6. 开始微调
    print("\n开始微调...")
    trainer.train()

    # 7. 微调后测试
    print("\n" + "=" * 60)
    print("微调后测试")
    print("=" * 60)
    test_model(model, tokenizer, test_prompts)


def test_model(model, tokenizer, prompts: list[str]) -> None:
    """测试模型是否学会了固定格式输出(【结果】XXX--AI小深)。"""
    model.eval()
    for prompt in prompts:
        messages = [{"role": "user", "content": f"请解释: {prompt}"}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        reply = tokenizer.decode(output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
        print(f"输入: {prompt}")
        print(f"回复: {reply}\n")


if __name__ == "__main__":
    main()
