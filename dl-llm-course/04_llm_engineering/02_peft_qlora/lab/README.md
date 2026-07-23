# Lab: QLoRA 微调 Qwen2.5(8GB 显存可跑)

## 目标

- 在 3070(8GB)上用 QLoRA 微调 Qwen2.5-0.5B-Instruct
- 学会用 PEFT 库 + SFTTrainer 做指令微调
- 观察:微调前后模型回答的变化

## 要求

打开 `finetune.py`，找到 2 个 `# TODO`：

1. **TODO 1**：配置 LoRA 参数(r、alpha、target_modules)
2. **TODO 2**：准备一份小型指令数据集

## 运行环境

- 推荐机器：3070(8GB) 或 5070Ti(16GB)
- 模型：Qwen2.5-0.5B-Instruct(4-bit 量化后 ~0.5GB，8GB 显存绰绰有余)
- 预估时长：微调 3-5 分钟(小数据集)，下载模型 ~1GB

## 前置准备

```bash
pip install peft trl datasets
```

## 验收标准

- [ ] 微调跑通，loss 下降
- [ ] 微调后模型按训练数据的风格回答
- [ ] 能解释 LoRA 配置参数(r、alpha、target_modules)的意义

## 提示

- 用小模型 Qwen2.5-0.5B(不是 1.5B)确保 3070 能跑 + 快
- 小数据集(几十条)足够观察微调效果
- target_modules 一般选 attention 的 q_proj、v_proj
