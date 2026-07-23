# Lab: Decoder-Only Transformer + 字符级语言模型训练

## 目标

- 在 L3.2 积木基础上加 DecoderBlock(带 mask)
- 组装完整 Decoder-Only Transformer(就是 GPT 的结构)
- 在字符级语言模型任务上训练(预测下一个字符)
- 训练后让模型生成文本

## 要求

打开 `train_lm.py`，找到 3 个 `# TODO`：

1. **TODO 1**：DecoderBlock forward（带 mask 的 Add & Norm，和 EncoderBlock 几乎一样，只是 attention 传 mask）
2. **TODO 2**：完整 Transformer 的 forward（embedding -> PE -> N 个 block -> 输出层）
3. **TODO 3**：文本生成（自回归采样循环）

## 运行环境

- 推荐机器：3070 笔记本（CPU 也能跑，稍慢）
- 预估显存：< 1 GB
- 预估时长：CPU 约 3-5 分钟，GPU 约 1 分钟

## 验收标准

- [ ] 代码跑通，loss 逐步下降
- [ ] 训练后能生成"像英文"的文本（不是完全随机字符）
- [ ] 能解释 mask 的作用、自回归生成的流程

## 提示

- **DecoderBlock** 和 EncoderBlock 唯一区别：attention 要传 mask
- **完整模型 forward**：`x = embedding(input) -> x = pe(x) -> for block: x = block(x, mask) -> logits = output_head(x)`
- **生成**：从起始字符开始，每次用模型预测下一个字符的概率，采样一个，拼到输入里，重复
