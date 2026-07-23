# Lab: 手写 Transformer 积木 + Encoder Block

## 目标

- 亲手实现 4 个积木:Positional Encoding、LayerNorm(用 PyTorch 内置)、FeedForward、EncoderBlock
- 把它们拼成一个完整的 Encoder Block
- 验证形状正确(能跑通前向传播)

## 要求

打开 `train.py`，找到 3 个 `# TODO`：

1. **TODO 1**：PositionalEncoding 的 forward（sin/cos 位置编码公式）
2. **TODO 2**：FeedForward 网络结构（两层 Linear + GELU）
3. **TODO 3**：EncoderBlock 的 forward（两个 Add & Norm）

MultiHeadAttention 直接从 L3.1 复制过来（已放在文件里）。LayerNorm 用 `nn.LayerNorm` 内置。

## 运行环境

- 推荐机器：3070 笔记本（CPU 也可）
- 预估显存：0
- 预估时长：< 3 秒

## 验收标准

- [ ] 代码跑通，打印 "All shape checks passed"
- [ ] 每个积木的输入输出形状符合预期
- [ ] 能解释 3 个 TODO 每一步在做什么

## 提示

- **PositionalEncoding**：先建一个 (max_len, d_model) 的矩阵，按 sin/cos 公式填充，forward 时取前 seq_len 行加到 x 上
- **FeedForward**：`nn.Sequential(Linear(d, 4d), GELU(), Linear(4d, d))`
- **EncoderBlock**：两个 Add & Norm。第一个包 MHA，第二个包 FFN。注意残差是 `x + sublayer(x)`，再过 LayerNorm
