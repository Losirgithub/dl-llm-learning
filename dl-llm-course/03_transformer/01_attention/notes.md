# L3.1 注意力机制 · 笔记

## 一、核心公式

### 四步计算

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

| 步 | 操作 | 目的 |
| --- | --- | --- |
| 1 | $S = QK^T$（点积） | 算 Q 和每个 K 的相似度 |
| 2 | $S / \sqrt{d_k}$ | 归一化方差，防 softmax 太极端 |
| 3 | $\text{softmax}(\cdot)$ | 相似度 → 概率权重（和为 1） |
| 4 | 加权 $V$ | 按权重取各 V 的加权平均 → 输出 |

### 术语精确命名（写代码时别混用）

| 位置 | 数学表达式 | 标准术语 |
| --- | --- | --- |
| 注意力中间结果 | $QK^T / \sqrt{d_k}$ | **attention scores**（注意力分数） |
| softmax 后 | $\text{softmax}(scores)$ | **attention weights**（注意力权重） |
| 加权 V 后 | $weights \cdot V$ | **attention output** / context vector |
| 模型最后一层 Linear | 未 softmax 的分数 | **logits**（特指分类输出） |

**attention scores 和 logits 都是"未归一化分数"，但语义完全不同**——前者是 token 间相似度，后者是类别原始分数。不要混用。

### Q/K/V 从哪来

同一个输入 $X$ 乘以三个不同的可学习矩阵：

$$
Q = X W^Q, \quad K = X W^K, \quad V = X W^V
$$

**$W^Q, W^K, W^V$ 都是随机初始化、训练时反向传播优化的参数**。同一个 $X$（词嵌入），三个不同的 $W$ 把它投射到三个不同的"空间"：

- $W_Q$：把词向量变成"问题空间"——这个 token 想问什么
- $W_K$：把词向量变成"标签空间"——这个 token 能被谁匹配
- $W_V$：把词向量变成"内容空间"——这个 token 携带什么信息

### 为什么要除以 √d_k（Scaling）

当 $d_k$ 大时，多个随机向量点积的方差 → ∞，softmax 输出趋近 one-hot（梯度几乎为 0）。除以 $\sqrt{d_k}$ 把方差归一化到 1，让 softmax 保持"软"。

## 二、为什么需要注意力——针对 RNN 的两大天花板

| 问题 | RNN | 注意力 |
| --- | --- | --- |
| 长距离依赖 | 梯度消失，指数衰减 | **一步直达**（Q·K 直接算所有对） |
| 并行训练 | 串行（必须等 $h_{t-1}$） | **全并行**（矩阵乘法一次算完所有对） |

这就是 2017 年 Transformer 淘汰 RNN 的根本原因。

## 三、Multi-Head Attention

**为什么多头**：单头只能学一种关系模式。8 个头同时关注不同模式（指代、修饰、主谓、介词……），最后拼起来——让模型从多个角度同时理解句子。

**实现**：把 d_model=512 切割成 8 份，每份 64 维，各头独立做注意力，最后拼回：

```python
# 拆头: (batch, seq, 512) → (batch, 8, seq, 64)
Q = Q.view(batch, seq, num_heads, d_k).transpose(1, 2)

# 各头独立注意力（scaled_dot_product_attention 自动对最后两维做）
attn_out, _ = scaled_dot_product_attention(Q, K, V)

# 拼回: (batch, 8, seq, 64) → (batch, seq, 512)
attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq, d_model)
```

**关键**：8 个头不是"额外创造 8 份数据"，只是把已有的 512 维重新解释为 8×64。每个头在自己那 64 维里独立做 Q·K·V 注意力。

## 四、Self-Attention vs Cross-Attention

| 类型 | Q 来源 | K, V 来源 | 用途 |
| --- | --- | --- | --- |
| Self-Attention | 同一句自身 | 同一句自身 | Encoder |
| Masked Self-Attn | 同一句自身 | 同一句自身（只看左侧） | Decoder |
| Cross-Attention | Decoder | Encoder 输出 | Encoder → Decoder |

**Self-Attention**：句子里的每个词都当一次 Query，对同一句话的所有词做注意力，得到带上下文的新表示。所有词同时做、并行。和普通 Attention 的区别：**所有词都做，不是只做一个词**。

## 五、Softmax 与 argmax（补充基础）

### argmax

`argmax = argument of the maximum`——返回最大值**所在的位置**，不是值本身。

- `[2, 1, 5].max() = 5` → 返回值
- `[2, 1, 5].argmax() = 2` → 返回位置（索引 2）

**训练 vs 推理**：

- 训练：logits → softmax → 概率分布 → 可导
- 推理：logits → argmax → 预测类别 → 不需要可导

### Softmax 公式

$$
\text{softmax}(x_i) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
$$

**关键性质**：输出是**向量**（不是单个数），每个元素 ∈ (0,1)，加起来 = 1。指数函数放大差异。可导，梯度极其简洁。

**温度参数**（大模型采样时的 slider）：

$$
\text{softmax}_T(x_i) = \frac{\exp(x_i / T)}{\sum_j \exp(x_j / T)}
$$

T 大 → 输出多样；T 小 → 输出保守。

### 为什么选 Softmax（三个核心原因）

- **指数放大差异**：对比线性归一化，softmax 让模型"更自信"，收敛更快
- **可导 + 梯度优雅**：梯度只依赖输出本身，反向传播极高效
- **最大熵原理**：唯一满足"输出概率 + 最大化熵 + 保留排序"的函数形式

### PyTorch 中的 Softmax

`nn.CrossEntropyLoss` 内部做了 softmax → log → NLL。**分类模型输出层只写 Linear，不加 softmax**。

### 替代品

| 函数 | 场景 | 流行度 |
| --- | --- | --- |
| Softmax | 多分类、注意力、LLM 采样 | ⭐⭐⭐⭐⭐ |
| Sigmoid | 二分类、门控机制 | ⭐⭐⭐⭐ |
| Sparsemax | 稀疏注意力 | ⭐⭐ |
| Gumbel-Softmax | 离散采样、强化学习 | ⭐⭐⭐ |

## 六、实现细节问答

### 为什么 K 要转置最后两维

矩阵乘法维度协议：Q (batch, seq_q, d_k) × Kᵀ (batch, d_k, seq_k) = (batch, seq_q, seq_k)。结果 (i, j) = Q 的第 i 行和 K 的第 j 行的点积。

### 为什么用 `transpose(-2, -1)` 而不是 `transpose(1, 2)`

负数索引：-1 = 最后一维，-2 = 倒数第二维。用 `-2, -1` 无论 Q/K/V 是 3 维还是 4 维都正确——**维数无关的工程写法**。

### 为什么 `attn_weights @ V` 叫"加权 V"而不是"乘 V"

数学上是矩阵乘法，但**语义是加权平均**。展开后每个输出 = w₁·V₁ + w₂·V₂ + ...，w 是 softmax 后的权重（和为 1）。所以叫 "weighted sum of values"。

### `Q @ Kᵀ` 是三维矩阵乘法吗

`@` 对 3D+ 张量：最后两维做矩阵乘，前面维度自动当成"独立批量"一一对应。batch 维不参与计算——**保证句子间不交互**，注意力只在句子内发生。

### `assert torch.allclose(...)` 在做什么

一行自动化测试：`allclose` 逐元素比较两个张量是否在误差范围内（atol=1e-5），`assert` 检查条件是否为真，否则程序崩溃。等价于"验证你的实现和官方一致"。

## 七、我的误解与纠正

| 我的理解 | 精确修正 |
| --- | --- |
| K 和 V 包括**词表中所有 token** | K 和 V 来自**当前输入序列**，不是整个词表。如 "cat sat mat" 只有 3 个词参与注意力 |
| Q 是"查"的 token 的属性，K/V 是"词表所有 token"的 | Q/K/V 全来自同一个输入 X，只是乘以不同的 W：`Q = XW^Q, K = XW^K, V = XW^V` |
| self-attention 和普通 attention 区别不清楚 | 单个词做注意力 = 普通 Attention；句中每个词都当一次 Query = **Self-Attention**。完整区分要对比 Encoder-Decoder 架构（L3.2） |
| Softmax 输出一个 0-1 的数 | 输出是**向量**，向量里每个数 ∈ (0,1)，加起来等于 1 |

## 八、实验产物

- 代码：[lab/train.py](lab/train.py)
- 验证：手写 Multi-Head Attention 和 PyTorch 官方 `nn.MultiheadAttention` 在相同权重下输出完全一致（`allclose` 通过，误差 < 1e-5）
- 性质：**验证实验，不是训练实验**——先确认公式正确，再嵌入完整 Transformer 训练（L3.2）

## 九、预告

L3.2 将把今天写的 attention 嵌入完整 Transformer：位置编码、Encoder/Decoder、mask、字符级语言模型训练、文本生成。L3.1 的验证是 L3.2 的地基——确认 attention 没问题，训练出问题时就不会怀疑是 attention 写错了。