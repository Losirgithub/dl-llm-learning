# Lab: 从零手写注意力机制

## 目标

- 亲手实现 scaled dot-product attention 公式
- 实现 Multi-Head Attention
- 用 PyTorch 官方 `nn.MultiheadAttention` 验证你的实现数学上正确

## 要求

打开 `train.py`，找到 4 个 `# TODO`：

1. **TODO 1**：实现单头注意力四步公式
2. **TODO 2**：实现单个注意力头的 forward(含 Q/K/V 线性变换)
3. **TODO 3**：实现 Multi-Head Attention(拆头→各头独立注意力→拼回)
4. **TODO 4**：验证脚本——对比你的实现和 PyTorch 官方版的输出

## 运行环境

- 推荐机器：3070 笔记本（CPU 也可）
- 预估显存：0
- 预估时长：< 5 秒

## 验收标准

- [ ] 代码跑通，打印 "All checks passed"
- [ ] 单头注意力的输出和 PyTorch 官方版**逐元素误差 < 1e-5**
- [ ] 能解释 4 个 TODO 每一步在做什么

## 公式速查

### Scaled Dot-Product Attention

```text
Attention(Q, K, V) = softmax(Q·Kᵀ / √dₖ) · V
```

### Multi-Head Attention

```text
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) · W_O

head_i = Attention(Q·W_Qⁱ, K·W_Kⁱ, V·W_Vⁱ)
```

## 提示

- **分母是 √dₖ**，其中 dₖ 是**每个头的维度**(不是总维度)
- **Softmax 沿最后一维(dim=-1)**——对每个 Query 的所有 Key 算相似度
- **Multi-Head 中的 reshape**：把 (batch, seq, d_model) 拆成 (batch, seq, num_heads, d_k)，再 transpose 让头维度靠前方便并行
