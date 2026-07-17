"""L3.1 实验：从零手写注意力机制。

亲手实现 scaled dot-product attention + Multi-Head Attention，
用 PyTorch 官方 nn.MultiheadAttention 验证数学正确性。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.utils import set_seed  # noqa: E402


# ---------------------------------------------------------------------------
# TODO 1: 实现 Scaled Dot-Product Attention
# ---------------------------------------------------------------------------
def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """单头注意力的核心公式。

    参数:
        Q: shape (..., seq_q, d_k)    Query
        K: shape (..., seq_k, d_k)    Key
        V: shape (..., seq_k, d_v)    Value
        mask: shape (... or 1, 1 or seq_q, seq_k),可选,值为 0 的位置不参与

    返回:
        output: shape (..., seq_q, d_v)   加权 V
        attn_weights: shape (..., seq_q, seq_k)  注意力权重矩阵(softmax 后)
    """
    d_k = Q.size(-1)

    # TODO 1a: Q · Kᵀ / √d_k —— 相似度 + 缩放
    # 提示: K.transpose(-2, -1) 把最后两维转置
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # raise NotImplementedError("TODO 1a: 算相似度分数 scores = Q·Kᵀ / √d_k")

    # TODO 1b: 如果有 mask,把 mask==0 的位置分数置为-inf(或极小值 -1e9)
    # 提示: scores.masked_fill(mask == 0, float("-inf"))
    # 为什么置 -inf: softmax(-inf) = 0,这些位置完全不被注意
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    # TODO 1c: softmax 沿最后一维,把分数变成权重
    attn_weights = F.softmax(scores, dim=-1)
    # TODO 1d: 权重加权 V
    output = torch.matmul(attn_weights, V)
    return output, attn_weights  # noqa: F821


# ---------------------------------------------------------------------------
# TODO 2: 单头注意力模块(包装 Q/K/V 线性变换 + 注意力)
# ---------------------------------------------------------------------------
class SingleHeadAttention(nn.Module):
    def __init__(self, d_model: int, d_k: int) -> None:
        super().__init__()
        # d_model: 输入 token 的特征维度(即 embedding 维度),如 512。
        #          这是整个 Transformer 的核心维度,所有子层的输入输出都保持
        #          d_model 维(残差连接要求维度一致),典型值 512/768/1024。
        # d_k:    Q/K/V 线性投影后的维度(单头注意力的维度),通常 d_k <= d_model。
        # Q、K、V 都是从同一个输入 x 通过可学习的线性变换得到的
        # 每个变换有独立的权重矩阵,训练时反向传播优化它们
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_k, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, d_model),返回注意力输出 (batch, seq, d_k)。"""
        # TODO 2: 先做 Q/K/V 线性变换,再调用 scaled_dot_product_attention
        # 提示:
        #   Q = self.W_Q(x)
        #   K = self.W_K(x)
        #   V = self.W_V(x)
        #   output, _ = scaled_dot_product_attention(Q, K, V)
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        output, _ = scaled_dot_product_attention(Q, K, V)
        return output
        # raise NotImplementedError("TODO 2: 实现 SingleHeadAttention.forward")


# ---------------------------------------------------------------------------
# TODO 3: Multi-Head Attention
# ---------------------------------------------------------------------------
class MyMultiHeadAttention(nn.Module):
    """手写 Multi-Head Attention。

    把 d_model 拆成 h 个头,每个头维度 d_k=d_model/h,
    各头独立做注意力后拼回,再过一个线性层 W_O。
    """

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 所有头的 Q/K/V 合并成一个大的线性变换(效率更高,和 PyTorch 官方一致)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, d_model),返回 (batch, seq, d_model)。"""
        batch, seq, _ = x.shape

        # 第一步: Q/K/V 线性变换
        Q = self.W_Q(x)  # (batch, seq, d_model)
        K = self.W_K(x)
        V = self.W_V(x)

        # TODO 3a: 拆头 —— 把 d_model 拆成 (num_heads, d_k)
        # 目标形状: (batch, num_heads, seq, d_k) —— 这样多头可以并行算 attention
        # 提示:
        #   1. view 成 (batch, seq, num_heads, d_k)
        #   2. transpose(1, 2) 把 num_heads 提到前面
        Q = Q.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        # raise NotImplementedError("TODO 3a: 拆头 reshape")

        # TODO 3b: 各头独立做注意力 —— 直接调 scaled_dot_product_attention
        # 因为 Q, K, V 形状已经是 (batch, num_heads, seq, d_k),
        # 函数内部 O @ Kᵀ 会自然沿 d_k 做点积、沿 seq 算权重、沿每个头独立并行

        attn_out, _ = scaled_dot_product_attention(Q, K, V)

        # TODO 3c: 拼回头 —— 把 (batch, num_heads, seq, d_k) 变回 (batch, seq, d_model)
        # 提示:
        #   1. transpose(1, 2) 把 seq 恢复到第 1 维 → (batch, seq, num_heads, d_k)
        #   2. contiguous().view(batch, seq, d_model) 拼回头
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq, self.d_model)

        # 第四步: 输出线性变换
        return self.W_O(attn_out)  # noqa: F821


# ---------------------------------------------------------------------------
# TODO 4: 验证 —— 和 PyTorch 官方版对比
# ---------------------------------------------------------------------------
def main() -> None:
    set_seed(42)

    d_model = 512  # 输入特征维度(embedding 维度),Transformer 的核心维度
    num_heads = 8
    batch, seq = 4, 10
    x = torch.randn(batch, seq, d_model)

    # ---- 创建两个 MultiHeadAttention,并让它们权重“完全相等” ----
    official = nn.MultiheadAttention(d_model, num_heads, batch_first=True, bias=False)
    mine = MyMultiHeadAttention(d_model, num_heads)

    # PyTorch 官方把 Q、K、V 合在一个大矩阵里 (in_proj_weight),拆给我们的 W_Q/W_K/W_V
    w_qkv: torch.Tensor = official.in_proj_weight  # shape: (3*d_model, d_model)
    w_q, w_k, w_v = w_qkv.chunk(3, dim=0)  # 各 (d_model, d_model)
    mine.W_Q.weight.data.copy_(w_q)
    mine.W_K.weight.data.copy_(w_k)
    mine.W_V.weight.data.copy_(w_v)
    mine.W_O.weight.data.copy_(official.out_proj.weight.data)

    # TODO 4a: 分别跑官方和你的实现
    # 提示:
    #   official_out, _ = official(x, x, x)        # PyTorch MHA: (query, key, value)
    #   mine_out = mine(x)                          # 你的 MHA
    official_out, _ = official(x, x, x)
    mine_out = mine(x)

    # TODO 4b: 对比输出,允许误差 < 1e-5
    # 提示: 用 torch.allclose(official_out, mine_out, atol=1e-5)
    # raise NotImplementedError("TODO 4: 验证")
    assert torch.allclose(official_out, mine_out, atol=1e-5), "Outputs do not match"
    # 如果上面通过了,这里打印 "All checks passed"
    print("[✓] All checks passed —— 你的 MultiHeadAttention 和 PyTorch 官方一致")  # noqa: F821


if __name__ == "__main__":
    main()
