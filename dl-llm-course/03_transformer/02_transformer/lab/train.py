"""L3.2 实验:手写 Transformer 积木 + Encoder Block。

实现 Positional Encoding、FeedForward、EncoderBlock,验证形状正确。
Decoder 和完整训练放下一轮。
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
# 从 L3.1 复制来的 MultiHeadAttention(已验证正确)
# ---------------------------------------------------------------------------
def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        Q = self.W_Q(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        attn_out = scaled_dot_product_attention(Q, K, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq, self.d_model)
        return self.W_O(attn_out)


# ---------------------------------------------------------------------------
# TODO 1: Positional Encoding
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """给输入加位置编码: x = x + PE[:seq_len]

    PE 用 sin/cos 公式计算,是固定值(不是可训练参数)。
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # TODO 1: 计算 PE 矩阵 (max_len, d_model)
        # 提示:
        #   1. position = torch.arange(0, max_len).unsqueeze(1).float()  -> (max_len, 1)
        #   2. div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        #      这是一个 (d_model//2,) 的向量,对应公式里的 10000^(2i/d)
        #      用 exp(-log(...)) 是数值稳定的写法,等价于 1/10000^(2i/d)
        #   3. pe[:, 0::2] = torch.sin(position * div_term)   # 偶数位填 sin
        #   4. pe[:, 1::2] = torch.cos(position * div_term)   # 奇数位填 cos
        #   5. pe = pe.unsqueeze(0)  -> (1, max_len, d_model),前面加 batch 维方便广播
        #   6. self.register_buffer('pe', pe)  # 不是参数,不训练,但跟着模型搬 GPU
        position = (
            torch.arange(0, max_len).unsqueeze(1).float()
        )  # unsqueeze(1) -> (max_len, 1) 竖着的一列，max_len行
        pe = torch.zeros(max_len, d_model)
        # torch.arange(0, d_model, 2):生成 [0, 2, 4, ..., d_model-2],形状 (d_model//2,)--这就是公式里的 2i(i=0,1,2,...时,2i=0,2,4,...)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # 为什么要绕一圈用 exp/log:数值稳定。直接算 10000^(大数) 会溢出,exp(-log(...)) 不会。这是深度学习代码常见的"数值稳定技巧"。
        pe[:, 0::2] = torch.sin(position * div_term)  # 0开始步长为2
        pe[:, 1::2] = torch.cos(position * div_term)  # 1开始步长为2
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)  # 不是参数,不训练,但跟着模型搬 GPU
        # raise NotImplementedError("TODO 1: 实现 sin/cos 位置编码")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, d_model),返回 (batch, seq, d_model)。"""
        # x + PE 的前 seq_len 行(靠广播自动对齐 batch 维)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# TODO 2: Feed-Forward Network
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    """两层 Linear + GELU: d_model -> 4*d_model -> d_model。"""

    def __init__(self, d_model: int, d_ff: int | None = None) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model  # 默认扩展 4 倍

        # TODO 2: 用 nn.Sequential 定义两层 Linear + 中间 GELU
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        # raise NotImplementedError("TODO 2: 实现 FeedForward 结构")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# TODO 3: Encoder Block(核心--两个 Add & Norm)
# ---------------------------------------------------------------------------
class EncoderBlock(nn.Module):
    """一个 Transformer Encoder Block = MHA + Add&Norm + FFN + Add&Norm。"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, d_model),返回 (batch, seq, d_model)。

        结构:
            ┌── x ──────────────────┐
            ↓                       │
        attention(x)                │ (残差)
            ↓                       │
        dropout                     │
            + ←─────────────────────┘
            ↓
          norm1  ── Add & Norm 1
            │
            ├── x ──────────────────┐
            ↓                       │
          ffn(x)                    │ (残差)
            ↓                       │
        dropout                     │
            + ←─────────────────────┘
            ↓
          norm2  ── Add & Norm 2
        """
        # TODO 3a: 第一个 Add & Norm(attention 子层)
        # 提示:
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        # raise NotImplementedError("TODO 3a: attention 的 Add & Norm")

        # TODO 3b: 第二个 Add & Norm(FFN 子层)
        # 提示:
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        # raise NotImplementedError("TODO 3b: FFN 的 Add & Norm")

        return x


# ---------------------------------------------------------------------------
# 验证:形状检查
# ---------------------------------------------------------------------------
def main() -> None:
    set_seed(42)

    d_model = 512
    num_heads = 8
    batch, seq = 4, 10

    x = torch.randn(batch, seq, d_model)
    print(f"输入 shape: {x.shape}")

    # 1. PositionalEncoding
    pe = PositionalEncoding(d_model)
    x_pe = pe(x)
    assert x_pe.shape == x.shape, f"PE 形状错: {x_pe.shape}"
    print(f"[✓] PositionalEncoding: {x_pe.shape}")

    # 2. FeedForward
    ffn = FeedForward(d_model)
    x_ffn = ffn(x)
    assert x_ffn.shape == x.shape, f"FFN 形状错: {x_ffn.shape}"
    print(f"[✓] FeedForward: {x_ffn.shape}")

    # 3. EncoderBlock
    block = EncoderBlock(d_model, num_heads)
    x_out = block(x)
    assert x_out.shape == x.shape, f"EncoderBlock 形状错: {x_out.shape}"
    print(f"[✓] EncoderBlock: {x_out.shape}")

    # 4. 堆叠多个 block(模拟完整 Encoder)
    n_layers = 6
    encoder = nn.Sequential(*[EncoderBlock(d_model, num_heads) for _ in range(n_layers)])
    x_final = encoder(x_pe)
    assert x_final.shape == x.shape, f"堆叠 Encoder 形状错: {x_final.shape}"
    print(f"[✓] 6 层 Encoder 堆叠: {x_final.shape}")

    # 5. 参数量
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"\n6 层 Encoder 参数量: {n_params:,}")
    print(f"(对比: 单层 MLP 在 MNIST 上约 235,000 参数)")

    print("\n[✓] All shape checks passed -- 积木 + Encoder Block 完成")


if __name__ == "__main__":
    main()
