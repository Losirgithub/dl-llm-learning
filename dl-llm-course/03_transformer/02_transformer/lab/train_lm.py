"""L3.2 后半场:Decoder-Only Transformer + 字符级语言模型训练。

在已有积木上加:DecoderBlock(mask)、完整模型、训练、文本生成。
这就是 GPT 的结构--训出来能生成文本。
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


# ===========================================================================
# 积木(从前面复制,自包含)
# ===========================================================================
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq, _ = x.shape
        Q = self.W_Q(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        attn_out = scaled_dot_product_attention(Q, K, V, mask)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq, self.d_model)
        return self.W_O(attn_out)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ===========================================================================
# TODO 1: DecoderBlock(带 mask 的版本)
# ===========================================================================
class DecoderBlock(nn.Module):
    """GPT 风格的 Decoder Block = Masked Self-Attention + Add&Norm + FFN + Add&Norm。

    和 EncoderBlock 唯一区别:attention 要接收 mask(防止偷看未来)。
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, d_model), mask: (1, 1, seq, seq) 或 (batch, 1, seq, seq)。

        和 EncoderBlock 几乎一样,只是 attention 要传 mask。
        """
        # TODO 1: 两个 Add & Norm,attention 那步要传 mask
        # 提示(和 EncoderBlock 对比看,就多一个 mask):
        #   attn_out = self.attention(x, mask)        ← 传 mask!
        #   x = self.norm1(x + self.dropout(attn_out))
        #   ffn_out = self.ffn(x)
        #   x = self.norm2(x + self.dropout(ffn_out))
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x
        # raise NotImplementedError("TODO 1: DecoderBlock forward(带 mask)")


# ===========================================================================
# TODO 2: 完整 Decoder-Only Transformer
# ===========================================================================
class DecoderOnlyTransformer(nn.Module):
    """GPT 结构:词嵌入 -> 位置编码 -> N 个 DecoderBlock -> 输出层。

    输出层把 d_model 映射回词表大小,得到每个位置对每个字符的 logits。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 4,
        n_layers: int = 4,
        max_len: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, num_heads, dropout) for _ in range(n_layers)]
        )
        self.norm_final = nn.LayerNorm(d_model)  # 最后过一层 LayerNorm(GPT-2 风格)
        self.lm_head = nn.Linear(d_model, vocab_size)  # 输出层:映射回词表

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (batch, seq) 整数 token id,返回 logits (batch, seq, vocab_size)。"""
        batch, seq = idx.shape

        # TODO 2: 完整前向
        # 提示:
        #   1. x = self.token_embedding(idx)        # (batch, seq, d_model) 整数->向量
        #   2. x = self.pos_encoding(x)             # 加位置编码
        #   3. 构造 causal mask: mask = torch.tril(torch.ones(seq, seq, device=idx.device))
        #      然后 mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq) 方便广播
        #   4. for block in self.blocks: x = block(x, mask)   # 逐层过 DecoderBlock
        #   5. x = self.norm_final(x)
        #   6. logits = self.lm_head(x)             # 映射回词表大小
        #   7. return logits
        x = self.token_embedding(idx)
        x = self.pos_encoding(x)
        mask = torch.tril(torch.ones(seq, seq, device=idx.device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm_final(x)
        logits = self.lm_head(x)
        return logits
        # raise NotImplementedError("TODO 2: 完整 Transformer forward")


# ===========================================================================
# 字符级数据准备(已给,不用改)
# ===========================================================================
# 一段英文文本(莎士比亚风格,公共领域),用来训练字符级语言模型
TRAIN_TEXT = """
To be, or not to be, that is the question:
Whether tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die, to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: tis a consummation
Devoutly to be wishd. To die, to sleep,
To sleep, perchance to dream; ay, theres the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause.
""".strip()


class CharDataset:
    """字符级数据集:把文本切成 (input, target) 对,target 是 input 右移一位。"""

    def __init__(self, text: str, block_size: int) -> None:
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}  # 字符 -> id
        self.itos = {i: ch for i, ch in enumerate(chars)}  # id -> 字符
        self.vocab_size = len(chars)
        self.block_size = block_size
        self.data = [self.stoi[c] for c in text]  # 整篇文本转成 id 序列

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        # input: data[i : i+block_size], target: data[i+1 : i+block_size+1]
        # target 是 input 右移一位 -> 预测下一个字符
        chunk = self.data[i : i + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


# ===========================================================================
# 训练循环(已给,不用改--和 L2.2 模板一样)
# ===========================================================================
def train(
    model: nn.Module, dataset: CharDataset, n_epochs: int, lr: float, device: torch.device
) -> None:
    model.train()  # 切换到训练模式
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # 用整个数据集当一个 batch(文本短,能放下)
    all_x = torch.stack([dataset[i][0] for i in range(len(dataset))]).to(device)
    all_y = torch.stack([dataset[i][1] for i in range(len(dataset))]).to(device)

    print(f"数据集: {len(dataset)} 个样本,词表大小: {dataset.vocab_size}")
    print(f"\n{'Epoch':<8}{'Loss':<12}")
    print("-" * 20)
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        logits = model(all_x)  # (n_samples, block_size, vocab_size)
        # CrossEntropyLoss 要 (N, C, ...) 的形状,所以把 batch 和 seq 合并
        loss = F.cross_entropy(logits.reshape(-1, dataset.vocab_size), all_y.reshape(-1))
        loss.backward()
        optimizer.step()
        if epoch % 200 == 0 or epoch == 1:
            print(f"{epoch:<8}{loss.item():<12.4f}")

    print(f"\n[✓] 训练完成,最终 loss: {loss.item():.4f}")


# ===========================================================================
# TODO 3: 文本生成(自回归采样)
# ===========================================================================
@torch.no_grad()
def generate(
    model: DecoderOnlyTransformer,
    dataset: CharDataset,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
) -> str:
    """从 prompt 开始,自回归生成 max_new_tokens 个字符。

    流程:
      1. prompt 转成 token id 序列
      2. 循环 max_new_tokens 次:
         a. 用模型前向,拿到最后一个位置的 logits
         b. logits / temperature 后过 softmax 得到概率
         c. 按概率采样一个字符 id
         d. 拼到序列末尾
      3. 把最终序列转回字符串
    """
    model.eval()
    device = next(model.parameters()).device  # 模型在哪个设备(GPU/CPU)
    # prompt 转成 id 序列
    idx = [dataset.stoi[c] for c in prompt if c in dataset.stoi]
    idx = torch.tensor([idx], dtype=torch.long).to(device)  # (1, seq) 搬到模型所在设备

    # TODO 3: 自回归生成循环
    # 提示:
    for _ in range(max_new_tokens):
        logits = model(idx)  # (1, seq, vocab_size)
        last_logits = logits[:, -1, :]  # (1, vocab_size) 只取最后一个位置
        probs = F.softmax(last_logits / temperature, dim=-1)  # 加温度
        next_id = torch.multinomial(probs, num_samples=1)  # 按概率采样
        idx = torch.cat([idx, next_id], dim=1)  # 拼到末尾
    # raise NotImplementedError("TODO 3: 自回归生成")

    # 把 id 序列转回字符串
    return "".join(dataset.itos[i.item()] for i in idx[0])


# ===========================================================================
# 主程序
# ===========================================================================
def main() -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    block_size = 64  # 每个样本 64 个字符
    dataset = CharDataset(TRAIN_TEXT, block_size)

    model = DecoderOnlyTransformer(
        vocab_size=dataset.vocab_size,
        d_model=128,
        num_heads=4,
        n_layers=4,
        max_len=512,  # 比 block_size 大,留足生成时的序列长度
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")

    # 训练
    train(model, dataset, n_epochs=2000, lr=3e-4, device=device)

    # 生成
    print("\n--- 生成示例 1 ---")
    print(generate(model, dataset, prompt="To be", max_new_tokens=200, temperature=0.7))

    print("\n--- 生成示例 2 (温度高,更多样) ---")
    print(generate(model, dataset, prompt="The ", max_new_tokens=200, temperature=1.0))


if __name__ == "__main__":
    main()
