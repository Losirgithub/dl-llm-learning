"""L2.1 实验：用 PyTorch autograd 重写 L1.3 的 Adam 优化。

对比 L1.3 手写版：
- L1.3: 手推梯度 grad = [10x, y]，手写 Adam 的 m、v、偏差修正
- L2.1: 只写损失函数(前向)，梯度和更新全部 PyTorch 自动完成

同一个"细长山谷"损失函数，同一个起点，看 PyTorch 版能不能达到相同效果。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# 引入课程通用工具(固定 seed 保证可复现)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.utils import set_seed  # noqa: E402


def loss_fn(theta: torch.Tensor) -> torch.Tensor:
    """L1.3 的细长山谷损失函数: L = 0.5 * (10x² + y²)。

    只写前向，不手推梯度——这是本实验的核心。
    """
    return 0.5 * (10.0 * theta[0] ** 2 + theta[1] ** 2)


def main() -> None:
    set_seed(42)

    # ---- 创建可训练参数(等价于 L1.3 的 start = [-2.0, 2.5])----
    # requires_grad=True 告诉 PyTorch: 追踪这个张量的所有运算,以便后续求梯度
    theta = torch.tensor([-2.0, 2.5], requires_grad=True)

    # ---- 用 PyTorch 内置的 Adam 优化器 ----
    # 只需告诉它要优化哪些参数,以及学习率,其他内部逻辑(m/v/偏差修正)自动完成
    optimizer = torch.optim.Adam([theta], lr=0.3)

    print(f"{'步':<5}{'x':<12}{'y':<12}{'loss':<12}")
    print("-" * 41)

    n_steps = 50
    for step in range(n_steps + 1):
        # ---- 前向传播: 算损失 ----
        loss = loss_fn(theta)

        # 打印前先记录当前状态(用 .item() 把标量 tensor 转成 Python float)
        if step % 5 == 0 or step == n_steps:
            print(
                f"{step:<5}{theta[0].item():+.4f}     "
                f"{theta[1].item():+.4f}     {loss.item():.6f}"
            )

        if step == n_steps:
            break  # 最后一步只打印,不再更新

        # ---- 训练标准 4 步 ----
        optimizer.zero_grad()  # 1. 清零上一步残留的梯度(必须!)
        loss.backward()        # 2. 反向传播: 自动算 loss 对 theta 的梯度
        optimizer.step()       # 3. 优化器更新: theta = theta - lr * grad

        # 注:zero_grad → backward → step 是深度学习训练的固定套路

    # ---- 和 L1.3 手写 Adam 对比 ----
    print("\n--- 对比 L1.3 手写 Adam ---")
    print("L1.3 手写 Adam 最终:  (+0.0386, -0.1424) loss=0.017578")
    print(
        f"L2.1 PyTorch  最终:  "
        f"({theta[0].item():+.4f}, {theta[1].item():+.4f}) "
        f"loss={loss.item():.6f}"
    )
    print("\n[✓] 只写前向,PyTorch 自动完成梯度计算和参数更新")


if __name__ == "__main__":
    main()
