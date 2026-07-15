"""L1.3 实验：四种优化器的收敛路径对比。

在一个"细长山谷"型二次曲面上，从同一起点出发，用 GD/SGD/Momentum/Adam 分别优化，
画出各自的收敛路径。这种病态曲面最能暴露四种优化器的差异。

运行：python train.py
产出：optimizer_paths.png（保存到本目录）
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 引入课程通用工具（固定 seed 保证可复现）
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.utils import set_seed  # noqa: E402


# --------------------------------------------------------------------------
# 损失函数：一个"细长山谷"
# L(x, y) = 0.5 * (10 * x^2 + y^2)
# x 方向"陡"(系数10)，y 方向"平"(系数1) —— 这就是病态曲面
# 最小值在 (0, 0)
# --------------------------------------------------------------------------
def loss(x: float, y: float) -> float:
    """损失函数值。"""
    return 0.5 * (10.0 * x**2 + y**2)


def grad(x: float, y: float) -> np.ndarray:
    """损失函数对 (x, y) 的梯度。"""
    return np.array([10.0 * x, y])


# --------------------------------------------------------------------------
# 优化器实现（手写，不用 torch.optim）
# 所有优化器都返回收敛路径 [(x0,y0), (x1,y1), ...]，方便后面画图
# --------------------------------------------------------------------------
def optimize_gd(
    start: np.ndarray, lr: float, n_steps: int, add_noise: bool = False
) -> np.ndarray:
    """朴素梯度下降 GD；add_noise=True 时模拟 SGD 的 mini-batch 噪声。"""
    theta = start.copy()
    path = [theta.copy()]
    for _ in range(n_steps):
        g = grad(theta[0], theta[1])
        if add_noise:
            # 模拟 SGD：给梯度加高斯噪声（真实 SGD 的噪声来自 mini-batch 采样）
            g = g + np.random.randn(2) * 3.0
        theta = theta - lr * g
        path.append(theta.copy())
    return np.array(path)


def optimize_momentum(
    start: np.ndarray, lr: float, n_steps: int, beta: float = 0.9
) -> np.ndarray:
    """Momentum：累积历史梯度，形成惯性。"""
    theta = start.copy()
    v = np.zeros(2)  # 速度变量
    path = [theta.copy()]
    for _ in range(n_steps):
        g = grad(theta[0], theta[1])
        v = beta * v - lr * g  # 累积速度
        theta = theta + v
        path.append(theta.copy())
    return np.array(path)


def optimize_adam(
    start: np.ndarray,
    lr: float,
    n_steps: int,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> np.ndarray:
    """Adam：Momentum + 每个参数自适应学习率。"""
    theta = start.copy()
    m = np.zeros(2)  # 一阶矩（动量）
    v = np.zeros(2)  # 二阶矩（梯度平方的滑动平均）
    path = [theta.copy()]
    for t in range(1, n_steps + 1):
        g = grad(theta[0], theta[1])
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        # 偏差修正（早期 m,v 偏小，修正一下）
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)
        path.append(theta.copy())
    return np.array(path)


# --------------------------------------------------------------------------
# 可视化
# --------------------------------------------------------------------------
def plot_paths(paths: dict[str, np.ndarray], save_path: Path) -> None:
    """在等高线图上画出四种优化器的收敛路径。"""
    # 等高线网格
    x_range = np.linspace(-2.5, 2.5, 200)
    y_range = np.linspace(-3.0, 3.0, 200)
    X, Y = np.meshgrid(x_range, y_range)
    Z = 0.5 * (10.0 * X**2 + Y**2)

    fig, ax = plt.subplots(figsize=(10, 8))
    # 用 log 尺度的等高线，让山谷形状更明显
    levels = np.logspace(-1, 2, 20)
    ax.contour(X, Y, Z, levels=levels, cmap="gray", alpha=0.4, linewidths=0.8)

    colors = {"GD": "tab:blue", "SGD": "tab:orange", "Momentum": "tab:green", "Adam": "tab:red"}
    for name, path in paths.items():
        ax.plot(
            path[:, 0], path[:, 1],
            "-o", color=colors[name], markersize=3, linewidth=1.5, alpha=0.85,
            label=f"{name} ({len(path)-1} steps)",
        )
        # 起点标记
        ax.plot(path[0, 0], path[0, 1], "k*", markersize=15, zorder=5)

    ax.plot(0, 0, "r+", markersize=20, markeredgewidth=3, label="最优点 (0,0)")
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title("四种优化器在细长山谷上的收敛路径\nL(x,y) = 0.5·(10x² + y²)", fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-3.0, 3.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"[✓] 图已保存到: {save_path}")


def main() -> None:
    set_seed(42)  # 固定 seed，保证 SGD 的噪声可复现

    # 所有优化器：同一起点，同一步数，同一学习率基准
    start = np.array([-2.0, 2.5])
    n_steps = 50
    lr = 0.08

    paths = {
        "GD": optimize_gd(start, lr, n_steps, add_noise=False),
        "SGD": optimize_gd(start, lr, n_steps, add_noise=True),
        "Momentum": optimize_momentum(start, lr, n_steps),
        "Adam": optimize_adam(start, lr=0.3, n_steps=n_steps),  # Adam 通常用更大 lr
    }

    # 打印每种优化器的最终损失（收敛程度）
    print("\n各优化器最终位置与损失：")
    print(f"{'优化器':<12}{'最终 (x, y)':<25}{'最终损失':<15}")
    print("-" * 52)
    for name, path in paths.items():
        final = path[-1]
        final_loss = loss(final[0], final[1])
        print(f"{name:<12}({final[0]:+.4f}, {final[1]:+.4f})     {final_loss:.6f}")

    # 画图
    out_path = Path(__file__).parent / "optimizer_paths.png"
    plot_paths(paths, out_path)


if __name__ == "__main__":
    main()
