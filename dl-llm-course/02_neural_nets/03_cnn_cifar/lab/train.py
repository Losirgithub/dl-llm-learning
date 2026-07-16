"""L2.3 实验：在 CIFAR-10 上训练 CNN。

你需要填 3 个 TODO。写完后运行：
    python train.py

目标：10 个 epoch 后验证准确率 > 75%。

对比：同结构 MLP 在 CIFAR-10 上只能到 45% 左右，你会亲眼看到 CNN 的威力。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 引入课程通用工具
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.data_utils import get_dataset_dir  # noqa: E402
from common.utils import set_seed  # noqa: E402


# --------------------------------------------------------------------------
# TODO 1: 定义 CNN 结构
# --------------------------------------------------------------------------
# 结构（见 README）：
#   Conv(3→32,k=3,p=1) → BN(32) → ReLU → MaxPool(2)
#   Conv(32→64,k=3,p=1) → BN(64) → ReLU → MaxPool(2)
#   Conv(64→128,k=3,p=1) → BN(128) → ReLU → MaxPool(2)
#   flatten → Linear(128*4*4, 256) → ReLU → Linear(256, 10)
#
# 建议：
#   - 卷积部分用 nn.Sequential 打包成 self.conv_layers（三个 CBR 块）
#   - 分类头用 nn.Sequential 打包成 self.classifier（Linear→ReLU→Linear）
#   - forward 里手动串起两部分（中间加 flatten）
class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # TODO 1a: 定义卷积部分 self.conv_layers（三个 CBR 块，每块后接 MaxPool）
        # 提示：nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     ... 再重复两组，通道数 32→64→128
        # )
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # raise NotImplementedError("请实现 CNN.__init__ 的 self.conv_layers")

        # TODO 1b: 定义分类头 self.classifier
        # 提示：Linear(128*4*4, 256) → ReLU → Linear(256, 10)
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),  # 三次池化把32x32减到4x4
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        参数：
            x: shape=(batch, 3, 32, 32)  CIFAR-10 彩色图

        返回：
            logits: shape=(batch, 10)
        """
        # TODO 2: 实现前向传播
        # 1. 过卷积部分 self.conv_layers
        # 2. flatten 成 (batch, 2048) —— 用 x.flatten(1)
        # 3. 过分类头 self.classifier
        # 4. 返回 logits
        x = self.conv_layers(x)
        x = x.flatten(1)
        logits = self.classifier(x)
        return logits
        # raise NotImplementedError("请实现 CNN.forward")


# --------------------------------------------------------------------------
# 数据加载（已给，不用改）
# --------------------------------------------------------------------------
def get_dataloaders(batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    """加载 CIFAR-10 训练集和测试集。"""
    # CIFAR-10 的官方均值方差（3 通道各一个）
    # ---- transform:每张图读出来后要做的预处理 ----
    mean = (0.4914, 0.4822, 0.4465)  # CIFAR-10 三通道各自的均值
    std = (0.2470, 0.2435, 0.2616)  # 各自的标准差
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # PIL 图 → tensor [0, 1]
            transforms.Normalize(mean, std),  # 归一化到均值 0 方差 1
        ]
    )
    # ---- 数据集本身(用课程根 data/,所有实验共享,避免重复占空间) ----
    data_dir = get_dataset_dir("CIFAR10")
    train_set = datasets.CIFAR10(str(data_dir), train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(str(data_dir), train=False, download=True, transform=transform)
    # ---- DataLoader 包装 ----
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


# --------------------------------------------------------------------------
# 训练与评估
# --------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """训练一个 epoch，返回平均训练损失。"""
    model.train()  # 训练模式：BatchNorm 用当前 batch 统计
    total_loss = 0.0
    n_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # TODO 3: 训练标准 5 步（复用 L2.2 的模板）
        optimizer.zero_grad()  # 1. 梯度清零
        logits = model(x)  # 2. 前向传播
        loss = loss_fn(logits, y)  # 3. 计算损失
        loss.backward()  # 4. 反向传播
        optimizer.step()  # 5. 更新参数

        # raise NotImplementedError("请实现训练主循环 5 步")

        total_loss += loss.item()  # noqa: F821
        n_batches += 1
    return total_loss / n_batches


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """在验证集上评估，返回准确率。"""
    model.eval()  # 评估模式：BatchNorm 用累积统计
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        predicted = logits.argmax(dim=1)
        correct += (predicted == y).sum().item()
        total += y.size(0)
    return correct / total


def main() -> None:
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    batch_size = 128
    lr = 1e-3
    n_epochs = 10  # CIFAR-10 比 MNIST 难，多训几个 epoch

    train_loader, test_loader = get_dataloaders(batch_size)
    model = CNN().to(device)

    # 打印模型结构和参数量（可选，帮助你理解）
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\n{'Epoch':<8}{'Train Loss':<15}{'Val Acc':<10}")
    print("-" * 33)
    best_acc = 0.0
    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_acc = evaluate(model, test_loader, device)
        marker = " ★" if val_acc > best_acc else ""
        best_acc = max(best_acc, val_acc)
        print(f"{epoch:<8}{train_loss:<15.4f}{val_acc:<10.4f}{marker}")

    print(f"\n[✓] 训练完成，最终 val acc: {val_acc:.2%}，历史最佳: {best_acc:.2%}")
    if best_acc > 0.75:
        print("[✓] 达到 75% 目标!")
    else:
        print("[!] 未达到 75%，检查实现或增加 epoch")


if __name__ == "__main__":
    main()
