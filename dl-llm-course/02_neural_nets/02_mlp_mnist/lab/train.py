"""L2.2 实验：在 MNIST 上训练 MLP。

你需要填 4 个 TODO。写完后运行：
    python train.py

目标：5 个 epoch 后验证准确率 > 97%。
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
# TODO 1: 定义 MLP 结构
# --------------------------------------------------------------------------
# 结构要求：
#   输入 784 → Linear → 256 → ReLU → Linear → 128 → ReLU → Linear → 10 输出
# 提示：
#   - 继承 nn.Module
#   - 在 __init__ 里定义各层（nn.Linear、nn.ReLU）
#   - 也可以用 nn.Sequential 把层串起来，代码更简洁
#   - MNIST 图 28×28，输入是 784(展平后)
class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # TODO 1: 在这里定义你的网络层
        # 例：self.fc1 = nn.Linear(784, 256)
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        参数：
            x: 输入张量，shape=(batch_size, 1, 28, 28) 或已展平的 (batch_size, 784)

        返回：
            logits: shape=(batch_size, 10)，10 类的原始分数(未 softmax)
        """
        # TODO 2: 实现前向传播
        # 提示：先把 x 展平成 (batch_size, 784)，用 x.view(-1, 784) 或 x = x.flatten(1)
        # 然后依次过你在 __init__ 里定义的层
        x = x.view(-1, 784)  # 展平
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        logits = self.fc3(x)  # 工业界代码里"未 softmax 的输出"就叫 logits
        return logits
        # raise NotImplementedError("请实现 MLP.forward")


# --------------------------------------------------------------------------
# 数据加载(已给,不用改)
# --------------------------------------------------------------------------
def get_dataloaders(batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    """加载 MNIST 训练集和测试集。"""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # PIL 图 → tensor,像素 [0,1]
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST 官方均值方差归一化
        ]
    )
    data_dir = get_dataset_dir("MNIST")
    train_set = datasets.MNIST(str(data_dir), train=True, download=True, transform=transform)
    test_set = datasets.MNIST(str(data_dir), train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
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
    model.train()
    total_loss = 0.0
    n_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # TODO 3: 完成训练一步的 4 步
        # 回顾 L2.1 笔记的"训练标准 4 步"：
        #   1. optimizer.zero_grad() —— 清零上次的梯度
        #   2. y_pred = model(x)     —— 前向：算预测
        #   3. loss = loss_fn(...)   —— 前向：算损失
        #   4. loss.backward()       —— 反向：算梯度
        #   5. optimizer.step()      —— 更新参数
        # 把这五步照实写下来(第 2、3 步是一起的前向)
        optimizer.zero_grad()  # 1. 清零梯度
        y_pred = model(x)  # 2. 前向：算预测
        loss = loss_fn(y_pred, y)  # 3. 前向：算损失
        loss.backward()  # 4. 反向：算梯度
        optimizer.step()  # 5. 更新参数
        # raise NotImplementedError("请实现训练主循环 4 步")

        total_loss += loss.item()  # noqa: F821  你实现完 TODO 3 后 loss 就有定义了
        n_batches += 1
    # 返回值:这一整个 epoch 的平均训练损失
    return total_loss / n_batches


@torch.no_grad()  # 评估不需要梯度，节省内存和速度
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """在验证集上评估，返回准确率(0-1)。"""
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)  # shape=(batch, 10)

        # TODO 4: 计算这个 batch 的正确数量
        # 提示：
        #   - 预测类别 = logits.argmax(dim=1)  形状 (batch,)
        #   - 和 y 逐元素比较,再 .sum().item() 得到正确数
        #   - 把它累加到 correct;把 batch 大小累加到 total
        predicted = logits.argmax(dim=1)  # 预测类别，argmax 返回最大值的索引
        correct += (predicted == y).sum().item()  # 累加正确
        total += y.size(0)  # 累加总数
        # raise NotImplementedError("请实现准确率计算")

    return correct / total


def main() -> None:
    set_seed(42)

    # 设备：有 GPU 用 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 超参数
    batch_size = 128
    lr = 1e-3
    n_epochs = 5

    # 数据、模型、损失、优化器
    train_loader, test_loader = get_dataloaders(batch_size)
    model = MLP().to(device)
    loss_fn = nn.CrossEntropyLoss()  # 内置 softmax + NLL, 直接吃 logits
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 训练
    print(f"\n{'Epoch':<8}{'Train Loss':<15}{'Val Acc':<10}")
    print("-" * 33)
    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_acc = evaluate(model, test_loader, device)
        print(f"{epoch:<8}{train_loss:<15.4f}{val_acc:<10.4f}")

    print(f"\n[✓] 训练完成，最终验证准确率: {val_acc:.2%}")
    if val_acc > 0.97:
        print("[✓] 达到 97% 目标!")
    else:
        print("[!] 未达到 97%，检查你的实现或增加 epoch")


if __name__ == "__main__":
    main()
