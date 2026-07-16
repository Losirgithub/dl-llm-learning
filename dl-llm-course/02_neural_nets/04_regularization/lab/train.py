"""L2.4 实验：CIFAR-10 CNN + 全套正则化/训练技巧，冲击 85%+。

对比 L2.3 (78%)，本实验加入：
1. Dropout - TODO 1 你决定加哪
2. 数据增强 (RandomFlip, RandomCrop)
3. Weight Decay (AdamW)
4. 学习率调度 (CosineAnnealingLR)
5. Early Stopping
6. AMP 混合精度 - TODO 2 你写训练循环
+ wandb 实验追踪
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---- 开关:USE_WANDB=1 用 wandb 可视化;=0 只用终端 ----
USE_WANDB = os.environ.get("USE_WANDB", "1") == "1"
if USE_WANDB:
    import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.data_utils import get_dataset_dir  # noqa: E402
from common.utils import set_seed  # noqa: E402


# --------------------------------------------------------------------------
# CNN：结构和 L2.3 一样，你只需要 TODO 1 加 Dropout
# --------------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self, dropout_p: float = 0.5) -> None:
        super().__init__()
        # 卷积部分：用 BN 已足够正则，不加 Dropout
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # TODO 1: 加 Dropout 到分类头
        # 提示：常规做法是 Linear→ReLU→Dropout→Linear
        # 用参数 dropout_p 而不是硬编码,方便调超参
        # 请把 Sequential 里的三行完善成:
        #   nn.Linear(128 * 4 * 4, 256),
        #   nn.ReLU(),
        #   nn.Dropout(dropout_p),  ← 加这行
        #   nn.Linear(256, 10),
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            # TODO 1: 在这加一行 nn.Dropout(dropout_p)
            nn.Dropout(dropout_p),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.flatten(1)
        return self.classifier(x)


# --------------------------------------------------------------------------
# 数据加载（关键：训练加增强，测试不加）
# --------------------------------------------------------------------------
def get_dataloaders(batch_size: int = 128) -> tuple[DataLoader, DataLoader]:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    # 训练：加数据增强
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),  # 50% 概率水平翻转
            transforms.RandomCrop(32, padding=4),  # 加 4 圈 padding 后随机裁 32×32
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    # 测试：不加增强,只做标准化
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    data_dir = get_dataset_dir("CIFAR10")
    train_set = datasets.CIFAR10(
        str(data_dir), train=True, download=True, transform=train_transform
    )
    test_set = datasets.CIFAR10(str(data_dir), train=False, download=True, transform=test_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


# --------------------------------------------------------------------------
# 训练循环（AMP 版本）
# --------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> float:
    """训练一个 epoch。用 AMP 加速。"""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # TODO 2: 训练标准 4 步（AMP 版本）
        # 对比 L2.3 的普通 5 步:
        #   1. optimizer.zero_grad()
        #   2. y_pred = model(x)
        #   3. loss = loss_fn(y_pred, y)
        #   4. loss.backward()
        #   5. optimizer.step()
        # AMP 版本改成:
        #   1. optimizer.zero_grad()
        #   2. with autocast("cuda"):
        #          logits = model(x)
        #          loss = loss_fn(logits, y)
        #   3. scaler.scale(loss).backward()  ← 替代 loss.backward()
        #   4. scaler.step(optimizer)          ← 替代 optimizer.step()
        #   5. scaler.update()                 ← 新增,更新缩放因子
        optimizer.zero_grad()
        with autocast("cuda"):
            logits = model(x)
            loss = loss_fn(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # raise NotImplementedError("请实现 AMP 训练循环")

        total_loss += loss.item()  # noqa: F821
        n_batches += 1
    return total_loss / n_batches


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast("cuda"):  # 评估也用 AMP,更快
            logits = model(x)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return correct / total


def main() -> None:
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 超参数
    config = {
        "batch_size": 128,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "n_epochs": 30,
        "dropout_p": 0.5,
        "patience": 5,
    }

    # 初始化 wandb
    if USE_WANDB:
        wandb.init(project="cifar10-cnn-regularization", config=config, name="run-1")
    print(f"使用设备: {device}")

    # 数据、模型、损失、优化器、调度器、AMP scaler
    train_loader, test_loader = get_dataloaders(config["batch_size"])
    model = CNN(dropout_p=config["dropout_p"]).to(device)
    loss_fn = nn.CrossEntropyLoss()
    # AdamW = Adam + 正确的 weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    # 余弦退火:lr 从 1e-3 平滑降到 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["n_epochs"])
    scaler = GradScaler("cuda")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")

    # Early stopping 状态
    best_acc = 0.0
    patience_counter = 0

    print(f"\n{'Epoch':<8}{'Train Loss':<15}{'Val Acc':<10}{'LR':<12}")
    print("-" * 45)
    for epoch in range(1, config["n_epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, device)
        val_acc = evaluate(model, test_loader, device)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        marker = ""
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            marker = " ★"
            # 保存最佳权重
            torch.save(model.state_dict(), Path(__file__).parent / "best.pt")
        else:
            patience_counter += 1

        # 记录到 wandb
        if USE_WANDB:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_acc": val_acc,
                    "lr": current_lr,
                    "best_val_acc": best_acc,
                }
            )
        print(f"{epoch:<8}{train_loss:<15.4f}{val_acc:<10.4f}{current_lr:<12.6f}{marker}")

        # Early stopping
        if patience_counter >= config["patience"]:
            print(f"\n[!] Early stopping at epoch {epoch} (patience={config['patience']})")
            break

    print(f"\n[✓] 训练完成,最佳 val acc: {best_acc:.2%}")
    if best_acc > 0.85:
        print("[✓] 达到 85% 目标!")
    else:
        print(f"[!] 未达到 85%,best={best_acc:.2%}。可以微调超参再试")

    wandb.finish() if USE_WANDB else None


if __name__ == "__main__":
    main()
