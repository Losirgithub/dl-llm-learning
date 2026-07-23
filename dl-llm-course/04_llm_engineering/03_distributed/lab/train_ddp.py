"""L4.3 实验:DDP 分布式训练(单机跑通)。

用 DDP 代码结构训练一个 MLP,单机单进程验证代码能跑。
跑通后可扩展到双机。

运行:torchrun --nproc_per_node=1 train_ddp.py
(注意是 torchrun,不是 python)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Windows 上 PyTorch 的 TCPStore 默认要用 libuv,但 Windows 版没编译进来
# 关掉它,否则 torchrun 报 "use_libuv was requested but PyTorch was built without libuv"
# 必须在 import torch.distributed 之前设
os.environ["USE_LIBUV"] = "0"

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from common.data_utils import get_dataset_dir  # noqa: E402
from common.utils import set_seed  # noqa: E402


# ===========================================================================
# 模型:简单 MLP(和 L2.2 一样)
# ===========================================================================
class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_mnist_dataset():
    """加载 MNIST 训练集。"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    data_dir = get_dataset_dir("MNIST")
    return datasets.MNIST(str(data_dir), train=True, download=True, transform=transform)


def main() -> None:
    # ===========================================================================
    # TODO 1: 初始化分布式进程组 + 获取 local_rank
    # ===========================================================================
    # torchrun 会设置环境变量:RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT
    # 提示:
    #   1. dist.init_process_group(backend="gloo")  # Windows 用 gloo,Linux 用 nccl
    #   2. local_rank = int(os.environ["LOCAL_RANK"])  # 本机第几张卡
    #   3. torch.cuda.set_device(local_rank)  # 绑定到这张卡
    #   4. print(f"进程初始化成功: rank={dist.get_rank()}, world_size={dist.get_world_size()}")
    dist.init_process_group(backend="gloo")  
    local_rank = int(os.environ["LOCAL_RANK"])  
    torch.cuda.set_device(local_rank)  
    print(f"进程初始化成功: rank={dist.get_rank()}, world_size={dist.get_world_size()}")
    #raise NotImplementedError("TODO 1: 初始化分布式")

    device = torch.device("cuda", local_rank)  # noqa: F821
    set_seed(42)

    # ===========================================================================
    # TODO 2: 用 DDP 包装模型
    # ===========================================================================
    # 提示:
    model = MLP().to(device)
    model = DDP(model, device_ids=[local_rank])  # 包成 DDP
    # DDP 包装后,backward() 会自动同步梯度
    # raise NotImplementedError("TODO 2: DDP 包装模型")

    # 数据:DistributedSampler 自动按 world_size 切分数据
    # 单机单进程时 world_size=1,sampler 拿全部数据(等价普通训练)
    train_set = get_mnist_dataset()
    sampler = DistributedSampler(train_set)
    loader = DataLoader(train_set, batch_size=128, sampler=sampler)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # noqa: F821

    # ===========================================================================
    # 训练循环(和 L2.2 一样,只多了 sampler.set_epoch)
    # ===========================================================================
    n_epochs = 2
    for epoch in range(n_epochs):
        sampler.set_epoch(epoch)  # 每个 epoch 重新打乱,保证各进程切分不同
        model.train()
        total_loss = 0.0
        n = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()  # DDP 在这里自动同步梯度!
            optimizer.step()
            total_loss += loss.item()
            n += 1

        # 只有 rank 0(主进程)打印,避免重复
        if dist.get_rank() == 0:
            print(f"Epoch {epoch+1}: loss={total_loss/n:.4f}")

    # 验证梯度同步:打印第一层权重的梯度范数
    # 单进程时只有一个值;双机时两台应该一样(证明梯度同步了)
    grad_norm = model.module.net[1].weight.grad.norm().item()  # noqa: F821
    print(f"[rank {dist.get_rank()}] 第一层梯度范数: {grad_norm:.4f}")

    dist.destroy_process_group()
    print("训练完成,进程组已销毁")


if __name__ == "__main__":
    main()
