"""跨课程复用的工具函数。

工程规范第一条：把随机性钉死，保证实验可复现。
所有课程都从这里导入 set_seed，而不是各自重写。
"""

import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """固定所有随机源，保证实验可复现。

    参数：
        seed: 随机种子，整个实验用同一个值
        deterministic: 是否开启 CUDA 确定性模式（牺牲少量速度换可复现）
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多卡也要固定
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
