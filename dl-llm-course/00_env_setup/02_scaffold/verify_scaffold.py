"""L0.2 脚手架验证：确认 set_seed 可复现、工程规范生效。

运行：python 00_env_setup/02_scaffold/verify_scaffold.py
"""

import sys
from pathlib import Path

# 把课程根目录加入路径，以便 import common
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from common.utils import set_seed


def main() -> None:
    # 核心验证：相同 seed 应产生相同随机数（可复现的本质）
    set_seed(42)
    t1 = torch.rand(3).tolist()
    n1 = np.random.rand(3).tolist()

    set_seed(42)
    t2 = torch.rand(3).tolist()
    n2 = np.random.rand(3).tolist()

    if t1 == t2 and n1 == n2:
        print("[✓] set_seed 可复现：相同 seed 产生相同随机数")
    else:
        print("[✗] set_seed 失败：相同 seed 产生不同随机数")
        print(f"  torch: {t1} vs {t2}")
        print(f"  numpy: {n1} vs {n2}")
        return

    print(f"  torch 随机数: {t1}")
    print(f"  numpy 随机数: {n1}")
    print(f"  CUDA 确定性模式: {torch.backends.cudnn.deterministic}")
    print("[✓] 脚手架验证通过")


if __name__ == "__main__":
    main()
