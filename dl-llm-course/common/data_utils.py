"""数据集路径统一管理。所有课程从这里获取数据集路径。

所有数据集集中放在 dl-llm-course/data/ 目录，多个实验共享一份，
不用每个 lab 复制一次占用存储空间。
"""

from pathlib import Path

# 课程根目录（本文件的 parent.parent）
COURSE_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = COURSE_ROOT / "data"


def get_dataset_dir(name: str) -> Path:
    """获取指定数据集的存放目录，不存在则自动创建。

    参数:
        name: 数据集名称(和 torchvision 惯例一致,如 "MNIST", "CIFAR10")

    返回:
        数据集目录的绝对路径，形如 e:\\...\\dl-llm-course\\data\\CIFAR10\\
    """
    path = DATA_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path
