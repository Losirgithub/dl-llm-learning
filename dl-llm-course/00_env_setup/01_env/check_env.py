"""环境自检脚本 —— L0.1

在每台机器上运行一次，确认 GPU 版 PyTorch 可用、算力被支持。
用法：conda activate <你的环境>  然后  python check_env.py
"""

import platform
import sys


def main() -> None:
    # ---- 基础环境信息（可复现的第一块基石：记录下来）----
    print("=" * 50)
    print("环境自检报告")
    print("=" * 50)
    print(f"操作系统      : {platform.platform()}")
    print(f"Python 版本   : {sys.version.split()[0]}")

    # ---- 检查 PyTorch 是否安装 ----
    try:
        import torch
    except ImportError:
        print("\n[✗] 没检测到 PyTorch。请先安装（见本课 README）。")
        return

    print(f"PyTorch 版本  : {torch.__version__}")

    # ---- 关键：CUDA（GPU）是否可用 ----
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA 可用     : {cuda_ok}")

    if not cuda_ok:
        # 最常见的坑：装成了 CPU 版 torch
        print("\n[✗] CUDA 不可用。最可能的原因：装的是 CPU 版 PyTorch。")
        print("    解决：卸载后按官网 selector 装 GPU 版（见本课 README）。")
        print(f"    参考——torch 编译时的 CUDA 版本: {torch.version.cuda}")
        return

    # ---- GPU 详情 ----
    print(f"编译 CUDA 版本: {torch.version.cuda}")
    print(f"GPU 数量      : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        cap = torch.cuda.get_device_capability(i)  # (major, minor) 算力
        sm = f"sm_{cap[0]}{cap[1]}"
        print(f"  GPU {i}: {name}  算力 {cap}  ({sm})")

    # ---- 实际做一次 GPU 运算，确认不是“假可用”----
    try:
        x = torch.randn(2000, 2000, device="cuda")
        y = (x @ x).sum().item()  # 矩阵乘 + 求和，逼它真正调用 GPU
        print(f"\n[✓] GPU 矩阵运算成功，结果标量 = {y:.2f}")
        print("[✓] 环境自检通过！这台机器可以开始训练了。")
    except Exception as e:  # noqa: BLE001  教学脚本，捕获所有异常给出提示
        print(f"\n[✗] GPU 运算失败：{e}")
        print("    若报 sm_120/kernel 不支持，说明 PyTorch 版本太旧不认新卡。")
        print("    （见 README 的 5070Ti 说明）")


if __name__ == "__main__":
    main()
