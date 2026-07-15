# L0.1 环境搭建 · 笔记

## 一、学习目标

在两台机器（3070 笔记本 + 5070Ti 台式机）都装好 GPU 版 PyTorch，理解"可复现"的第一块基石：**版本锁定**。

## 二、核心成果

- 两机 `torch.cuda.is_available()` 都是 `True`
- 3070（Ampere, sm_86）+ 5070Ti（Blackwell, sm_120）都能跑 GPU 矩阵运算
- 建立环境自检脚本 [check_env.py](../01_env/check_env.py)，可复用到任何新机器

## 三、关键概念

### 1. GPU 版 vs CPU 版 PyTorch（头号坑）

**看版本号后缀**判断：

- `2.9.0+cu130` → **GPU 版**（CUDA 13.0 构建）
- `2.9.0+cpu` → **CPU 版**（不能用 GPU）

**默认 `pip install torch` 装的是 CPU 版**！必须从 PyTorch 官方 CUDA 源装：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

一句话验证：

```bash
python -c "import torch; print(torch.__version__)"
```

看后缀是 `+cu1xx` 还是 `+cpu` 立判。

### 2. 算力（Compute Capability / sm 版本）

每块 NVIDIA GPU 有一个算力号（如 `sm_86`, `sm_120`），代表它的架构。PyTorch 编译时只支持特定 sm 版本。

| GPU | 架构 | 算力 |
| --- | --- | --- |
| 3070 Laptop | Ampere | sm_86 |
| 5070Ti | Blackwell | sm_120 |

**新架构（如 sm_120）需要新版 PyTorch**。旧版本没编译支持 sm_120 → `torch.cuda.is_available()` 可能是 True，但一做运算报 `no kernel image is available`。

**你的实测**：PyTorch 2.9.0 + CUDA 13.0 已足够新，5070Ti 直接支持，没踩这个坑。

### 3. 为什么锁版本

深度学习实验必须"可复现"，环境是地基：

- 同一份代码，3070 和 5070Ti 可能需要**不同版本**的 PyTorch（新卡需要新框架）
- 论文复现 = 用同版本的 PyTorch/CUDA/GPU
- 三个月后回来 = 环境不锁死无法重跑

**工程规范**：记录 Python 版本、CUDA 版本、PyTorch 版本、GPU 型号、commit hash——像实验数据一样管理。

## 四、我的误解与纠正 / 补充问答

### 1. Anaconda vs Miniconda 选哪个

**Miniconda 更好**（工业界主流）。

| | Anaconda | Miniconda |
| --- | --- | --- |
| 大小 | 几 GB，全家桶 | 几百 MB，极简 |
| 预装 | 数百个科学计算包 | 只有 conda + Python |
| 缺点 | 依赖冗余，冲突多 | 要什么自己装 |

深度学习只需要 torch/transformers 十来个包，Anaconda 的一堆预装包只会增加冲突风险。

### 2. Windows 11 为什么 `platform.platform()` 显示"Windows-10"

Windows 11 内核版本号仍是 `10.0`，微软只是把 build 号提到 22000+。老 Python 只读内核版本，标为 "Windows-10"。**判断 Win10/11 看 build 号**：≥ 22000 就是 Win11。

### 3. 环境激活错了的排查

**症状**：脚本报"没检测到 PyTorch"，但你确定装过。

**流程**：

```bash
conda info --envs                  # 看当前激活的环境
python -c "import sys; print(sys.executable)"  # 看当前 python 的路径
pip show torch                     # 看 torch 装在哪
```

三步定位：torch 装的环境和当前激活的环境**不是同一个**。跑脚本前**永远先确认环境**。

### 4. PowerShell 禁止运行脚本

**报错**：`无法加载 .ps1，因为在此系统上禁止运行脚本`

**解法**（不需管理员）：

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

`RemoteSigned` 是安全推荐级别：本地脚本能跑，网上下的需签名。

### 5. SSH 从笔记本远程操控台式机

Windows 11 自带 OpenSSH。台式机开服务：

```powershell
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Start-Service sshd
Set-Service -Name sshd -StartupType 'Automatic'
New-NetFirewallRule -Name sshd -DisplayName 'OpenSSH SSH Server' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22
```

笔记本连：`ssh .\用户名@IP`（`.\` 强制本地账户，避开微软账户）。

**空密码不能远程登录**：给 5070Ti 设个临时密码 `net user 用户名 密码`。

### 6. 国内镜像源使用

- **conda 源**：清华最全 `mirrors.tuna.tsinghua.edu.cn/anaconda`
- **pip 源**：清华 `pypi.tuna.tsinghua.edu.cn/simple`（或阿里云）
- **PyTorch 源**：**必须用官方 CUDA 源**或上交镜像，**不能用清华源**（清华的 torch 是 CPU 版！）

规则：**普通包走清华，PyTorch 走专用 CUDA 源**。

### 7. Windows 静默安装踩的坑

- **conda-hook.ps1 找不到**：静默安装可能失败却没验证。任何静默/远程操作后**必须 `Test-Path` 验证产物**
- **pip.ini 报 BOM 乱码**：PowerShell `Out-File -Encoding utf8` 带 BOM，pip 不认。用 `[System.IO.File]::WriteAllText(路径, 内容, [System.Text.UTF8Encoding]::new($false))` 写无 BOM

## 五、check_env.py 的价值

一个可复用工具，输出：

- 操作系统 + Python 版本（记录环境）
- PyTorch 版本 + CUDA 编译版本（判断 GPU 版）
- GPU 名称 + 算力（判断架构支持）
- **实际做一次 GPU 矩阵运算**（不是"假可用"）

这个脚本以后每台新机器、每个新环境都可以直接跑，秒诊断。
