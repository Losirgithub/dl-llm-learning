# L0.2 项目脚手架与工程规范 · 笔记

## 一、学习目标

搭好课程项目的标准脚手架，装好代码风格工具（ruff），理解深度学习工程规范的核心：**可复现**。

## 二、深度学习为什么特别讲究"可复现"

传统软件：代码一样，结果一样。

**深度学习不是**：

- 权重随机初始化
- 数据随机打乱
- GPU 运算有非确定性

同一份代码跑两次，准确率能差 1-2%。这意味着如果工程不规范，**你永远分不清"是我的改进有效"还是"只是随机波动"**。

这就是顶级实验室死磕规范的根本原因。

## 三、set_seed 是可复现的核心机制

**关键理解**：`set_seed` 不是给某个具体操作用的，而是**改变全局随机源的初始状态**，后续所有随机操作自动受影响。

### 类比

把"随机数生成器"想象成摇号机：

- 默认摇号机：开机在随机位置，每次开机不一样
- `set_seed(42)`：强行把摇号机从"42 号位置"开始，之后摇出的号码永远固定
- torch、numpy、random 里所有随机操作都从这台摇号机取数

### set_seed 实际做的事

```python
def set_seed(seed=42, deterministic=True):
    random.seed(seed)                    # Python 内置 random
    np.random.seed(seed)                 # NumPy 随机
    torch.manual_seed(seed)              # PyTorch CPU 随机
    torch.cuda.manual_seed_all(seed)     # PyTorch GPU 随机
    if deterministic:
        torch.backends.cudnn.deterministic = True   # CUDA 底层确定性
        torch.backends.cudnn.benchmark = False      # 关掉自动选最快算法
```

**为什么要分这么多**：PyTorch/NumPy/Python 各自有独立的摇号机，得**分别固定**。CUDA 底层还有非确定性算法（如卷积），要额外关掉。

### L2.2 训练里哪些地方"偷偷摇号"

看似只调用一次 `set_seed(42)`，但至少 4 处随机源被固定：

1. **模型权重初始化**（`MLP()` 里 `nn.Linear` 用 Kaiming 初始化随机填充 W）
2. **DataLoader shuffle=True**（每 epoch 随机打乱 60000 张图的顺序）
3. **Dropout**（本课没用，L2.4 会加：训练时随机丢神经元）
4. **数据增强**（本课没用，L2.4 会加：随机翻转/裁剪图像）

**验证方法**：删掉 `set_seed(42)` 连跑 3 次 → 结果都不一样；保留 → 结果 bit-level 一致。

## 四、可复现的价值

**没固定 seed 的痛苦**：

- 你调超参数从 97.6% 涨到 97.8%，是**改进**还是**运气**？无法判断
- 论文复现对不上，是**你造假**还是**seed 不同**？争论无解
- 三个月后重跑，结果不对，不知道哪里错

**固定 seed 后**：

- 相同 seed + 相同代码 → 结果 bit-level 一致
- 改一个超参数直接对比，改进真伪一目了然

## 五、ruff：现代代码规范工具

**ruff** = Rust 写的 linter + formatter，比传统工具（flake8 + isort + black）快几十倍。一个工具替代一堆。

**两个功能**：

- `ruff check .` → 找潜在问题（未用变量、import 顺序乱、行太长）
- `ruff format .` → 自动排版（缩进、空格、引号统一）

**为什么工业标配**：

1. 代码风格统一 = 别人和三个月后的你能读懂
2. 提前抓 bug（拼错变量、忘 import）
3. 团队 review 时不争格式，专注模型逻辑

## 六、pyproject.toml 配置

```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]   # 错误/风格/import/升级/bug
```

配置类别：

- `E`：错误（如缩进）
- `F`：pyflakes（如未用 import）
- `I`：import 排序
- `UP`：语法现代化（旧写法自动升级）
- `B`：bug 隐患

## 七、课程根目录结构

```text
dl-llm-course/
├── PROGRESS.md              # 进度与错题本（每课更新）
├── README.md                # 课程总览
├── pyproject.toml           # 统一 lint/format 配置
├── common/
│   └── utils.py             # 跨课复用工具（如 set_seed）
├── 00_env_setup/
├── 01_foundations/
├── 02_neural_nets/
├── ...
└── 07_frontier/
```

## 八、我的误解与纠正 / 补充问答

### 1. Windows 环境写 .ini 文件出现乱码

**问题**：`Out-File -Encoding utf8` 写入带 BOM，pip 解析报 `File contains no section headers`，行首出现 `锘縖global]` 乱码。

**解法**：用 .NET 方法写无 BOM UTF-8：

```powershell
[System.IO.File]::WriteAllText($path, $content, [System.Text.UTF8Encoding]::new($false))
```

### 2. OpenMP 冲突（Windows 上 numpy + matplotlib）

**报错**：`OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.`

**原因**：conda 环境里 numpy 和 matplotlib 各自带一份 Intel OpenMP DLL，加载冲突。

**解法**：在**导入 matplotlib 之前**设环境变量：

```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib
```

顺序很重要——`matplotlib.use("Agg")` 太晚，DLL 已加载完。

### 3. matplotlib 显示中文

默认字体 DejaVu Sans 不含中文。加两行用 Windows 自带黑体：

```python
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False  # 修正负号
```

### 4. `set_seed` 后运行不同 seed 会得到不同结果吗

是的。**seed 值本身没意义**（42 只是 meme），关键是**用了 seed 后每次相同**。

- `set_seed(42)` 跑 3 次 → 每次结果都一样
- `set_seed(7)` 跑 3 次 → 每次结果都一样（但和 42 时不同）
- 不调用 set_seed → 每次不一样

## 九、实验产物

- 脚手架代码：[common/utils.py](../../common/utils.py)
- pyproject.toml：课程根目录
- 验证脚本：[verify_scaffold.py](../02_scaffold/verify_scaffold.py) 打印相同 seed 的确定性验证
