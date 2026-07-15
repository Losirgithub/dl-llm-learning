# L2.1 PyTorch 张量与自动微分 · 笔记

## 一、PyTorch 是什么

Meta 开源的深度学习框架，做三件事：

1. **张量库**（替代 numpy，能上 GPU）
2. **自动微分**（automatic differentiation，autograd）
3. **神经网络工具**（内置层、损失、优化器）

## 二、Tensor —— PyTorch 版的多维数组

Tensor 和 numpy 数组 API 几乎一样，但有两个超能力：

1. **能在 GPU 上算**（numpy 只能 CPU）
2. **能自动追踪梯度**（numpy 不行）

### 常用 API

```python
import torch

# 创建
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.zeros(3, 4)
c = torch.randn(2, 3)

# 属性
a.shape, a.dtype, a.device

# CPU ↔ GPU
a = a.cuda()      # 或 a.to("cuda")
a = a.cpu()

# Tensor ↔ numpy
n = a.cpu().numpy()          # GPU tensor 必须先 .cpu() 才能转 numpy
t = torch.from_numpy(n)
```

**关键坑**：GPU 上的 tensor **不能直接 `.numpy()`**——numpy 只理解 CPU 内存。必须先 `.cpu()`。

## 三、autograd 核心：数值计算 vs 符号计算

**PyTorch 不做符号计算**（那是 SymPy 的活）。它只做**具体数值**的微分：

- 你给 x 一个具体值（如 3.0）
- PyTorch 记录所有运算到**计算图**
- `backward()` 沿图反向遍历，算出**当前点的梯度数值**（不是公式）

```python
x = torch.tensor(3.0, requires_grad=True)  # x 是"可训练变量"，当前值 3.0
y = x ** 2                                  # 立即算出 y=9，同时记录"y=x²"关系
y.backward()                                # 算 dy/dx 在 x=3 处的数值
print(x.grad)                               # tensor(6.)
```

- `requires_grad=True` 含义：这个数值容器是**可优化的**，训练时会不断更新，不是"抽象符号"
- 输出 `x.grad` 是**数字** 6.0，不是表达式 `2x`

## 四、四个易混概念

| 概念 | 是什么 | 举例 |
| --- | --- | --- |
| **损失函数** | 一个**公式**（规则） | MSE、交叉熵 |
| **损失** | 用公式算出的**具体数字** | `loss=0.023` |
| **前向传播** | 从输入 x 一路算到 loss | `y_pred = model(x); loss = loss_fn(y_pred, y)` |
| **反向传播** | 从 loss 反推每个参数的梯度 | `loss.backward()` |

**判断前向/反向的一句话规则**：看数据流方向。x → y → loss 是前向；loss → grad → 参数 是反向。

## 五、autograd 五条必知规则

1. **只对 `requires_grad=True` 的张量求梯度**（模型参数需要，输入数据不需要）
2. **梯度默认累加**，必须手动 `optimizer.zero_grad()`（新手最大的坑）
3. `backward()` **默认只能对标量调用**（loss 必须是单个数字，向量要先 `.sum()`）
4. **中间变量的梯度默认不保留**（省内存，需要用 `.retain_grad()`）
5. 推理时用 `with torch.no_grad():` **关闭追踪**（省内存、加速）

## 六、深度学习训练标准 4 步（记住这个模板）

```python
for batch in data:
    optimizer.zero_grad()       # 1. 清零上次的梯度
    y_pred = model(x)           # 2. 前向：算预测
    loss = loss_fn(y_pred, y)   # 3. 前向：算损失
    loss.backward()             # 4. 反向：算梯度
    optimizer.step()            # 5. 更新参数
```

**所有 PyTorch 训练代码都是这五行**，无一例外。从 MLP 到 Transformer 到 LLM 微调都一样。

**注意**：`backward()` 只算梯度，**不更新参数**；`step()` 才更新。这两步严格分开，是深度学习框架的设计精髓——中间可以插入梯度裁剪、梯度累积等操作。

## 七、L1.3 手写 vs L2.1 PyTorch 对比

同一个"细长山谷"损失，同一起点，同一 lr，Adam 优化 50 步：

| L1.3 手写版（约 30 行） | L2.1 PyTorch 版（约 5 行） |
| --- | --- |
| 手推梯度 `grad(x,y)=[10x,y]` | ❌ 不需要 |
| 手写 m、v 更新 | ❌ 不需要 |
| 手写偏差修正 | ❌ 不需要 |
| 手写参数更新公式 | ❌ 不需要 |
| 只写损失函数即可 | ✅ |

**实验结果**：两版最终位置和损失完全一致 `(+0.0386, -0.1424) loss=0.017578` —— 证明 PyTorch autograd 是精确的链式法则实现，不是近似。

## 八、核心 insight

**深度学习 2012 年后爆发的技术底座不是理论突破，而是 autograd**——它把"手推梯度"这堵墙拆掉了。

- 传统数值优化：写函数 → 手推梯度 → 写优化器 → 反复调试梯度公式
- 深度学习：写前向 → PyTorch 自动搞定后面所有 → 专注设计模型

真实神经网络有百万到十亿参数，没有 autograd 根本无法工作。这也是为什么所有现代模型（CNN/Transformer/LLM）都基于 PyTorch 或 TensorFlow。

## 九、我的误解与纠正

| 我的误解 | 正确 |
| --- | --- |
| PyTorch 里 `x = torch.tensor(3.0, requires_grad=True)` 是"符号变量" | PyTorch 做**数值计算**不是符号计算；`x` 是一个存着数值的容器，`requires_grad=True` 只表示"可优化"，不是"抽象符号"。符号计算用 SymPy |
| 算损失 `loss = loss_fn(y_pred, y)` 属于反向传播？ | 属于**前向传播**。前向的范围是"从输入 x 到 loss 的整个过程"，只要还是从输入方向往输出方向算就是前向。反向从 `loss.backward()` 才开始 |

## 十、实验产物

- 代码：[lab/train.py](lab/train.py)
- 验证：PyTorch autograd 版与 L1.3 手写 Adam 结果完全一致（数值到小数点后 4 位）
