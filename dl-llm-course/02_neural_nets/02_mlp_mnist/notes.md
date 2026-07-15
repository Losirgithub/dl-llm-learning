# L2.2 MLP 与训练循环 · 笔记

## 一、MLP 结构

**MLP (Multi-Layer Perceptron，多层感知机)** = 若干全连接层叠加，中间加激活函数。是最简单的神经网络。

本课的 MNIST MLP 结构：

```text
输入 (batch, 1, 28, 28)
    ↓ flatten
(batch, 784)
    ↓ Linear(784, 256) → ReLU
(batch, 256)
    ↓ Linear(256, 128) → ReLU
(batch, 128)
    ↓ Linear(128, 10)
输出 logits (batch, 10)
```

**为什么中间加 ReLU**：如果只堆 Linear，多层等价于单层（线性变换的复合还是线性）。ReLU 引入非线性，让"深度"有意义。**这是深度神经网络能拟合复杂函数的根本原因**。

**为什么输出不加 Softmax**：`nn.CrossEntropyLoss` 内置 softmax，直接吃 logits 更数值稳定。分类模型的输出层永远是 Linear，不加激活。

## 二、PyTorch nn.Module

**nn.Module** 是所有神经网络的基类。继承它得到四个自动化能力：

1. 自动追踪所有子层的参数（`model.parameters()`）
2. 自动搬 GPU（`model.to("cuda")`）
3. 自动切换 train/eval 模式
4. 自动保存/加载权重

**两个必写方法**：

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()          # 必须调父类初始化
        # 定义层作为属性
        self.fc1 = nn.Linear(784, 256)
        # ...

    def forward(self, x):
        # 定义数据怎么流过
        x = self.fc1(x)
        # ...
        return x
```

- `__init__`：定义"网络长什么样"（有哪些层）——只在创建模型时运行一次
- `forward`：定义"数据怎么流过"——每次前向都运行

**调用**：`output = model(x)`（直接调 `model()`，PyTorch 内部会调 `forward`）。

## 三、常用层

| 层 | 作用 | 参数量 |
| --- | --- | --- |
| `nn.Linear(in, out)` | 全连接层 $y = Wx + b$ | $W$: (out, in)，$b$: (out,) |
| `nn.ReLU()` | 激活函数 $\max(0, x)$ | 无参数 |
| `nn.Flatten()` | 展平（保留 batch 维） | 无参数 |
| `nn.Sequential(...)` | 层的容器 | 无参数 |

## 四、深度学习训练标准 5 步（记住模板）

```python
optimizer.zero_grad()         # 1. 清零上次梯度
y_pred = model(x)             # 2. 前向：算预测
loss = loss_fn(y_pred, y)     # 3. 前向：算损失
loss.backward()               # 4. 反向：算梯度
optimizer.step()              # 5. 更新参数
```

所有 PyTorch 训练代码都是这五行——MLP、CNN、Transformer、LLM 微调，一模一样。

## 五、Main 函数 6 步模板

所有深度学习训练脚本都是这个骨架：

1. **固定 seed**（可复现地基）
2. **选择设备**（`torch.device("cuda" if ... else "cpu")`）
3. **超参数**（batch_size、lr、n_epochs）
4. **四件套**：DataLoader、Model、Loss、Optimizer
5. **训练循环**：`for epoch: train → evaluate → 打印`
6. **汇报结果**（是否达标）

**四件套的类比**：

- 数据（DataLoader）= 食材供应链
- 模型（nn.Module）= 厨师
- 损失（loss_fn）= 食客评分
- 优化器（Optimizer）= 厨师根据评分改进

**关键要点**：

- `model` 和 `optimizer` **在 for 循环外**创建，只创建一次，参数在整个训练中**连续演化**
- `optimizer` 内部有状态（如 Adam 的 m、v），循环里重建会清空历史
- 模型必须 `.to(device)`，数据也要 `.to(device)`，两者在同一设备

## 六、Epoch / Iteration / Batch

| 术语 | 是什么 | MNIST 示例 |
| --- | --- | --- |
| **Batch** | 一小批数据 | 128 张 |
| **Iteration / Step** | 一次参数更新（处理一个 batch） | 一次 |
| **Epoch** | 遍历训练集一次 | 60000/128 ≈ 469 次 iteration |

**训练 5 个 epoch** = 训练集重复过 5 遍 = 2345 次参数更新。参数**在同一个 model 上连续演化**，不是每 epoch 重新训。

**为什么要多个 epoch**：一遍学不透，每次参数只走一小步（Adam lr=1e-3）。反复遍历训练集，参数逐步逼近最优。

**为什么不训太多**：边际收益递减 + 过拟合风险（训练 loss 降但验证 acc 停滞或下降）。

**大模型场景**：数据集太大，可能连一个 epoch 都跑不完，改按 iteration/step 计数。Llama 训练 2.4 万亿 tokens，数据集只过 1-2 遍。

## 七、train() / eval() 模式

`nn.Module` 有内部开关 `training`，控制两种模式：

- `model.train()`：训练模式（Dropout 生效、BN 用当前 batch 均值）
- `model.eval()`：评估模式（Dropout 关闭、BN 用累积均值）

**对 MLP 现在没影响**（Linear/ReLU 不受模式影响），**但仍要写**：

1. 未来加了 Dropout/BN 不用回来补
2. 工程规范，代码可移植

**两个函数的模板**：

```python
def train_one_epoch(...):
    model.train()          # 训练模式
    for x, y in loader:
        # 5 步训练循环

@torch.no_grad()           # 关闭梯度追踪：省内存、加速
def evaluate(...):
    model.eval()           # 评估模式
    for x, y in loader:
        # 只前向，不 backward、不 step
```

**注意**：`torch.no_grad()` 和 `model.eval()` **两者独立**：

- `no_grad`：管**梯度追踪**（省内存/加速）
- `eval()`：管**层的行为**（Dropout/BN）

评估时**两者都要写**，缺一个都会出问题。

## 八、DataLoader

**DataLoader = 批量数据流水线**。你只写 `for x, y in loader:`，它自动做：

1. **索引**：每张图有编号
2. **打乱**（shuffle=True）：每 epoch 打乱一次
3. **分批**：按 batch_size 切成一段段
4. **打包**：读图、过 transform、叠成 tensor

**三层解耦**：

- 数据集（`datasets.MNIST`）:只管"我有什么"
- DataLoader:只管"怎么打乱、分批、并行读"
- 模型:只管"来一批处理一批"

想改 batch 只调 DataLoader 一行，模型代码一字不改。

## 九、我的误解与纠正 / 补充问答

### 1. `fc` 是什么缩写

**fc = Fully Connected**（全连接）的缩写。是历史惯例——早期把这种层叫 fully connected layer，PyTorch 后来改叫 `nn.Linear`，但变量名约定俗成还是用 `fc1/fc2/fc3`。其他常见约定：

| 缩写 | 全称 | 用途 |
| --- | --- | --- |
| `fc` | fully connected | 全连接层 |
| `conv` | convolution | 卷积层 (CNN) |
| `bn` | batch normalization | 批归一化 |
| `attn` | attention | 注意力层 (Transformer) |
| `ffn` / `mlp` | feed-forward network | 前馈层 |
| `embed` | embedding | 嵌入层 |

### 2. Sequential 和经典写法的区别

三点本质差异：

- **灵活度**：Sequential 只能"一层接一层"直筒走，不能分叉/跳连；经典写法能在 forward 里任意组合（如 ResNet 残差 `return x1 + x2`）
- **可读性**：经典写法 forward 里清楚看数据流；Sequential 藏在 `__init__` 里
- **调试**：经典写法层有名字（`model.fc1`），Sequential 是数字编号（`model.net[1]`）

**工程实践**：

- 顶层结构（含残差/多分支）→ 经典写法
- 重复小模块（"Conv+BN+ReLU"三件套）→ Sequential 打包

### 3. forward 怎么知道 batch_size

**不需要知道**。Linear 层是逐样本独立处理，batch 维只是"路过"：

- 输入 (batch, 784)，$W$ 形状 (256, 784) —— **和 batch 无关**
- 输出 (batch, 256) —— batch 维自动保留

**神经网络所有层都遵守**：batch 维永远是第 0 维，层只处理后面的维度。所以 `x.view(-1, 784)` 用 `-1` 让 PyTorch 自动推断 batch，训练/推理不同 batch 都能用同一份代码。

### 4. 为什么要展平 x，不同展平方法的区别

**为什么展平**：Linear 只吃一维向量，MNIST 4D 张量 (batch, 1, 28, 28) 必须先拉平成 (batch, 784)。

**代价**：丢失空间结构。像素 (0,0) 和 (1,0) 本来上下相邻，展平后差 28 个位置。这是 MLP 图像任务不如 CNN 的根本原因（CNN 不展平，保留 2D 结构）。

**四种展平方法**：

| 方法 | 特点 |
| --- | --- |
| `x.view(-1, 784)` | 快，不复制内存，但要求张量连续 |
| `x.reshape(-1, 784)` | 类似 view，不连续时自动复制 |
| `x.flatten(1)` | 语义清晰："从第 1 维开始展平所有维度"，换数据集自动适配 |
| `nn.Flatten()` 层 | 作为网络层，可放 Sequential，工业界推荐 |

### 5. 网络维度 784/256/128 怎么选

**硬约束**：

- 输入 784 = MNIST 28×28（任务决定，不能改）
- 输出 10 = 10 个类别（任务决定，不能改）
- 相邻层输入输出必须匹配

**中间维度（256/128）是人的选择**：

- 通常逐层"渐缩"（金字塔），信息压缩
- 常用 2 的幂（GPU 高效、约定俗成）
- 太小欠拟合，太大过拟合 + 慢
- 实际做法：抄论文默认值 / 网格搜索 / 贝叶斯优化

**层数**：新任务先试 2-3 层，不够再加。MNIST 上 2-3 层 MLP 就够。

### 6. Tensor 的 `requires_grad=True` 是符号变量吗

**不是**。PyTorch 做**数值计算**，不是符号计算（符号用 SymPy）。

- `x = torch.tensor(3.0, requires_grad=True)` 里 x 是"当前值 3.0 的可训练变量"
- `y = x**2` 立即算出 y=9，同时把关系记入计算图
- `y.backward()` 算的是**当前点的梯度数值**（6.0），不是公式（2x）

**"可训练变量"含义**：一个存着数值的容器，训练循环里值会被优化器更新（3.0 → 2.9 → 2.7 → ...），不是抽象符号。

### 7. 损失、损失函数、前向、反向的区分

| 概念 | 是什么 | 举例 |
| --- | --- | --- |
| **损失函数** | 公式（规则） | MSE、交叉熵 |
| **损失** | 具体数字 | `loss = 0.023` |
| **前向传播** | x 一路算到 loss | `y_pred = model(x); loss = loss_fn(y_pred, y)` |
| **反向传播** | 从 loss 反推参数梯度 | `loss.backward()` |

**判断前向/反向**：看数据流方向。x → y → loss 是前向；loss → grad → 参数 是反向。

**注意**：算 loss 也属于前向——只要还是从输入方向往输出方向算就是前向。反向从 `loss.backward()` 开始。

### 8. `loss_fn` 是什么

`loss_fn` 是**损失函数的实例**（一个可调用对象）。`loss_fn = nn.CrossEntropyLoss()` 创建对象，`loss = loss_fn(y_pred, y)` 调用它算数。

`nn.CrossEntropyLoss` 特性（分类任务标配）：

- 内置 softmax → 直接吃 logits
- 内置 log → 数值稳定
- 吃 int 类别索引（不是 one-hot）
- 返回一个标量 tensor（batch 平均）

### 9. `train_one_epoch` 返回什么

**返回这个 epoch 的平均训练损失**（Python float）。

- 一个 epoch = 469 个 batch（MNIST）
- 每个 batch 算一个 loss，累加到 `total_loss += loss.item()`
- 返回 `total_loss / n_batches`

**为什么用 `.item()`**：把标量 tensor 转 Python float，断开计算图，避免内存爆炸。**训练循环里凡是记录/打印/累计数值，都用 `.item()`**。

### 10. MNIST 数据集在哪分成 469 批

**DataLoader 干的**，在你看不见的地方。`for x, y in loader:` 每次迭代自动抽 128 张打包成 batch 交给你。60000/128 ≈ 469 批后循环自动结束。

### 11. 训练循环里每次都是"继续训"，不是重新训

`model` 和 `optimizer` 在 for 循环**外**创建，只建一次。每个 epoch 拿到**当前的 model**（带上一 epoch 的参数），跑 469 次更新后继续到下一 epoch。**参数连续演化**，5 epoch = 2345 次更新的累积效果。

极少数场景（交叉验证、多次实验取平均、超参搜索）才在 for 循环里重建模型。

### 12. `argmax` 是什么

**argmax = "argument of the maximum"**：返回最大值**所在的位置（索引）**，不是最大值本身。

- `max`：返回最大值 → `[1, 5, 3].max() = 5`
- `argmax`：返回位置 → `[1, 5, 3].argmax() = 1`

**分类任务用法**：logits 形状 (batch, 10)，每行 10 个类别的分数。**位置 = 类别**，argmax 找出分数最高的位置 = 预测的类别。

**dim=1** 表示沿类别维（水平方向）取 argmax，输出 (batch,)——每个样本一个答案。

### 13. `model.train()` 不写有影响吗

**对当前 MLP 没影响**（Linear/ReLU 不受模式影响，训练能正常跑）。

**什么时候有影响**：模型里有 Dropout 或 BatchNorm 时（L2.3 CNN 会遇到）。那时若前面 evaluate 切成了 eval 模式，训练时不切回 train 模式，Dropout/BN 行为错乱。

**规范**：无论当前有没有必要，都写。工程性防错。

### 14. 训练 loss 降但验证 acc 降？—— 过拟合的萌芽

**观察到的现象**（本次训练）：

| Epoch | Train Loss | Val Acc |
| --- | --- | --- |
| 1 | 0.2716 | 0.9639 |
| 2 | 0.1033 | 0.9737 |
| 3 | 0.0686 | 0.9768 |
| 4 | 0.0503 | **0.9776** ← 峰值 |
| 5 | 0.0404 | 0.9764 ← 掉了 |

**这就是过拟合开始的信号**。原理：模型容量足够大时，前期学"通用规律"（数字形状），后期开始"背训练集独有的噪声"——训练 loss 继续降，但学到的东西在新数据上无效或误导。

**类比**：训练集 = 练习册；验证集 = 期末考卷。前几遍学通用解题法（练习/考试都能考好），太多遍开始背练习册答案（练习满分，考试反而更差）。

**深度洞见**：**训练 loss 和验证 acc 必须同时看**，只看训练 loss 会被"进步"错觉误导。深度学习工程师日常盯两条曲线——训练 loss + 验证 acc/loss，它们分岔的地方就是过拟合起点。

**三个应对方法**：

- **Early Stopping**：验证 acc 不再涨就停，保存那时的权重
- **正则化**（L2.4 会讲）：Dropout、Weight decay、数据增强
- **更多数据**：数据越多越难过拟合（大模型靠海量数据吃下巨大参数）

**工具**：wandb / tensorboard 可视化这两条曲线，L2.4 引入。

## 十、实验产物

- 代码：[lab/train.py](lab/train.py)
- 结果：MNIST 验证准确率 **> 97%**（5 个 epoch，GPU 约 1 分钟）
