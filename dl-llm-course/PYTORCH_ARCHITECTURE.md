# PyTorch 代码架构 · 通用参考

> 这是从 L3.2 提炼的通用知识，不绑定任何一课。写/读 PyTorch 代码时随时参考。

## 一、三层架构

```text
┌─────────────────────────────────────────┐
│  训练脚本层 (train.py 的 main)            │  ← 编排:怎么用模型训练
│  数据加载、训练循环、评估、打印            │
├─────────────────────────────────────────┤
│  模型层 (nn.Module 类)                   │  ← 定义:模型长什么样
│  Transformer -> Encoder/Decoder -> Block  │
│  -> Attention/FFN/PE                     │
├─────────────────────────────────────────┤
│  工具函数层 (普通函数)                    │  ← 纯计算:无状态
│  scaled_dot_product_attention, set_seed  │
└─────────────────────────────────────────┘
```

## 二、核心判断:类 vs 函数

**有参数/有状态 → 类(继承 nn.Module);纯数学计算 → 函数。**

### 用 nn.Module 类的情况

内部有 **weight、bias、buffer** 之类需要:
- 反向传播更新(参数)
- `.to(device)` 搬 GPU
- `.train()/.eval()` 切模式
- `state_dict()` 存盘

```python
class MultiHeadAttention(nn.Module):
    def __init__(self):
        self.W_Q = nn.Linear(...)   # ← 有可训练参数
    def forward(self, x):
        ...
```

### 用普通函数的情况

纯数学公式，无参数，输入确定输出就确定。

```python
def scaled_dot_product_attention(Q, K, V):
    scores = Q @ K.transpose(-2,-1) / sqrt(d_k)
    return F.softmax(scores, dim=-1) @ V
```

**判断信号**：`__init__` 里啥都没有 → 用函数更简洁。

### 数学与参数分离(标准范式)

```python
# 数学抽成函数，复用
def scaled_dot_product_attention(Q, K, V): ...

# 类只管参数 + 调函数
class MultiHeadAttention(nn.Module):
    def forward(self, x):
        Q, K, V = self.W_Q(x), self.W_K(x), self.W_V(x)
        return scaled_dot_product_attention(Q, K, V)
```

**分工**：函数管"怎么算"，类管"算什么(参数)"。Decoder 的 cross-attention 也能复用同一个函数，只是传不同的 Q/K/V。

## 三、__init__ 和 forward 的分工

```python
class EncoderBlock(nn.Module):
    def __init__(self):
        # __init__: 声明"我有哪些零件"(买零件)
        self.attention = MultiHeadAttention(...)
        self.ffn = FeedForward(...)
        self.norm1 = nn.LayerNorm(...)

    def forward(self, x):
        # forward: 定义"零件怎么连起来"(组装/接线)
        attn_out = self.attention(x)
        x = self.norm1(x + attn_out)
        return x
```

- **__init__ = 声明有什么**(零件清单)
- **forward = 定义怎么连**(电路图)

没有 forward，nn.Module 不知道拿到数据后该干嘛。

## 四、为什么用 `model(x)` 不用 `model.forward(x)`

调用时写 `block(x)`，不是 `block.forward(x)`：

```python
out = block(x)              # ✅ 推荐
out = block.forward(x)      # ❌ 不推荐
```

**nn.Module 的 `__call__` 帮你做了 3 件事，然后才调 forward**:

1. 触发前向钩子(forward hooks，调试用)
2. 设置 autograd(自动微分追踪)
3. 调用 `self.forward(x)`

直接调 `forward()` 会跳过这些，可能出问题(梯度追踪丢失等)。**只实现 forward，用 `()` 调用**。

## 五、积木拼装的层次结构

每一层都是 nn.Module，通过 `__init__` 买子模块、`forward` 接线拼装。

```text
最底层: nn.Linear, nn.LayerNorm                ← PyTorch 内置(叶节点)
        ↓ 拼装
中间层: MultiHeadAttention, FeedForward, PE    ← 自定义积木
        ↓ 拼装
上层:   EncoderBlock, DecoderBlock             ← 块
        ↓ 拼装
顶层:   Transformer                             ← 完整模型
        ↓
训练:   main() 用 Transformer 训练              ← train.py
```

### 两种拼装方式

**方式 1:手动在 forward 里接线**(灵活，能加残差/分支)

```python
class EncoderBlock(nn.Module):
    def forward(self, x):
        x = self.norm1(x + self.attention(x))   # 手动接线
        x = self.norm2(x + self.ffn(x))
        return x
```

**方式 2:nn.Sequential 自动串联**(简洁，数据直筒过)

```python
encoder = nn.Sequential(*[EncoderBlock(...) for _ in range(6)])
# 等价于: x = block1(x); x = block2(x); ...
```

**选择**:
- 简单串联(一层接一层)→ Sequential
- 有残差/分支/复杂连接 → 手动 forward

## 六、模型内部数据的三种注册方式

| 方式 | 用途 | 是否训练 | 例子 |
| --- | --- | --- | --- |
| `self.x = nn.Linear(...)` | 子模块(含参数) | 训练 | 层 |
| `self.register_parameter('w', nn.Parameter(...))` | 单独参数 | 训练 | 自定义权重 |
| `self.register_buffer('pe', pe)` | 非参数数据 | **不训练** | 位置编码、BN 的 running_mean |

**普通属性 `self.x = pe` 能用，但 `.to(device)` 不搬它、存盘不存它**。所以模型内部数据必须用上面三种之一注册，不能用普通属性。

### register_buffer 的特点

`self.register_buffer('pe', pe)` 做两件事:
1. 注册 pe 到模型(跟着 `.to(device)` 搬 GPU、跟着 `state_dict` 存盘)
2. 同时设为属性 `self.pe`(内部已赋值，**不用再写 `self.pe = pe`**)

forward 里直接 `self.pe` 访问。

## 七、通用模板:训练脚本 6 步

所有 PyTorch 训练代码都是这个骨架(从 MLP 到 Transformer 到 LLM 微调):

1. **固定 seed**(可复现)
2. **选设备**(cuda/cpu)
3. **超参数**(batch_size、lr、epochs)
4. **四件套**:DataLoader、Model、Loss、Optimizer
5. **训练循环**:`for epoch: train -> evaluate -> 打印`
6. **汇报结果**

训练循环标准 5 步(所有模型一样):

```python
optimizer.zero_grad()      # 清梯度
y_pred = model(x)          # 前向
loss = loss_fn(y_pred, y)  # 算损失
loss.backward()            # 反向
optimizer.step()           # 更新
```

## 八、关键工程习惯

### 1. 逐模块验证形状

每个积木单独 `assert output.shape == expected`，再拼起来验整体。比训练时 loss 变 NaN 再回来调好得多。

### 2. 调试时打印 shape

```python
def forward(self, x):
    print(x.shape)
    x = self.conv_layers(x)
    print(x.shape)
    ...
```

CNN/Transformer 调试最有效的手段。

### 3. 参数量检查

```python
n_params = sum(p.numel() for p in model.parameters())
print(f"参数量: {n_params:,}")
```

凭参数量能判断模型规模是否合理(对比同类模型)。

### 4. 训练时用 `.item()` 累加 loss

```python
total_loss += loss.item()   # ✅ 断开计算图
# total_loss += loss        # ❌ 累加 tensor，内存爆炸
```

## 九、张量形状操作：squeeze / unsqueeze / view / reshape

四个最常用的形状变换方法，**都不改数据本身**，只改"怎么看这块内存"。

### 速查表

| 方法 | 作用 | 改变元素顺序 | 要求内存连续 |
| --- | --- | --- | --- |
| `unsqueeze(dim)` | 插入 size=1 的新维度(升维) | 否 | 否 |
| `squeeze(dim)` | 删除 size=1 的维度(降维) | 否 | 否 |
| `view(...)` | 重塑成指定形状 | 否 | **是** |
| `reshape(...)` | 重塑成指定形状 | 否 | 否(不连续时自动拷贝) |

### 1. unsqueeze(dim)：插入一个 size=1 的维度

```python
x = torch.tensor([1, 2, 3])   # (3,)
x.unsqueeze(0)                 # (1, 3)  最前面加 batch 维
x.unsqueeze(1)                 # (3, 1)  变列向量
x.unsqueeze(-1)                # (3, 1)  最后加，同上
```

**常见用途**：

- 加 batch 维：`(d_model,) -> (1, d_model)` 单样本推理
- 构造广播形状：`(N,) -> (N, 1)` 变列向量，好和行向量广播

位置编码里的典型用法：

```python
position = torch.arange(0, max_len)   # (max_len,)
position = position.unsqueeze(1)      # (max_len, 1) 列向量
# 后面 position * div_term: (max_len,1) × (d_model//2,) -> 广播成 (max_len, d_model//2)
```

### 2. squeeze(dim)：删除 size=1 的维度(unsqueeze 的逆操作)

```python
x = torch.zeros(1, 3, 1)       # (1, 3, 1)
x.squeeze(0)                   # (3, 1)   删第0维
x.squeeze(-1)                  # (1, 3)   删最后一维
x.squeeze()                    # (3,)     不带参数：删掉所有 size=1 的维度
```

⚠️ `squeeze()` 不带参数会删掉**所有** size=1 的维度，有时会误删想要的维度。建议显式指定 `dim`。

### 3. view(...)：重塑形状(要求内存连续)

```python
x = torch.arange(12)           # (12,)
x.view(3, 4)                   # (3, 4)
x.view(2, -1)                  # (2, 6)   -1 自动推断
x.view(2, 2, 3)                # (2, 2, 3)
```

- **元素总数必须不变**：`view(3,4)` 要求正好 12 个元素
- `-1` 表示自动推断该维大小(整张张量只能有一个 -1)
- **要求内存连续**：transpose 后内存不连续，直接 view 会报错，需先 `.contiguous()`

```python
x = torch.randn(2, 3)
y = x.transpose(0, 1)          # (3, 2) 内存不连续
y.view(6)                      # ❌ RuntimeError
y.contiguous().view(6)         # ✅
```

### 4. reshape(...)：重塑形状(自动处理连续性)

```python
x = torch.arange(12)           # (12,)
x.reshape(3, 4)                # (3, 4)
x.reshape(2, -1)               # (2, 6)
```

- 用法和 `view` 几乎一样
- **区别**：内存连续时返回 view(共享内存)，不连续时自动拷贝，**不会报错**
- 等价于 `contiguous().view()` 的安全版本

transpose 后的对比：

```python
y = x.transpose(0, 1)          # (3, 2) 不连续
y.reshape(6)                   # ✅ 自动拷贝，不报错
y.view(6)                      # ❌ 报错
```

### 怎么选

| 需求 | 选哪个 |
| --- | --- |
| 加/删一个 size=1 的维度 | `unsqueeze`/`squeeze`(语义最明确) |
| 简单重塑，确定内存连续 | `view`(最快，共享内存) |
| 简单重塑，不确定连续性 | `reshape`(最安全，不报错) |
| transpose 后要重塑 | `reshape` 或 `contiguous().view()` |

**经验法则**：拿不准就用 `reshape`，不会出错；要明确表达"插一维/删一维"用 `unsqueeze`/`squeeze`。

### 共同点：都不改数据(共享内存)

四个方法都**不复制数据**(reshape 遇到不连续时除外)，只改"怎么看"这块内存。所以改新张量，原张量可能跟着变：

```python
x = torch.arange(6)
y = x.view(2, 3)
y[0, 0] = 99
print(x[0])                    # 99  (x 也变了，因为共享内存)
```

> 记忆钩子：`view`/`reshape` 是"换眼镜看同一块数据"，`unsqueeze`/`squeeze` 是"给数据套/脱一层 size=1 的壳"。要真正复制数据用 `.clone()`。
