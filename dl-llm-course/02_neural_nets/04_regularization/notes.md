# L2.4 正则化与训练调优 · 笔记

## 一、核心问题

L2.3 CNN 在 CIFAR-10 达到 78%，但**过拟合明显**——训练 loss 降到 0.23，验证 acc 停在 78% 且开始下降。L2.4 用六大技术专治过拟合，同时引入实验追踪工具。

## 二、六大正则化/训练技巧

### 1. Dropout（随机丢弃）

**机制**：训练时随机把 x% 的神经元输出置零，评估时全神经元工作。

```python
self.classifier = nn.Sequential(
    nn.Linear(2048, 256),
    nn.ReLU(),
    nn.Dropout(0.5),      # 训练时每次随机丢 50%
    nn.Linear(256, 10),
)
```

**为什么防过拟合**：强迫神经元不能过度依赖单个特征——某个神经元有 50% 概率被抹掉，网络被迫学 robust 特征。

**放置位置**：全连接层之间；卷积层用 BN 已足够，不放 Dropout。

**train/eval 差异**：Dropout 完全依赖 `model.train()` / `model.eval()` 开关；忘 `eval()` 会让评估结果随机抖动。

### 2. 数据增强（Data Augmentation）

**机制**：每次读图时随机变换，训练集"变大"。

```python
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),           # 50% 概率水平翻转
    transforms.RandomCrop(32, padding=4),        # 加 4 圈 padding 后随机裁 32×32
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
```

**为什么防过拟合**：每次读同张图都给一个变体，模型永远看不到"两张完全相同的图"，无法背答案。

**关键规则**：**只用于训练集，不用于测试集**。测试集要衡量"真实分布"上的表现，不能人为改造。

**CIFAR-10 经验**：RandomHorizontalFlip + RandomCrop 是标配；不能用 VerticalFlip（倒过来的飞机不是训练目标）。

### 3. Weight Decay（权重衰减 / L2 正则）

**机制**：在损失里加 $\lambda \cdot \sum w^2$，惩罚大权重。

$$
\text{新 loss} = \text{原 loss} + \lambda \cdot \sum_i w_i^2
$$

**为什么防过拟合**：过拟合模型往往有大权重（某个 W 特别大只为记住某张图）。加此项后大 W 被惩罚，模型倾向于用**小权重、简单解**。奥卡姆剃刀原理。

**PyTorch 用法**：优化器内置

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

**AdamW vs Adam**：AdamW 修正了 Adam 里 weight decay 的实现 bug，是**大模型微调的默认选择**。

### 4. 学习率调度（LR Scheduler）

**问题**：固定 lr 训到底不理想——初期需要大 lr 快下降，后期需要小 lr 精细收敛。

**Cosine Annealing**（现代最常用）：

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
# lr 按余弦曲线从 initial_lr 平滑降到 0
```

公式：$lr = 0.5 \cdot lr_0 \cdot (1 + \cos(\pi \cdot t / T))$

调用时机：**每个 epoch 结束调一次 `scheduler.step()`**。

### 5. Early Stopping（早停）

**机制**：监控验证 acc，连续 N 个 epoch 不涨就停，保存最佳权重。

```python
best_acc = 0
patience_counter = 0

for epoch in range(n_epochs):
    ...
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best.pt")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break
```

**价值**：自动检测过拟合起点 + 保存最佳权重版本。

### 6. AMP 混合精度训练（Automatic Mixed Precision）

**机制**：大部分运算用 float16（省显存、快 2-3 倍），敏感处保 float32。

**核心 4 步**（替换普通 5 步）：

```python
optimizer.zero_grad()
with autocast("cuda"):                    # 前向和 loss 在 float16
    logits = model(x)
    loss = loss_fn(logits, y)
scaler.scale(loss).backward()             # 缩放 loss 防溢出
scaler.step(optimizer)                    # unscale 梯度 + 更新参数
scaler.update()                           # 调整下次缩放因子
```

**Loss Scaling 原理**：float16 数值范围窄（±65504），训练后期梯度太小会归零。Scaler 把 loss 放大一个大数（如 65536），梯度也跟着放大避免下溢；更新前再除回来。

**效果**：显存减半 + 速度 2-3 倍，精度几乎无损。**大模型必用**。

### 7. wandb（实验追踪）

**机制**：把每 epoch 的指标发到云端，浏览器看曲线。

```python
import wandb
wandb.init(project="...", config={...})
wandb.log({"train_loss": ..., "val_acc": ..., "lr": ...})
```

**必须掌握**：工业界标配，后面训 LLM 时会追踪几十个指标。

## 三、实验结果

30 epoch，CIFAR-10：

| 指标 | L2.3（无正则） | L2.4（有正则） |
| --- | --- | --- |
| 最终 train_loss | 0.23 | 0.55 |
| 最终 val_acc | 78.19% | **82.82%** |
| train-val 差距 | 大（过拟合） | 小（泛化好） |
| 过拟合迹象 | 有（第 9 epoch 后开始）| 无 |

**核心洞察**：train_loss 变大 + val_acc 变高 = **正则化用训练集准确率换泛化能力**。这是正则化的精髓。

**为什么没到 85%**：模型只有 5 层，容量有限。85%+ 通常需要 ResNet-18 级别（更深、残差连接）。L2.5 会讲。

## 四、我的误解与纠正

| 我的误解 | 正确 |
| --- | --- |
| `model.eval()` 的意义 = "用不同数据集" | eval() 是**层的行为开关**：关掉 Dropout/BN 的训练行为，让模型稳定预测。和数据集划分是两码事 |
| 数据增强不用测试集的原因 = "可复现" | 核心是要衡量模型在**真实数据分布**上的表现。测试集代表部署后遇到的真实图，不能改；可复现是次要理由 |

## 五、补充问答

### 1. `with autocast` 是什么

`with` 是 Python 上下文管理器：进入自动开、离开自动关。`autocast` 在 with 块内自动把运算切成 float16。

**必须包住前向 + loss 计算**（反向和参数更新在外面）。反向传播的精度是前向决定的，backward 不需要放在 autocast 里，PyTorch 官方文档明确要求。

### 2. Loss Scaling 具体流程

float16 范围窄导致小梯度下溢：

```text
正常:  loss(小) → grad(极小,float16 归零) → 更新失败
放大:  loss × 65536 → grad × 65536(不再下溢) → 更新前除回 → 正确更新
```

`scaler.step()` 内部自动 unscale；`scaler.update()` 动态调整缩放因子（无溢出就增大，有溢出就减小并跳过这步）。

### 3. 前面实验为什么不用 AMP

L1.3~L2.3 都用 float32 全精度训练——PyTorch 默认，精度高但慢。**AMP 是性能优化**，不是必要功能，小模型差别不大。**大模型必用**（LLM 用 float32 显存直接爆），后面 Stage 4 训 LLM 会必用。

## 六、实验产物

- 代码：[lab/train.py](lab/train.py)
- 最佳权重：`lab/best.pt`
- wandb 项目：`cifar10-cnn-regularization`
- 结果：val_acc **82.82%**（vs L2.3 的 78.19%，提升 4.6 个百分点）
