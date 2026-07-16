# Lab: 在 CIFAR-10 上训练 CNN

## 目标

- 亲手实现一个简单 CNN，在 CIFAR-10 上达到 **75%+** 验证准确率
- 对比 MLP 在同数据集上（约 45%），感受 CNN 对图像任务的优势
- 掌握 Conv2d + BatchNorm2d + ReLU + MaxPool 的组合

## 要求

打开 `train.py`，你会看到 3 个 `# TODO`。**必须自己写这三段**：

1. **TODO 1**：定义 CNN 结构（3 组 CBR + 分类头）
2. **TODO 2**：实现 `forward()`（卷积部分 → flatten → 全连接分类头）
3. **TODO 3**：训练主循环 4 步（和 L2.2 一样，复用你的模板）

真的卡住了参考 [L2.2 的 train.py](../../02_mlp_mnist/lab/train.py)，模板几乎一样。

## 运行环境

- 推荐机器：3070 笔记本或 5070Ti（有 GPU 强烈推荐，CNN 训练量比 MLP 大 5-10 倍）
- 预估显存：< 2 GB
- 预估时长：3070 GPU 上约 3-5 分钟（10 epoch）；CPU 上会等到你怀疑人生
- 首次运行自动下载 CIFAR-10（约 170MB），之后本地缓存

## 网络结构（TODO 1 参考）

```
输入 (batch, 3, 32, 32)
    ↓ Conv2d(3, 32, 3, padding=1) → BN → ReLU → MaxPool(2)
(batch, 32, 16, 16)
    ↓ Conv2d(32, 64, 3, padding=1) → BN → ReLU → MaxPool(2)
(batch, 64, 8, 8)
    ↓ Conv2d(64, 128, 3, padding=1) → BN → ReLU → MaxPool(2)
(batch, 128, 4, 4)
    ↓ flatten
(batch, 128*4*4 = 2048)
    ↓ Linear(2048, 256) → ReLU → Linear(256, 10)
输出 logits (batch, 10)
```

## 验收标准

- [ ] 代码能跑通，无报错
- [ ] 每 epoch 打印训练 loss 和验证 accuracy
- [ ] 10 个 epoch 后验证 accuracy > 75%
- [ ] 能解释每个 TODO 的作用

## 提示

- **`padding=1` + `kernel_size=3`**：让卷积不改变空间尺寸（配合 MaxPool 每次砍半）
- **CBR 三件套顺序**：Conv2d → BatchNorm2d → ReLU（不是 ReLU → BN！）
- **flatten 时**：可以用 `x.flatten(1)` 或 `nn.Flatten()`，把 (batch, 128, 4, 4) 拉成 (batch, 2048)
- **CIFAR-10 数据集**：`torchvision.datasets.CIFAR10`，用法和 MNIST 几乎一样，只是图是彩色的
