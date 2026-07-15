# Lab: 在 MNIST 上训练一个 MLP

## 目标

- 亲手实现一个 MLP，能在 MNIST 上达到 97%+ 验证准确率
- 独立完成 `nn.Module` 定义 + 前向传播 + 训练循环 + 准确率计算
- 感受深度学习的"生活方式"：DataLoader、train/val 划分、逐 epoch 观察 loss 和 acc

## 要求

打开 `train.py`，你会看到 4 个 `# TODO` 标记。**必须自己写这四段**，写完运行代码验证：

1. **TODO 1**：定义 MLP 的层结构（784 → 256 → 128 → 10，中间加 ReLU）
2. **TODO 2**：实现 `forward()`，把输入按层过一遍
3. **TODO 3**：写训练一步的 4 步循环（zero_grad → forward → backward → step）
4. **TODO 4**：计算一个 batch 的准确率

写不出来别急着看答案：先回顾 L2.1 笔记的"训练标准 4 步"和 MLP 结构图。真的卡住了，参考解在 `reference/train_solution.py`。

## 运行环境

- 推荐机器：3070 笔记本（GPU 会快很多；CPU 也能跑，只是慢）
- 预估显存：< 1 GB
- 预估时长：GPU 上约 1 分钟；CPU 上约 5 分钟
- 首次运行会自动下载 MNIST（约 12 MB），之后缓存在本地

## 验收标准

- [ ] 代码能跑通，无报错
- [ ] 每 epoch 打印训练 loss 和验证 accuracy
- [ ] 5 个 epoch 后验证 accuracy > 97%
- [ ] 能解释 4 个 TODO 每一段代码的作用

## 提示

- `nn.Linear(in_features, out_features)` 是全连接层
- `nn.ReLU()` 是激活函数
- MNIST 图是 28×28，MLP 需要**展平**成 784 维向量。用 `x.view(-1, 784)` 或 `nn.Flatten()`
- 准确率 = 预测正确的数量 / 总数量。预测类别 = logits 最大值所在的位置（`argmax(dim=1)`）
