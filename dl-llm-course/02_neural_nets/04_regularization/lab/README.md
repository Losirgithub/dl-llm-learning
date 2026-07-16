# Lab: CIFAR-10 CNN 加正则化，冲击 85%+

## 目标

把 L2.3 的 CNN(78%)通过**六大正则化/训练技巧**顶到 **85%+**：

1. Dropout —— 全连接层随机丢
2. 数据增强 —— 训练集加 RandomFlip + RandomCrop
3. Weight Decay —— AdamW + 惩罚大权重
4. 学习率调度 —— CosineAnnealingLR 平滑降 lr
5. Early Stopping —— 验证 acc 不涨 5 epoch 就停
6. AMP 混合精度 —— float16 加速训练

同时接入 **wandb** 可视化训练过程。

## 要求

打开 `train.py`，找到 2 个 `# TODO`：

1. **TODO 1**：在 CNN 里加 Dropout（决定加在哪、p 多少）
2. **TODO 2**：训练循环改成 AMP 版本（4 步替换普通 5 步）

其他都给完整了，读懂即可。

## 前置准备

1. 注册 wandb 免费账号：<https://wandb.ai>
2. 命令行登录：`wandb login`（粘贴网页里给的 key）
3. 装依赖：`pip install wandb`

## 运行环境

- 推荐机器：3070 或 5070Ti
- 预估显存：< 2 GB
- 预估时长：约 5-10 分钟（30 epoch，可能 early stopping 提前结束）

## 验收标准

- [ ] 代码跑通，wandb 能看到训练曲线
- [ ] 30 epoch 后 val acc > 85%
- [ ] 能对比 L2.3(78%)看清正则化威力
- [ ] 观察 train/val 差距（对比 L2.3 明显缩小）

## 提示

- **Dropout 放位置**：卷积层之间用 BN 已够，Dropout 主要加在**全连接层之间**（如 `Linear→ReLU→Dropout→Linear`）
- **AMP 是替换原有 5 步，不是新增**：`autocast` 包住前向和 loss 计算，`scaler` 替换 backward 和 step
- **数据增强只用于训练集**：test_transform 不加增强
