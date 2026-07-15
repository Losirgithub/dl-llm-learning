# Lab: 用 PyTorch autograd 重写 L1.3 的优化器

## 目标
- 亲手体验：只写前向计算，PyTorch 自动算梯度
- 对比 L1.3 手写版和 PyTorch 版，直观感受 autograd 的价值
- 掌握"前向 → loss → backward → 更新"的完整训练循环

## 要求
- 用 `torch.tensor(..., requires_grad=True)` 建可训练参数
- 只定义损失函数（前向），不手推梯度
- 用 `loss.backward()` 自动算梯度
- 用 PyTorch 内置的 `torch.optim.Adam` 更新参数
- 打印每一步的 (x, y) 和 loss，观察收敛

## 运行环境
- 推荐机器：3070 笔记本（CPU 就够，无需 GPU）
- 预估显存：0
- 预估时长：3 秒

## 验收标准
- [ ] 代码能跑，打印 50 步的收敛过程
- [ ] 最终损失和 L1.3 手写 Adam 接近（数量级一致即可）
- [ ] 能对比 L1.3 手写版和 PyTorch 版的代码差异
