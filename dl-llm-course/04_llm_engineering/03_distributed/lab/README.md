# Lab: DDP 分布式训练(单机跑通)

## 目标

- 掌握 DDP 代码结构:init_process_group、DDP 包装、DistributedSampler
- 用 torchrun 启动训练
- 单机单进程跑通(验证代码),为双机做准备

## 要求

打开 `train_ddp.py`，找到 3 个 `# TODO`：

1. **TODO 1**：初始化分布式进程组 + 获取 local_rank
2. **TODO 2**：用 DDP 包装模型
3. **TODO 3**：用 DistributedSampler 切分数据

## 运行方式(注意:用 torchrun,不是 python)

```bash
# 单机单进程跑通(验证 DDP 代码结构)
torchrun --nproc_per_node=1 train_ddp.py
```

## 运行环境

- 推荐机器：3070 或 5070Ti
- 数据：MNIST(已在 data/MNIST,不用重下)
- 预估时长：< 1 分钟

## 验收标准

- [ ] torchrun 能启动,打印 "进程初始化成功"
- [ ] 训练正常跑,loss 下降
- [ ] 能解释 DDP 的三步:init、wrap、sampler

## 双机扩展(可选挑战)

跑通单机后,两台机器联机的命令(在 README 末尾):

```bash
# 机器A(主节点,IP 192.168.x.x)
torchrun --nnodes=2 --nproc_per_node=1 --master_addr=<主节点IP> --master_port=29500 train_ddp.py
# 机器B(从节点)
torchrun --nnodes=2 --nproc_per_node=1 --master_addr=<主节点IP> --master_port=29500 train_ddp.py
```
