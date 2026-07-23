# L4.3 分布式训练 · 笔记(理论版,执行跳过)

> 理论与代码结构已掌握,执行因 Windows 环境坑跳过。正经分布式训练应用 Linux/WSL2。

## 一、为什么需要分布式训练

单卡训不动:数据太多、模型太大、太慢。分布式用多 GPU 成倍加速。

## 二、数据并行(DDP)原理

```
模型复制两份:
  机器A: 模型副本1 + 数据前一半
  机器B: 模型副本2 + 数据后一半

各自训练:
  A: forward -> loss -> backward -> grad_A
  B: forward -> loss -> backward -> grad_B

同步梯度(AllReduce):
  两台交换梯度,平均 -> 都得到 (grad_A + grad_B)/2

各自 step:
  两台参数始终一致,相当于 2 倍 batch 训练
```

**关键点**:模型复制(不是拆分)、数据切分、backward 后同步 `.grad`、参数一致。

### 梯度存在哪:`.grad` 属性

每个参数 `backward()` 后梯度存在 `param.grad`。`zero_grad` 清空、`backward` 填充、`step` 使用。**DDP 同步的就是 `.grad`**。

### 数据并行 vs 模型并行

| | 数据并行(DDP) | 模型并行 |
| --- | --- | --- |
| 怎么分 | 数据分,模型复制 | 模型拆,数据相同 |
| 适用 | 模型放得下单卡 | 模型放不下(70B+) |
| 难度 | 简单成熟 | 复杂 |

## 三、torchrun + torch.distributed

### 启动命令

```bash
# 单机
torchrun --nproc_per_node=1 train_ddp.py

# 双机(机器A主节点)
torchrun --nnodes=2 --nproc_per_node=1 \
    --master_addr=<主节点IP> --master_port=29500 train_ddp.py
```

参数:`--nnodes`(机器数)、`--nproc_per_node`(每机进程数=GPU数)、`--master_addr`(主节点IP)。

### 后端选择

- **nccl**:NVIDIA 官方,Linux 专用,GPU 通信最快
- **gloo**:跨平台,**Windows 只能用这个**,稍慢

### DDP 代码三步

```python
# 1. init
dist.init_process_group(backend="gloo")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

# 2. wrap
model = MLP().to(device)
model = DDP(model, device_ids=[local_rank])  # backward 自动同步梯度

# 3. sampler
sampler = DistributedSampler(train_set)  # 自动按 world_size 切分数据
loader = DataLoader(train_set, sampler=sampler)
# 训练循环里: sampler.set_epoch(epoch) 每个 epoch 重新打乱
```

**DDP 魔法**:`model = DDP(model)` 包装后,`backward()` 自动同步梯度。`DistributedSampler` 自动切分数据。

### accelerate 简化封装

HuggingFace `accelerate` 自动处理 init/DDP包装/梯度同步,代码几乎和单卡一样:

```python
from accelerate import Accelerator
accelerator = Accelerator()
model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
accelerator.backward(loss)  # 代替 loss.backward
```

启动:`accelerate launch --num_processes=2 train.py`。实际工程多用 accelerate 或 transformers.Trainer。

## 四、Windows 分布式的坑(为什么跳过执行)

1. **libuv**:torchrun 的 TCPStore 默认用 libuv,Windows 版 PyTorch 没编译。需 shell 设 `USE_LIBUV=0`(脚本里设太晚,torchrun 启动器在脚本前就报错)
2. **gloo 后端**:Windows 不能用 nccl,只能 gloo(慢+不稳)
3. **环境一致**:两台机器要相同 PyTorch/代码/数据
4. **防火墙**:要开放 master_port

**结论**:正经分布式训练用 **Linux/WSL2**(能上 nccl,无 libuv 问题)。Windows 双机 DDP 仅作学习理解。

## 五、我的误解与纠正

(本课主要补了 `.grad` 属性这个之前没讲的基础知识)

| 知识点 | 内容 |
| --- | --- |
| 梯度存在哪 | `param.grad`,backward 填充、step 使用、zero_grad 清空。DDP 同步的就是它 |
| 环境变量在脚本 vs shell | 脚本内 `os.environ` 只对脚本 import 的库有效;启动器(torchrun)需要的变量必须在 shell 设 |

## 六、实验产物

- 代码:[lab/train_ddp.py](lab/train_ddp.py)(DDP 三步结构完整,单机可读)
- 执行:跳过(Windows libuv 坑),理论理解达标
