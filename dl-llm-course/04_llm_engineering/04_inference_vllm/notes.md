# L4.4 推理优化与部署(vLLM)· 笔记(理论版,执行跳过)

> 理论与 API 已掌握,执行因 Windows 不支持 vLLM 跳过。正经推理部署用 Linux/WSL2。

## 一、为什么 LLM 推理慢

### 自回归生成的代价

生成 N 个 token = N 次前向传播。每次 attention 要算 `Q(新) × K(全部) × V(全部)`:

```text
生成 token 1:  Q₁ × [K₁] × [V₁]
生成 token 2:  Q₂ × [K₁,K₂] × [V₁,V₂]
...
生成 token 100: Q₁₀₀ × [K₁...K₁₀₀] × [V₁...V₁₀₀]
```

每次重算前 N 个 token 的 K、V--重复计算。复杂度 O(N²)。

## 二、KV cache:缓存过去的 K/V

### 解法

把已算过的 K、V 存起来,下次直接用,不重算。每个 token 只算自己的 K/V 一次。复杂度从 O(N²) 降到 O(N)。

**新 token 的 Q、K、V 都要算**(它是新词),但**过去 token 的 K、V 不用重算**--缓存复用。

### KV cache 显存计算

公式:`2 × num_layers × num_heads × head_dim × dtype_size × seq_len`

以 7B 模型、32 层、序列 2048、fp16 为例:

| 数字 | 含义 |
| --- | --- |
| 2 | K 和 V 两个 |
| 32(第一个) | 层数 num_layers |
| 32(第二个) | 注意力头数 num_heads |
| 128 | 每头维度 head_dim = d_model/num_heads = 4096/32 |
| 2字节 | fp16 |
| 2048 | 序列长度 |

$$
2 \times 32 \times 32 \times 128 \times 2 \times 2048 = 1{,}073{,}741{,}824 \text{ 字节} \approx 1 \text{ GB}
$$

**序列 2048 时 KV cache 就 1GB**。32K 上下文要 16GB--比模型权重(7B fp16 ~14GB)还大。这就是 PagedAttention 存在的意义。

### 代价

占显存,随序列长度线性增长。长序列 + 大模型,KV cache 能占几十 GB。

## 三、vLLM 两大创新

### 创新 1:PagedAttention(KV cache 的虚拟内存)

借鉴 **OS 分页机制**:KV cache 分成固定大小"块"(如每块 16 token),一个请求的 KV cache 不连续存放,分散在不同块里,用"块表"记录映射(像 OS 页表)。

好处:
- **无内存碎片**:零散块都能用,显存利用率 ~60% -> ~96%
- **共享**:相同前缀(如系统提示)的请求共享同一块 KV cache

### 创新 2:Continuous Batching(连续批处理)

传统批处理:凑齐一批一起跑,最慢的没完不能接新请求(像公交车满员发车)。

Continuous Batching:每一步动态调整批次,某请求生成完**立刻移出**,空位让给新请求(像传送带随时上下)。吞吐量大幅提升。

### 效果对比

| | transformers | vLLM |
| --- | --- | --- |
| 吞吐量 | 1x | **10-20x** |
| 显存利用 | ~60% | ~96% |
| 批量推理 | padding 浪费 | 连续批处理 |
| KV cache | 连续分配,碎片 | PagedAttention |

## 四、vLLM API

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")
outputs = llm.generate(["你好"], SamplingParams(temperature=0.7, max_tokens=200))
```

或起服务(OpenAI 兼容):
```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000
```

## 五、OpenAI SDK 的作用

openai Python 包是客户端库,能指向任何兼容 OpenAI 接口的服务器(改 `base_url`):

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")  # 指向本地 vLLM
```

**价值**:
- 代码通用:同一套代码,改 base_url 就能调本地 vLLM / OpenAI / 各种兼容服务
- 本地测试,线上部署不用改代码
- OpenAI API 是事实标准,vLLM/Ollama/Together 都兼容

**对 Stage 6 安全研究的意义**:用 openai SDK 写攻击脚本,base_url 指向本地 vLLM(快速免费),同一套代码也能打真实 API。**vLLM + openai SDK 是安全研究标准工具链**。

## 六、7B 模型为什么是 32 层

### 7B = 7 Billion = 70 亿参数

参数量衡量模型规模。B = Billion = 10亿。

### 32 层是设计选择

每层参数 ≈ 7 × d_model²(attention 4d² + FFN 3d²)。Llama-7B 配置 d_model=4096, num_layers=32:

$$
\text{每层} \approx 7 \times 4096^2 \approx 1.17 \text{ 亿}, \quad 32 \text{ 层} \approx 37.5 \text{ 亿} + \text{嵌入/输出层} \approx 70 \text{ 亿}
$$

层数和宽度(d_model)共同决定总参数。32×4096 是 7B 的经验最优配置(Llama 开头,后来者照搬)。层数多=深=抽象强但难训练;层数少=浅=好训练但抽象弱。**层数是设计选择,不是 7B 决定的**。

## 七、我的误解与纠正

| 知识点 | 内容 |
| --- | --- |
| 梯度存在哪(L4.3 补) | `param.grad`,backward 填充、step 使用、zero_grad 清空。DDP 同步的就是它 |
| KV cache 只缓存 K/V | 新 token 的 Q/K/V 都要算(它是新词),但过去 token 的 K/V 不用重算 |
| 环境变量脚本 vs shell | 脚本内 os.environ 只对脚本 import 的库有效;启动器(torchrun)需要的变量必须在 shell 设 |

## 八、为什么执行跳过

vLLM 在原生 Windows 装不上:
1. **无 Windows 预编译包**:下载源码要从源码编译,需 CUDA 编译工具链
2. **长路径限制**:vLLM 目录结构超过 Windows 260 字符限制

**正经推理部署用 Linux/WSL2**。Stage 6 真要用时一次性搭 WSL2 + vLLM + 其他工具。

## 九、实验产物

- 理论与 API 已掌握
- 执行跳过(Windows 不支持 vLLM),Stage 6 时上 WSL2
