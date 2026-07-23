# L4.1 HuggingFace 生态 · 笔记

## 一、HuggingFace 是什么

**HuggingFace 是 AI 界的 GitHub**--模型/数据集/代码的共享平台。三样东西:

1. **Hub**(模型仓库):几十万个开源模型，`from_pretrained("模型名")` 一行下载使用
2. **Transformers 库**:加载/使用这些模型的 Python 库(L3.3 已用)
3. **生态库**:datasets、tokenizers、accelerate、peft、bitsandbytes

**为什么重要**:几乎所有开源 LLM(Llama、Qwen、Mistral、DeepSeek)都发布在 HF 上，都用 transformers 库加载。**掌握 HF = 掌握用任何开源大模型的能力**。

## 二、HF 生态库速览

| 库 | 作用 | 何时用 |
| --- | --- | --- |
| `transformers` | 加载/推理/训练模型 | L3.3 已用，L4.1 主力 |
| `datasets` | 加载/处理数据集 | L4.2 微调时 |
| `tokenizers` | 高效分词(BPE) | 通常 transformers 内部调 |
| `accelerate` | 分布式/多设备训练 | L4.3 多机训练 |
| `peft` | 参数高效微调(LoRA) | L4.2 微调 |
| `bitsandbytes` | 量化(4bit/8bit) | L4.1 把大模型塞进小显存 |

## 三、加载 LLM 的标准三步:AutoClass

L3.3 用具体类名 `GPT2LMHeadModel`--每个模型类名不同(GPT2LMHeadModel、LlamaForCausalLM、Qwen2ForCausalLM...)，记不住。

**实际工程用 AutoClass**--HF 的"自动识别"机制:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
```

**`AutoModelForCausalLM`**:不管什么模型架构(GPT/Llama/Qwen)，自动识别并用对应的类加载。**换模型不用改代码**。

**`CausalLM` = 因果语言模型 = 自回归 = GPT 式**(只能看左边，L3.2 写的那种)。

### LM 是什么缩写

**LM = Language Model(语言模型)**--预测下一个 token 的模型。

- `GPT2LMHeadModel` = GPT-2 的语言模型版本(带生成头 `lm_head`)
- `BertForMaskedLM` = BERT 做掩码语言模型任务
- 名字带 `LM` = 这模型是用来做"语言建模"任务(预测词)的

## 四、Base 模型 vs Instruct 模型(重要)

模型名常带 `Instruct` 或 `Chat`:

- `Qwen2.5-1.5B`--**Base 模型**(纯预训练，只会续写)
- `Qwen2.5-1.5B-Instruct`--**Instruct 模型**(经过指令微调，会对话)

### 区别

```text
Base 模型(预训练产物):
  输入 "中国的首都是"
  输出 "北京天津上海..."  ← 续写，可能跑题

Instruct 模型(预训练 + 指令微调):
  输入 "用户: 中国的首都是哪? 助手:"
  输出 "中国的首都是北京。"  ← 回答问题，听话
```

- **Base 模型**只会"续写文本"(L3.2 玩具 GPT 就是 base)
- **Instruct 模型**经过指令微调(Stage 5 讲)，会"听指令回答"

**用哪个**:做对话/问答/助手 -> 用 **Instruct** 版。做研究/续写 -> 用 Base 版。

## 五、Chat Template(对话模板)

Instruct 模型怎么知道"这是用户问的，我该回答"?靠 **chat template**--一套格式约定。

每个模型有自己的模板，比如 Qwen:

```text
<|im_start|>user
中国的首都是哪?<|im_end|>
<|im_start|>assistant
```

`<|im_start|>` 和 `<|im_end|>` 是特殊 token，标记"谁在说话"和"说完没"。模型看到这格式就知道"该 assistant 回答了"。

**不用手写模板**--HF 提供 `apply_chat_template`:

```python
messages = [
    {"role": "user", "content": "中国的首都是哪?"},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

**`add_generation_prompt=True`**:末尾加"该 assistant 说了"的提示，模型从这开始生成。

## 六、生成对话完整流程

```python
# 1. 加载模型和 tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

# 2. 构造对话
messages = [{"role": "user", "content": "什么是深度学习?"}]

# 3. 应用 chat template
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 4. 生成
output = model.generate(**inputs, max_new_tokens=512)

# 5. 解码(只取新生成的部分)
generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(generated)
```

### 关键参数

**`torch_dtype="auto"`**:自动用模型合适的精度(fp16/bf16)。**大模型用 fp16 省一半显存**。

**`device_map="auto"`**:自动把模型放到合适的设备(GPU)。大模型放不下单卡时还能自动拆分到多卡。

**`.to(model.device)`**:输入搬到模型所在设备(L3.2 踩过这坑)。

**`output[0][inputs["input_ids"].shape[1]:]`**:只取新生成的部分，去掉 prompt。`shape[1]` 是 prompt 长度，从这之后切片。

## 七、4-bit 量化:把大模型塞进小显存

3070 只有 8GB。7B 模型 fp16 要 14GB，放不下。**4-bit 量化**把每个权重从 16 位压到 4 位，**显存降到 1/4**:

```text
7B 模型:
  fp16:  14 GB  ← 3070 放不下
  4-bit:  ~5 GB  ← 3070 能跑!
```

用 `bitsandbytes` 库:

```python
from transformers import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,                     # 4-bit 量化
    bnb_4bit_quant_type="nf4",            # 量化类型，nf4 是 QLoRA 论文推荐
    bnb_4bit_compute_dtype=torch.bfloat16, # 计算时用 bf16(精度够)
)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    quantization_config=quant_config,
    device_map="auto",
)
```

**代价**:4-bit 略有精度损失，但推理质量几乎不变。**L4.2 微调时 QLoRA 就是用这个**。

## 八、我的误解与纠正

| 我的理解 | 精确修正 |
| --- | --- |
| "4-bit 量化让小显存也能训练模型" | 4-bit 主要用于**加载/推理**和 **QLoRA 微调**，不是从零训练。从零训练不能用 4-bit(精度不够更新权重)。是"让小显存也能**跑/微调**大模型" |

## 九、实验产物与分析

### 实验:加载 Qwen2.5-1.5B-Instruct 对话

- 代码:[lab/chat.py](lab/chat.py)
- 结果:模型 15.4 亿参数，加载到 cuda:0，能流畅回答中文问题(如"用三句话解释 Transformer")

### 本地部署的意义

模型完全在本地运行(下载到 `E:/hf_cache`，加载到 GPU，推理无网络请求)。这就是**本地部署**。

| | 本地部署(本次) | API 调用(ChatGPT/Claude) |
| --- | --- | --- |
| 模型在哪 | 你的硬盘 + GPU | 服务商服务器 |
| 运行在哪 | 你的机器 | 远程 |
| 成本 | 硬件 + 电费 | 按 token 付费 |
| 隐私 | 数据不出本地 | 数据发到服务商 |
| 可控 | 能改/微调 | 黑盒 |

**安全研究必须本地部署**:Stage 6 越狱/注入实验要在本地模型上跑攻击，不能攻击别人 API(违规)。

## 十、补充问答:模型的记忆与上下文管理

### 模型有记忆吗

**没有。模型是无状态的(stateless)**。每次 `model.generate()` 独立，模型不记住之前的对话。

### "记忆"怎么实现

**手动维护对话历史，每次调用全发**:

```python
messages = [
    {"role": "user", "content": "我叫小明"},
    {"role": "assistant", "content": "你好小明"},
    {"role": "user", "content": "我叫什么?"},   # 新问题
]
text = tokenizer.apply_chat_template(messages, ...)
# 模型看到完整历史，能回答"小明"
```

ChatGPT 的"记忆"也是每次重发历史。模型靠"看到历史"来"记住"。

### 上下文窗口(Context Window)

模型一次能"看到"的最大 token 数(输入 + 输出):

| 模型 | 上下文窗口 |
| --- | --- |
| Qwen2.5-1.5B | 32768 tokens |
| GPT-4 | 128000 tokens |
| Claude 3 | 200000 tokens |

超过窗口，早期对话被"挤出"，模型就"忘"了开头。

### 长对话的上下文管理策略

1. **截断**:只保留最近 N 轮，丢早期(简单粗暴)
2. **滑动窗口**:固定保留最近 K 个 token
3. **摘要**:定期让模型总结之前对话，用摘要替代原始历史(ChatGPT 长对话的做法)
4. **检索 RAG**:历史存外部数据库，每次检索相关几条塞进上下文(Agent 记忆)

### 比喻

模型 = 只看一次试卷的考生。你(程序)负责管理"试卷"(对话历史)--决定印哪些、怎么压缩。模型只看这次递给它的试卷，答完就忘。

## 十一、关键工程坑

### 1. HF 模型下载到 C 盘爆满

默认缓存 `C:\Users\...\.cache\huggingface`。设 `os.environ["HF_HOME"]="E:/hf_cache"`(import transformers 之前)。或全局设用户环境变量。

### 2. HF 国内下载失败

设 `os.environ["HF_ENDPOINT"]="https://hf-mirror.com"`(import transformers 之前)。

### 3. `torch_dtype` deprecated

新版 transformers 用 `dtype` 代替 `torch_dtype`，功能一样。

### 4. attention_mask warning

`generate` 不传 attention_mask 且 pad==eos 时报警告。单条生成(batch=1)无害可忽略。
