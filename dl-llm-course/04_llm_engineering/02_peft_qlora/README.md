# L4.2 参数高效微调 PEFT: LoRA / QLoRA

## 学习目标

- 解释 PEFT 为什么只训练少量新增参数也能适配任务。
- 从公式上理解 LoRA 的低秩分解:冻结原权重,只训练两个小矩阵。
- 理解 QLoRA 的显存节省来源:4-bit NF4、double quantization、冻结量化基座、LoRA adapter。
- 在 3070(8GB)上跑通一个小模型 QLoRA 指令微调实验。

## 前置依赖(前置门控核对清单)

- [ ] L3.2:能说清 Transformer attention 里的 `q_proj` / `k_proj` / `v_proj` / `o_proj`。
- [ ] L4.1:能用 `AutoTokenizer` / `AutoModelForCausalLM` 加载模型,并使用 chat template。
- [ ] 工程:知道 HF 镜像与缓存目录必须在 import transformers 前设置。

## 理论要点

全参微调会更新模型里所有权重,显存主要花在参数、梯度、优化器状态和激活上。LoRA 的思路是冻结原始权重 `W`,只学习一个低秩增量:

```text
W' = W + ΔW
ΔW = B A
A ∈ R^{r×d}, B ∈ R^{k×r}, r << min(d, k)
```

QLoRA 再进一步:把冻结的基座模型以 4-bit 量化加载,梯度只穿过量化模型流向 LoRA adapter。这样训练的可学习参数很少,基座权重也不需要以 16-bit 全量常驻。

## 必需第三方库

- `transformers`:加载 tokenizer、模型、bitsandbytes 量化配置。
- `peft`:创建 LoRA adapter,准备 k-bit 训练模型。
- `trl`:使用 `SFTTrainer` 做 supervised fine-tuning。
- `datasets`:构造或加载指令数据集。
- `bitsandbytes`:4-bit / 8-bit 量化内核。

## 必读文献与文档

- LoRA paper:低秩适配的原始方法。
- QLoRA paper:NF4、double quantization、paged optimizer。
- Hugging Face PEFT quantization docs:当前 QLoRA 工程做法。
- Hugging Face TRL SFTTrainer docs:当前 SFT 数据格式与训练参数。

## 实验

- Lab:QLoRA 微调 Qwen2.5-0.5B-Instruct — 见 `lab/`

## 掌握判定标准(检查点)

- 概念:能回答 `quiz.md` 中 5 个问题。
- 实操:完成 `lab/finetune.py` 的 2 个 TODO,训练能启动并看到 loss 下降。
- 解释:能指出 LoRA 的 `r`、`lora_alpha`、`target_modules` 分别控制什么。
- 工程:能说清为什么 `HF_HOME` / `HF_ENDPOINT` 要放在 import transformers 前。
