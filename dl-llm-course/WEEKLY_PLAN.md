# 周计划

> 节奏参考,允许提前完成。实际进度以 [PROGRESS.md](PROGRESS.md) 为准。

## 当前进度快照(第 5 天,2026-07-18)

- ✅ Stage 0 环境与工程基建
- ✅ Stage 1 数学基础(概率/信息论/优化)
- ✅ Stage 2 神经网络与训练(MLP/CNN/RNN/正则化)
- 🔶 Stage 3 进行中:L3.1 注意力已完成,下一步 L3.2 手写 Transformer

---

## 第 5 周(本周):Stage 3 收尾 - 注意力与 Transformer

| 课 | 内容 | 预计时长 | 产出 |
| --- | --- | --- | --- |
| L3.1 注意力机制 | ✅ 已完成 | - | 手写 Multi-Head Attention,验证通过 |
| **L3.2 从零手写完整 Transformer** | 位置编码、Encoder/Decoder、mask、FFN、残差+LayerNorm;在字符级语言建模任务上训练 | 3-4 天(本周重点) | 一个能生成文本的完整 Transformer |
| L3.3 预训练范式 | BERT(MLM) vs GPT(自回归)、tokenization、scaling | 1 天 | 理解 + 论文精读笔记 |

**本周目标**:完成 Stage 3,能默画 Transformer 架构图、解释每个组件、亲手训出一个能生成文本的 Transformer。

**难点预警**:L3.2 手写 Transformer 是课程第一堵墙。分步推进:位置编码 -> 单层 Transformer block -> 完整模型 -> 训练循环。卡住随时问。

---

## 第 6 周:Stage 4 前半 - LLM 工程基础

| 课 | 内容 | 预计时长 | 产出 |
| --- | --- | --- | --- |
| L4.1 HuggingFace 生态 | transformers/datasets/tokenizers、加载推理开源模型 | 1-2 天 | 本地跑通 7B 推理(5070Ti 或 3070 4bit) |
| L4.2 PEFT(LoRA/QLoRA) | 低秩分解原理、量化、bitsandbytes | 2 天 | 在 3070(8GB)上 QLoRA 微调小模型 |

**本周目标**:掌握 LLM 工程栈,能加载开源模型推理 + 用 QLoRA 微调。

---

## 第 7 周:Stage 4 后半 - 分布式与部署(双机+云)

| 课 | 内容 | 预计时长 | 产出 |
| --- | --- | --- | --- |
| L4.3 分布式与局域网多机 | DDP、torch.distributed、accelerate、双机组网 | 2-3 天 | 3070+5070Ti 跑 DDP 训练 |
| L4.4 推理优化 | KV cache、量化推理、vLLM | 1-2 天 | vLLM 部署 + 压测对比 |
| L4.5 模型服务化(新增) | FastAPI 封装 vLLM、并发处理、locust 压测 | 1-2 天 | 一个 HTTP API 服务 + 性能报告 |

**本周目标**:完成 L0.3 延后的双机 DDP;掌握模型部署和服务化全链路。

---

## 第 8-9 周:Stage 5 - 对齐基础

| 课 | 内容 | 预计时长 |
| --- | --- | --- |
| L5.1 SFT 指令微调 | 指令数据、chat 格式、SFT 训练 | 2 天 |
| L5.2 RLHF | 奖励模型、PPO、KL 约束 | 3-4 天 |
| L5.3 DPO 与偏好优化 | DPO 推导、与 RLHF 对比 | 2 天 |
| L5.4 Constitutional AI | 原则驱动、RLAIF | 1-2 天 |

**目标**:理解从预训练到"有用且无害"助手的对齐链路。**为 Stage 6 安全专题铺底**--KL 约束、reward hacking 等概念直接连到安全。

---

## 第 10-13 周:Stage 6 - 大模型安全专题(核心重心)

| 课 | 内容 | 预计时长 |
| --- | --- | --- |
| L6.1 威胁模型与安全总览 | OWASP LLM Top 10、评测基准 | 2 天 |
| L6.2 越狱 Jailbreak | 手法分类、GCG/PAIR 自动化攻击、防御 | 1 周 |
| L6.3 提示注入 | 直接/间接注入、RAG/Agent 风险、隔离防御 | 1 周 |
| L6.4 对齐安全与失败模式 | reward hacking、over-refusal、安全评测 | 3-4 天 |
| L6.5 红队与自动化评估 | 自动化红队 pipeline、安全基准复现 | 3-4 天 |

**目标**:这是你的核心方向,放慢、加练、多复现。每个子课题都有动手实验(攻击+防御)。

---

## 第 14 周:Stage 7 - 毕业项目

**四选一**(详见 curriculum.md L7.4):

- A:红队自动化工具
- B:越狱检测与防御系统
- C:论文复现+扩展
- D:对齐安全评测平台

**目标**:端到端做一个能写进简历的项目。

---

## 节奏说明

- **每周 20+ 小时**,按上面节奏约 14 周完成全部。
- **允许提前**:你这 4 天已完成 Stage 0+1+2(原计划 4-5 周),明显快于预估。提前完成本周计划直接说,我给下一周。
- **可放慢**:L3.2(手写 Transformer)、Stage 6(安全专题)是难点,允许一周学一周半。
- **实际进度以 PROGRESS.md 为准**,周计划只是导航。
