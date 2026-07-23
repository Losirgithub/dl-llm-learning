# Lab: 体验预训练模型(GPT-2 生成 + BERT 遮词)

## 目标

- 用 HuggingFace 加载真正的预训练模型(GPT-2 small / BERT-base)
- 对比 L3.2 你手写的玩具 GPT(50 万参数) vs 真正的 GPT-2(1.24 亿参数)
- 体验 BERT 的 MLM(遮词预测)
- 看 BPE tokenizer 怎么切词

## 要求

打开 `play_pretrained.py`，找到 2 个 `# TODO`：

1. **TODO 1**：用 GPT-2 生成文本(改 prompt 和生成长度，对比 L3.2 的输出)
2. **TODO 2**：用 BERT 做 MLM(手动遮一个词，看 BERT 预测 top5)

## 运行环境

- 推荐机器：3070(8GB) 或 5070Ti(16GB)
- 预估显存：< 2 GB
- 预估时长：首次下载模型约几分钟(GPT-2 ~500MB，BERT ~440MB)，之后秒级

## 验收标准

- [ ] GPT-2 生成流畅英文(对比 L3.2 的重复文本，感受规模差距)
- [ ] BERT 遮词预测给出合理 top5
- [ ] 能解释为什么预训练模型比 L3.2 玩具模型强这么多

## 提示

- GPT-2 生成：`model.generate(input_ids, max_new_tokens=100, ...)`，用 temperature/top_p 控制随机性
- BERT MLM：把目标词换成 `[MASK]` token，看模型对 MASK 位置的预测分布
