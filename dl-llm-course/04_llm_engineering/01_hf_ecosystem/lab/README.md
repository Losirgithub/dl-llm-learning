# Lab: 加载 Qwen 大模型并对话 + 4-bit 量化

## 目标

- 用 AutoClass 加载真正的开源 LLM(Qwen2.5-1.5B-Instruct)
- 构造 chat template 和模型对话
- 体验 4-bit 量化,把 7B 模型塞进 8GB 显存

## 要求

打开 `chat.py`，找到 2 个 `# TODO`：

1. **TODO 1**：构造对话消息 + 应用 chat template
2. **TODO 2**：调用 generate 生成回复 + 解码

## 运行环境

- Qwen2.5-1.5B-Instruct(fp16 ~3GB):3070(8GB) 或 5070Ti(16GB) 都行
- Qwen2.5-7B-Instruct(4-bit ~5GB):3070 也能跑(用 4-bit 量化)
- 首次下载模型(1.5B ~3GB，7B ~15GB)，用 hf-mirror 镜像

## 验收标准

- [ ] 模型能正常回复中文/英文问题
- [ ] 4-bit 量化的 7B 模型能跑通(显存 < 8GB)
- [ ] 能解释 chat template、device_map、4-bit 的作用

## 前置准备

```bash
pip install transformers accelerate bitsandbytes
```
