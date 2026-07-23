# L3.2 从零手写完整 Transformer · 笔记

## 一、全局图:Transformer 的 5 个积木

```text
1. Multi-Head Attention   ← L3.1 已完成
2. Positional Encoding    ← 位置编码
3. LayerNorm              ← 层归一化
4. Residual Connection    ← 残差连接
5. Feed-Forward Network   ← FFN
```

光有 attention 不能叫 Transformer--attention 有个致命缺陷:**它不知道词的顺序**。其他积木分别解决"顺序""归一化""深了训不动""注意力之后要思考"等问题。

## 二、积木 1:Positional Encoding(位置编码)

### 问题:attention 是"无序"的

`Attention(Q,K,V) = softmax(QKᵀ/√d)V` 整个公式没有位置信息--只看 Q 和 K 内容算相似度。所以 "猫追狗" 和 "狗追猫" 对 attention 来说输入向量集合一样，输出也一样，但意思相反。

RNN 天然按顺序处理知道顺序，attention 为并行牺牲了顺序感，**必须人为加回位置信息**。

### 解法:词嵌入 + 位置编码

```text
最终输入 = 词嵌入 + 位置编码
         ↑           ↑
      语义信息     顺序信息
```

原论文用 sin/cos:

$$
PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d}), \quad PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})
$$

- pos:词在句子中的位置
- i:位置向量的第 i 维
- d:维度(等于词嵌入维度)

**为什么用 sin/cos**:

1. 每个位置编码唯一
2. 能泛化到更长序列(周期函数)
3. 相对位置可学(sin/cos 平移性质)

**关键**:位置编码是**固定公式算出来的**，不是可训练参数(`register_buffer`)。GPT 后来用可学习的位置编码(learned positional embedding)。

### 实现细节:sin/cos 公式怎么变成代码

公式:`PE(pos, 2i) = sin(pos / 10000^(2i/d))`，`PE(pos, 2i+1) = cos(pos / 10000^(2i/d))`

代码 6 行:

```python
pe = torch.zeros(max_len, d_model)
position = torch.arange(0, max_len).unsqueeze(1).float()
div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
pe = pe.unsqueeze(0)
```

**1. `torch.arange(0, max_len).unsqueeze(1)`**:`arange` 生成一维 `[0,1,...,max_len-1]`，`unsqueeze(1)` 加一维变 `(max_len, 1)` 竖列。**加维是为了后面和横行 div_term 广播相乘成矩阵**。

**2. `torch.arange(0, d_model, 2)`**:第三个参数是**步长 step**。`0::2` 等价于"从 0 开始步长 2"-> `[0, 2, 4, ..., d_model-2]`，这就是公式里的 `2i`。

**3. `torch.exp(... * (-math.log(10000.0) / d_model))`--数值稳定技巧**:公式分母是 `10000^(2i/d)`，代码要算它的倒数 `10000^(-2i/d)`。直接算 `10000^(大数)` 会溢出，改用 `exp(-log(...))`:

```text
10000^(-2i/d) = exp( log(10000^(-2i/d)) ) = exp( -2i/d · log(10000) ) = exp( 2i · (-log(10000)/d) )
```

**深度学习代码常见技巧**:用 exp/log 改写幂运算，避免溢出。

**4. `position * div_term`--广播**:

- position: `(max_len, 1)` 竖列
- div_term: `(d_model//2,)` 横行
- 广播相乘 -> `(max_len, d_model//2)` 矩阵

**竖列 × 横行 = 矩阵**，每个元素 (pos, i) = `pos / 10000^(2i/d)`，正好是 sin/cos 的参数。

**5. `pe[:, 0::2]` 和 `pe[:, 1::2]`--切片带步长**:

- `0::2`:从第 0 列开始步长 2 -> 偶数列(0, 2, 4, ...)
- `1::2`:从第 1 列开始步长 2 -> 奇数列(1, 3, 5, ...)

分别填 sin 和 cos。**Python 切片 `start:stop:step` 的用法**，stop 省略表示到底。

**6. `pe.unsqueeze(0)`**:加 batch 维 `(max_len, d_model)` -> `(1, max_len, d_model)`。**为了 forward 里 `x + self.pe[:, :seq]` 能和 `(batch, seq, d_model)` 广播对齐**。

**7. `register_buffer('pe', pe)` vs `Parameter`**:

- `Parameter`:可训练参数，优化器更新它(如 Linear 的 weight)
- `buffer`:不是参数不训练，但**跟着模型搬 GPU、存盘**

位置编码是固定公式不需要训练，用 buffer。普通属性 `self.pe = pe` 在 `.to(device)` 时不会搬 GPU 会报错。

### 切片步长语法总结

```python
tensor[start:stop:step]   # stop 省略=到底，step 省略=1
```

- `pe[:, 0::2]`:所有行 + 偶数列(步长 2 从 0 开始)
- `pe[:, 1::2]`:所有行 + 奇数列(步长 2 从 1 开始)
- `x[:5]`:前 5 个
- `x[::-1]`:反转

## 三、积木 2:LayerNorm(层归一化)

### 核心区别:沿哪个方向归一化

假设输入 (batch=3, seq=4, d=5):

```text
BatchNorm(沿 batch 归一化):
  看"3 个句子里的同一个位置、同一维特征"的均值方差
  -> 需要 batch，batch=1 时统计只有 1 个数，崩

LayerNorm(沿特征维归一化):
  看"这一个词的 5 维特征"的均值方差
  -> 不需要 batch，每个词自己归一化自己
```

**具体例子**:一个词的 5 维特征 `[0.2, 1.5, -0.3, 0.8, 0.1]`，LayerNorm 算这 5 个数的均值(0.46)方差，归一化到均值 0 方差 1。**和 batch 里其他词无关**。

### 公式

$$
\text{LN}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$

μ, σ² 是**这一个 token 的 d 维特征的均值方差**(不是 batch 的)。

### 为什么 Transformer 用 LN 不用 BN

| | BatchNorm | LayerNorm |
| --- | --- | --- |
| 归一化方向 | 跨样本(横着) | 单样本内部(竖着) |
| 依赖 batch | ✅ 是 | ❌ 不依赖 |
| 序列长度可变 | ❌ 难处理 | ✅ 每个词独立算 |
| batch=1 推理 | ❌ 崩 | ✅ 正常 |

**输入输出维度其实一样**(都是 batch, seq, d)。区别在沿哪个轴算统计量。

## 四、积木 3:Residual Connection + Add & Norm

### 问题:深了训不动

任何深层网络都有梯度消失问题。Transformer 原论文堆 6 层(现在 GPT 堆近 100 层)，不解决根本训不动。

### 解法:加捷径，让信息直接流过去

$$
\text{out} = x + \text{Sublayer}(x)
$$

让层学"残差"(和输入的差)，而不是完整变换。这层没用时梯度可走捷径传回去，不消失。

**类比**:下属把"原稿+修改"一起给你，而不是只给修改后版本。改错了还能看到原稿。

### Add & Norm = 残差(Add) + 归一化(Norm)

每个子层(attention 或 FFN)都包残差 + LayerNorm:

```text
out = LayerNorm(x + Sublayer(x))
```

**Add & Norm 不是独立的一层**，是"残差连接 + LayerNorm"两个积木的组合用法，原论文这么叫是因为它们总是一起出现。

### dropout 的放置位置(关键)

```python
x = self.norm1(x + self.dropout(attn_out))
#                 ↑           ↑
#              残差主干      子层输出
#              (保持干净)    (这里 dropout)
```

**dropout 加在子层输出上，不加在残差主干上**:

- 残差主干 x 必须保持干净--它是"捷径"，打断它梯度就传不回去，残差连接废了
- dropout 加在 attn_out 上 = "随机忽略部分子层贡献，但保留原始信息"

原论文原文:"We apply dropout to the output of each sub-layer, before it is added to the residual."

```python
# attention 子层
attn_out = self.attention(x)
x = self.norm1(x + self.dropout(attn_out))   # Add & Norm

# FFN 子层
ffn_out = self.ffn(x)
x = self.norm2(x + self.dropout(ffn_out))    # Add & Norm
```

## 五、积木 4:Feed-Forward Network(FFN)

### 是什么

```text
FFN(x) = Linear2( GELU( Linear1(x) ) )
```

三步，对一个 token 的 d 维向量:

1. **Linear1**: d_model -> 4*d_model (512 -> 2048) --**升维**
2. **GELU 激活**: 非线性筛选
3. **Linear2**: 4*d_model -> d_model (2048 -> 512) --**降维回来**

每个 token 独立做(position-wise)。

### 为什么"升维再降维"

**升维**:把信息投射到更大空间，挤在一起的特征被拉开，模型更容易区分不同模式。类比:小房间里混在一起的东西难分，搬到大厅铺开一眼能看出分类。

**GELU**:在升维后的 2048 维里做非线性筛选，把"有用的维度"放大，"没用的"压下去。**非线性变换让模型学到复杂模式**。

**降维**:压缩回原来大小，方便和残差连接相加(残差要求输入输出维度一致)。

### attention 和 FFN 的精确分工

| | attention | FFN |
| --- | --- | --- |
| 作用域 | **跨 token**(词和词之间) | **单 token 内部**(自己消化) |
| 做什么 | 信息**搬运**(从相关词拉信息) | 信息**变换**(非线性提取特征) |
| 类比 | 开会交流(收到别人信息) | 闭门思考(把收到的消化成自己的理解) |

**具体场景**:"猫 追 狗，因为 它 饿了"

- attention:"它"发现和"猫"最相关(权重 0.85)，新向量 = 0.85×猫的向量 + ...
- 但这只是"混了信息"，还没消化成"我是代词，指代猫"的抽象理解
- FFN:把混合向量做非线性变换，提取出"代词指代"这种高层语义特征

**两者缺一不可**:

- 只有 attention:信息能流动，但没深度加工，模型很"浅"
- 只有 FFN:每个词自己想，看不到上下文，"它"不知道指代谁

**交替进行**:attention(交流) -> FFN(思考) -> attention(再交流) -> ... 反复 N 层，理解越来越深。

### FFN 参数量占大头

- 一个 attention 层:4 个矩阵(Q/K/V/O)，各 d×d = 4d² (≈1M)
- 一个 FFN:两个矩阵，d×4d + 4d×d = 8d² (≈2M)

**FFN 参数是 attention 的 2 倍**。Transformer 大部分参数在 FFN。这也是 LLM 剪枝常动 FFN 的原因(如 MoE 把 FFN 拆成多个专家)。

## 六、积木 5:Masked Attention(Decoder 的核心)

### 问题:生成任务不能"偷看未来"

Encoder 双向(所有词互相看)。但生成任务是**逐词生成**:

```text
生成 "我 爱 深 度 学 习":
  生成 "我" 时,只能看到 "我",不能看到 "爱深度学习"
  生成 "爱" 时,只能看到 "我爱",不能看到 "深度学习"
```

偷看未来 = 作弊，训练学不到真本事，推理时(后面还没生成)就崩。

### 解法:Mask(掩码)

attention 算 softmax 前，把"未来位置"的分数**置成 -inf**:

```text
未 mask 的 scores (4×4):
  词0: [2.1, 3.5, 1.0, 0.5]    词0 能看到 词0,1,2,3
  词1: [1.8, 2.2, 4.1, 0.9]
  ...

加 mask(上三角置 -inf):
  词0: [2.1, -inf, -inf, -inf]    词0 只能看到 词0
  词1: [1.8, 2.2, -inf, -inf]     词1 只能看到 词0,1
  词2: [0.5, 1.1, 3.3, -inf]      词2 只能看到 词0,1,2
  词3: [1.0, 0.8, 2.5, 3.0]       词3 能看到所有

softmax 后,-inf 变 0,未来位置权重 = 0,完全不参与
```

这就是 L3.1 写的 `scaled_dot_product_attention` 里 `mask` 参数的用途。

### 怎么生成 mask

`torch.tril` = lower triangular(下三角)，保留对角线及以下:

```python
mask = torch.tril(torch.ones(4, 4))
# [[1, 0, 0, 0],
#  [1, 1, 0, 0],
#  [1, 1, 1, 0],
#  [1, 1, 1, 1]]
```

传给 attention，`mask==0` 的位置被置 -inf。

### Decoder-Only(GPT 结构)

原论文是 Encoder-Decoder(翻译任务，有 cross-attention)。但 **GPT 系列是 Decoder-Only**--只有 masked self-attention，没有 cross-attention:

```text
GPT 的 Block:
    输入 x
     ↓
 Masked Self-Attention + Add & Norm
     ↓
 Feed-Forward + Add & Norm
     ↓
    输出
```

比 Encoder Block 多一个 mask，比完整 Decoder 少一个 cross-attention。**GPT/BERT/Claude/Llama 全是这种结构的变体**。

DecoderBlock 和 EncoderBlock 唯一区别:attention 要传 mask。

## 七、完整 Decoder-Only Transformer

### d_model 是什么

**d_model = 模型的核心维度**，贯穿所有子层(词嵌入/PE/attention/FFN 输入输出都是 d_model)。所有子层保持 d_model 维不变--这是能堆叠、能加残差的前提。

| 模型 | d_model |
| --- | --- |
| 本实验(玩具) | 128 |
| 原论文 Transformer | 512 |
| BERT-base | 768 |
| GPT-3 | 12288 |

### 完整模型 forward

```python
def forward(self, idx):                       # idx: (batch, seq) 整数 token id
    x = self.token_embedding(idx)             # (batch, seq) -> (batch, seq, d_model)
    x = self.pos_encoding(x)                  # 加位置编码
    mask = torch.tril(torch.ones(seq, seq))   # causal mask
    mask = mask.unsqueeze(0).unsqueeze(0)     # (1, 1, seq, seq) 广播对齐
    for block in self.blocks:
        x = block(x, mask)                    # 逐层过 DecoderBlock
    x = self.norm_final(x)                    # 最后过 LayerNorm(GPT-2 风格)
    logits = self.lm_head(x)                  # 映射回词表 (batch, seq, vocab_size)
    return logits
```

**数据流**:整数 id -> embedding(向量) -> +PE -> N 层 block -> 输出层 -> logits(每位置对每字符的分数)。

### 为什么 mask 要 unsqueeze 两次

mask 原始 `(seq, seq)`，但 attention 里 scores 是 `(batch, num_heads, seq, seq)`。加两维变 `(1, 1, seq, seq)`，前面两个 1 自动复制对齐 batch 和 num_heads。

## 八、字符级语言模型训练

### 任务:预测下一个字符

CharDataset 把文本切成 (input, target) 对，target 是 input 右移一位:

```text
文本: "To be"
input:  [T, o,  , b]    -> 4 个字符
target: [o,  , b, e]    -> 右移一位
```

模型学的是"看到前面几个字符，预测下一个"。这就是 GPT 的训练方式(只是 GPT 用 token 不是字符)。

### 训练循环

和 L2.2 模板一样(标准 5 步)。loss 用 `F.cross_entropy`，把 (batch, seq, vocab_size) reshape 成 (batch*seq, vocab_size) 和 (batch*seq,) 算。

## 九、文本生成(自回归采样)

### 自回归 = 每次预测一个，拼回去，重复

```python
@torch.no_grad()
def generate(model, dataset, prompt, max_new_tokens=200, temperature=0.8):
    model.eval()
    device = next(model.parameters()).device
    idx = [dataset.stoi[c] for c in prompt if c in dataset.stoi]
    idx = torch.tensor([idx], dtype=torch.long).to(device)

    for _ in range(max_new_tokens):
        logits = model(idx)                       # (1, seq, vocab_size)
        last_logits = logits[:, -1, :]            # (1, vocab_size) 只取最后位置
        probs = F.softmax(last_logits / temperature, dim=-1)  # 分数->概率
        next_id = torch.multinomial(probs, num_samples=1)     # 按概率采样
        idx = torch.cat([idx, next_id], dim=1)    # 拼到末尾
    return "".join(dataset.itos[i.item()] for i in idx[0])
```

### 关键变量

- **`idx`**:输入的 token id 序列，整数，形状 (1, seq)。每次循环变长 1
- **`probs`**:下一个字符的概率分布，(1, vocab_size)，和为 1。由 logits 过 softmax 得到
- **`temperature`**:温度参数。`logits / T` 后 softmax:T 小概率尖(保守)，T 大概率平(多样)
- **`torch.multinomial(probs, num_samples=1)`**:按概率随机采样。像掷灌铅骰子--90% 概率选最大，10% 选别的

### 为什么用 multinomial 不用 argmax

- argmax 总选最大 -> 生成重复死板("eeeeee...")
- multinomial 按概率采样 -> 偶尔选别的 -> 文本多样自然
- **这就是 ChatGPT 同一 prompt 每次回答不一样的原因**

### 温度的作用

```text
probs = softmax(logits / T)
```

- T 小(0.5):概率尖，输出保守确定
- T 大(1.5):概率平，输出多样随机
- T=0:等价 argmax

## 十、实验结果分析

### 训练结果

loss 降到 0.0543--非常低，说明模型几乎**背下了**训练文本(~600 字符的莎士比亚片段)。

### 生成结果

```text
示例1 (T=0.7):
To be, or not to be, that is the question:    ← 完美复现开头
Whether tis nobler in the mind to mind to mind to to suto sufffe  ← 陷入重复

示例2 (T=1.0):
The heart-ache and the thousand natural shocks  ← 复现训练文本
That flesh is heir to: tis to: a co: cons tis tis mmmmation  ← 更乱
```

### 好的部分

- 学到字符级英语模式(大小写、空格、标点)
- 从 prompt 续写
- 复现训练文本片段

### 退化的部分(重复循环)

三个原因叠加:

1. **模型太小**(4 层，128 维，~50万参数)--容量有限
2. **数据太小**(~600 字符)--loss 0.0543 说明背下来了，没学到泛化语言规律
3. **采样退化**--小模型 + 温度采样容易陷入重复循环(degeneration)

**这是小模型 LM 的典型表现，不是代码问题**。真正的 GPT 有几十亿参数 + 几千亿 token，才不这样。

### 这个实验证明了什么

模型**架构完全正确**:词嵌入+PE、Masked Self-Attention、残差+LayerNorm+FFN、自回归生成。**这就是 GPT 的结构**。GPT-2/GPT-3/Claude/Llama 都是这个架构，只是层数更多、维度更大、数据更多。

**从零写了一个"玩具 GPT"**--麻雀虽小，五脏俱全。

## 十一、我的误解与纠正

| 我的理解 | 精确修正 |
| --- | --- |
| "LayerNorm 可以输出不同维度，BatchNorm 固定维度" | 两者输入输出维度一样(都是 batch, seq, d)。区别在**沿哪个轴算统计量**:BN 跨样本(横着)，LN 单样本内部(竖着)。LN 不依赖 batch |
| FFN 讲的太简单，没懂 | 升维(拉开特征)->GELU(非线性筛选)->降维(压缩回原维度)。attention 是"跨词搬运信息"，FFN 是"单词内部非线性变换提取特征" |
| dropout 应该加在哪 | 加在子层输出(attn_out/ffn_out)上，**不加在残差主干 x 上**。主干要保持干净让梯度走捷径 |
| self-attention 和普通 attention 区别不清 | 单个词做注意力=普通 Attention；句中每个词都当一次 Query=Self-Attention。Decoder 还有 Masked + Cross-Attention |
| 生成时 idx 在 CPU 报设备错 | 推理时新建的 tensor 要 `.to(device)`；用 `next(model.parameters()).device` 取模型设备 |
| max_len=block_size 生成时报 PE 长度错 | PE 的 max_len 要按"最长可能序列"设，不是按训练 block_size 设 |

## 十二、补充问答

### 1. 为什么 attention 需要位置编码

Transformer 不像 RNN 有序列关系，attention 公式里没有任何位置信息(只看 Q/K 内容)。所以 "猫追狗" 和 "狗追猫" 对 attention 来说输入向量集合一样，输出一样，但意思相反。必须人为加位置信息。

### 2. 残差连接解决了什么

深度太深时梯度传递到下层越来越小(梯度消失)，下层几乎学不到东西。残差 `out = x + Sublayer(x)` 让梯度有捷径直接传回去，不消失。

### 3. 为什么 GELU 不是 ReLU

GELU 比 ReLU 更平滑(在 0 附近是光滑曲线而非硬截断)，梯度更友好，Transformer/BERT/GPT 都用 GELU。ReLU 在 0 处不可导，GELU 可导。

### 4. nn.Module 为什么都有 forward

forward 是 nn.Module 的"契约"--定义"数据怎么流过这模块"。`__init__` 声明有什么零件，`forward` 定义零件怎么连。调用用 `model(x)` 不用 `model.forward(x)`--PyTorch 的 `__call__` 帮你包装 autograd。

### 5. 积木在哪里拼装

拼装在 `__init__`(买零件)和 `forward`(接线)里。EncoderBlock 拼 4 个积木，Transformer 拼 Embedding+PE+N 个 Block+输出层。详见 [PYTORCH_ARCHITECTURE.md](../../PYTORCH_ARCHITECTURE.md)。

## 十三、关键工程坑

### 1. 推理时 tensor 设备不匹配

新建 tensor 默认在 CPU，模型在 GPU，embedding 查表报错。修复:`idx = torch.tensor([idx]).to(next(model.parameters()).device)`。

### 2. 位置编码 max_len 不够

训练时 block_size=64 够用，生成时序列增长超过 64 报错。修复:PE 的 max_len 设大(如 512)，按"最长可能序列"设。

### 3. 数据集集中管理

所有数据集放 `dl-llm-course/data/`，用 `common.data_utils.get_dataset_dir()` 取路径，避免每个 lab 复制一份占空间。

## 十四、实验产物

- 积木验证代码:[lab/train.py](lab/train.py)(Encoder Block，形状检查通过)
- 完整训练代码:[lab/train_lm.py](lab/train_lm.py)(Decoder-Only Transformer，生成文本)
- 结果:loss 0.0543，生成莎士比亚风格文本(开头好，后面重复--小模型典型表现)
