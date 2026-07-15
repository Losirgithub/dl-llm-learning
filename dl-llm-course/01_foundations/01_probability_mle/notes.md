# L1.1 概率与统计视角 · 笔记

> 每节课结束时由 tutor 生成，固化推导、公式、以及学员的误解与正确答案。

## 一、关键公式

### MLE 框架

$$
L(\theta) = \prod_{i=1}^{n} P(x_i \mid \theta) \quad \text{（各点概率连乘）}
$$

$$
\log L(\theta) = \sum_{i=1}^{n} \log P(x_i \mid \theta) \quad \text{（取 log 变连加，好求导）}
$$

### 正态分布（Gaussian）

$$
P(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

- $\mu$：钟形中心；$\sigma^2$：离散程度（$\sigma$ 大=矮胖分散，$\sigma$ 小=高瘦集中）

### 伯努利分布（二分类）

$$
P(y \mid x) = p^y \cdot (1-p)^{1-y}
$$

- $y=1 \Rightarrow P=p$；$y=0 \Rightarrow P=1-p$（一个式子统一两种情况）

## 二、核心推导

### 正态假设 → MSE

回归场景，假设误差 $\varepsilon = y - \hat{y}$ 服从 $\mathcal{N}(0, \sigma^2)$：

$$
P(y \mid x, w, b) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(y-\hat{y})^2}{2\sigma^2}\right)
$$

取 log：

$$
\log P = \underbrace{\log\frac{1}{\sqrt{2\pi\sigma^2}}}_{\text{常数 } C} - \frac{(y-\hat{y})^2}{2\sigma^2}
$$

$n$ 个数据点：

$$
\log L = nC - \frac{1}{2\sigma^2} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

最大化 $\log L$ $\Longleftrightarrow$ **最小化** $\sum(y_i - \hat{y}_i)^2$ = **MSE**。

### 伯努利假设 → 交叉熵

二分类，$p$ = 预测正类概率：

$$
\log P = y \cdot \log p + (1-y) \cdot \log(1-p)
$$

$n$ 个数据点：

$$
\log L = \sum_{i=1}^{n} \left[ y_i \log p_i + (1-y_i) \log(1-p_i) \right]
$$

最大化 $\log L$ $\Longleftrightarrow$ 最小化其负值 = **二元交叉熵损失(BCE)**。

## 三、核心 insight：统一框架

| 场景 | 概率假设 | MLE 推出的损失 |
| --- | --- | --- |
| 回归 | 误差 $\varepsilon \sim \mathcal{N}(0, \sigma^2)$ | MSE |
| 二分类 | 服从伯努利 $\text{Bern}(p)$ | 交叉熵 |

同一个 MLE 框架，换分布假设 → 换损失函数。**损失函数是概率假设的必然结果，不是拍脑袋定的。** 遇到新任务，先想"数据/误差服从什么分布"，对应损失自然出来。

**变量 = 要优化的参数**：回归里是 $w, b$；$\sigma$ 是数据噪声的固有属性，当常数（不训练它）。

## 四、我的误解与纠正（个性化）

| 我的误解 | 正确 |
| --- | --- |
| MLE 是"找极大值" | MLE 是找让**似然**（观测数据出现的概率）最大的**参数** —— 漏了"参数"和"似然"两个关键词 |
| 推导出 RMSE（均方根误差） | 推导出 **MSE**（均方误差）；RMSE = $\sqrt{\text{MSE}}$ 是评估指标，不是推导出的损失 |
| "数据"服从正态分布 | 是"预测**误差** $\varepsilon = y - \hat{y}$"服从正态，不是数据本身；$y$ 可以任意分布 |
