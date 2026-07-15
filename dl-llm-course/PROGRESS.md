# 学习进度

> 每次会话开始先读此文件；每课结束更新。

## 当前位置

- 阶段：Stage 1 - 深度学习数学基础
- 阶段：Stage 2 - 神经网络与训练
- 正在学：L2.2 已完成
- 下一步：L2.3 CNN（在 CIFAR-10 上训练 ResNet，正式用到 GPU 加速）

## 掌握检查点台账

| 课程 | 状态 | 掌握度 | 备注/难点 |
| --- | --- | --- | --- |
| L0.1 环境搭建 | ✅ 完成 | 熟练 | 3070(sm_86)✅ 5070Ti(sm_120)✅ 两机GPU就绪 |
| L0.2 项目脚手架与工程规范 | ✅ 完成 | 熟练 | 脚手架+ruff+set_seed 可复现 已验证 |
| L0.3 双机与局域网协同 | 🔶 进行中 | 部分 | SSH远程开发✅(今日实操)；DDP延后到有训练脚本 |
| L1.1 概率与统计视角 | ✅ 完成 | 部分 | MLE推MSE/交叉熵掌握；MLE定义/MSEvsRMSE/误差vs数据 3处概念需精确 |
| L1.2 信息论 | ✅ 完成 | 熟练 | 熵/交叉熵/KL全通过；混淆过熵与自信息(已纠正) |
| L1.3 数值优化 | ✅ 完成 | 熟练 | 手写四种优化器实验通过；三题全对；Adam计算过程写详细在笔记 |
| L2.1 PyTorch 张量与自动微分 | ✅ 完成 | 熟练 | autograd 版与手写 Adam 结果完全一致；概念(数值vs符号/前向vs反向)已澄清 |
| L2.2 MLP 与训练循环 | ✅ 完成 | 熟练 | 亲手实现4个TODO；MNIST达97.76%峰值；观察到过拟合萌芽并理解 |

状态图例：⬜ 未开始 / 🔶 进行中 / ✅ 完成 / 🔁 待复习

## 待复习项（薄弱点）

- L0.1 核心知识点 1: 判断 CPU/GPU 版 torch:看版本后缀,`+cu1xx`=GPU 版,`+cpu`=CPU 版。
- L0.1 核心知识点 2: PyTorch 版本包含了它支持的 GPU 算力列表,新显卡需要用新 PyTorch 版本。

## 错题本 / 重复问题（每次犯错更新；次数≥2 必须升级处理）

> tutor 每节课第 0 步先读此区；命中同类问题要提醒"第 N 次"并升级。

| # | 问题 | 归类 | 出现次数 | 根因 | 正确做法 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | check_env.py 报"没检测到 PyTorch" | 工程性 | 1 | 跑脚本时激活了错误的 conda 环境（torch 在另一个环境） | 跑任何脚本前先确认环境：命令行前缀括号名 / `conda info --envs` 看 `*` | 已克服 |
| 2 | 用清华源 pip install torch 装成 CPU 版 | 工程性 | 1 | 清华源的 torch 是 CPU 版，不含 CUDA | 装 PyTorch 必须用 PyTorch 官方 CUDA 源或上交镜像：`--index-url https://download.pytorch.org/whl/cu130` | 观察中 |
| 3 | 假设静默安装成功直接用，报"找不到 conda-hook.ps1" | 工程性 | 1 | 静默安装可能失败却没验证，且新会话变量丢失 | 任何静默/远程安装后必须 `Test-Path`/`ls` 验证产物存在；变量不跨会话保留，要重设或写绝对路径 | 已克服 |
| 4 | 运行 .ps1 脚本报"禁止运行脚本(ExecutionPolicy)" | 工程性 | 1 | Windows PowerShell 默认禁止运行未签名脚本 | `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`（无需管理员） | 已克服 |
| 5 | pip 报"File contains no section headers"，行首出现 `锘縖` 乱码 | 工程性 | 1 | PowerShell `Out-File -Encoding utf8` 写入带 BOM，pip 解析 ini 不认 BOM | 写配置文件用无 BOM 编码：`[System.IO.File]::WriteAllText($path, $content, [System.Text.UTF8Encoding]::new($false))` | 观察中 |
| 6 | MLE 定义说成"找极大值"；MSE 说成"均方根误差(RMSE)"；说"数据服从正态" | 概念性 | 1 | 漏关键词；混淆损失函数与评估指标；误把数据当误差 | MLE=找让似然最大的参数；推导出 MSE(非 RMSE)；是预测误差 ε=y-ŷ 服从正态，非数据本身 | 观察中 |
| 7 | Windows conda 环境 numpy+matplotlib 报 `OMP Error #15` 崩溃 | 工程性 | 1 | numpy 和 matplotlib 各自带一份 Intel OpenMP dll，加载冲突 | 在**导入 matplotlib 之前**设 `os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"`；`matplotlib.use()` 来得太晚 | 观察中 |

归类：概念性 / 工程性 / 习惯性　状态：观察中 / 反复出现 / 已克服

## 前置能力摸底结果（首次启动填）

- 概率统计：待摸底（Stage 1 再考察）
- 线性代数：待摸底（Stage 1 再考察）
- Python 工程：良好（conda、SSH、排错能力强，环境配置中已体现）

## 会话日志

- 2026-07-14：首次启动。创建课程脚手架，进入 L0.1 环境搭建。
- 2026-07-14：3070 环境自检通过（torch 2.9.0+cu130，sm_86）。5070Ti 配 Miniconda+conda 环境，PyTorch 下载中。L0.2 脚手架+ruff+set_seed 可复现验证通过。
