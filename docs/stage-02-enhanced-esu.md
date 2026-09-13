# 增强 ESU：SA-TA 与可学习 SimTier

## 方法来源与适配

[论文 §4.3，式 4–5](https://arxiv.org/html/2512.07216v1#S4.SS3) 在 softmax 前融合 ID 注意力分数 `d` 与检索结果的内容余弦 `r`：

```text
fused = gamma_1 * d + gamma_2 * r + gamma_3 * d * r
interest = masked_softmax(fused) @ V(history_ID)
```

三个 gamma 可学习，初始化为 `[1, 0, 0]`。本实现采用论文的缩放点积分数与融合公式，未采用官方代码的额外温度参数及 SiLU 变换。Q/K 继续使用公共主干的有偏置线性层；V 为无偏置、初始恒等的可训练线性层。

在所有后续 R0–R4 对照中共同使用该 V 投影，只有语义融合和统计分支开关不同。阶段 1 的 `SearchModel` 保留为迁移检查参考，其未训练 V；后续效果对照统一使用 `MUSEModel`，不能直接混用两套模型的训练结果。

## 可学习 SimTier

参考[官方 SimTier 实现](https://github.com/alimama-tech/MUSE/blob/main/model/base_model/simtier.py) 的 eps=0.1 布局：

```text
bin = ceil(clamp(r, -1, 1) * 10) + 10
feature = log1p(valid_count_per_bin) * learned_scale
```

保留 22 维布局；归一化余弦实际落在 0–20，末桶为空。这与前序 CIKM 的 20 桶 floor/clamp 原始计数明确区分。计数仅包含有内容的有效配对，padding 和缺失内容不计入零相似度桶。

scale 初始化为 1，区别于官方的随机均匀初始化。统计特征通过零初始化线性投影接入公共 MLP 第一层，因此初始预测与关闭统计分支一致。第一步统计 scale 梯度为零；投影更新后可正常获得梯度。此设计服务于受控实验，属于显式初始化适配。

## 数据流与边界

ESU 语义分数从 GSU 选中的同一批历史重新计算，ID、内容和 mask 使用同一位置索引。短历史分支保持不变，统计分支仅处理被研究的长历史 Top-K；未添加官方模型的短历史 SimTier、均值旁路或辅助损失。

padding 在分数融合之后屏蔽，防止可学习负系数将 padding 变成高分。全空历史产生零兴趣。目标内容缺失时，GSU 沿用最近有效历史回退，ESU 的语义项和统计计数为零，仍可通过 ID 路径预测。

## 测试与资源

14 项测试通过，覆盖融合公式、负系数与 padding、全空历史、分桶端点、缺失计数、共同初始预测、V/gamma/统计参数梯度以及非默认参数检查点重载。

```powershell
python -m unittest discover -s tests
python scripts/probe_search.py --store <shared-feature-store> --c0-run <original-c0-run> --output-dir <fresh-r4-probe> --enhanced
```

完整 R4 结构在现有 1% 词表上通过两组 20 步检查，每组训练 640 条记录：

|L|K|Batch|峰值分配显存 MiB|峰值保留显存 MiB|
|---:|---:|---:|---:|---:|
|200|50|32|2835.99|2858|
|1000|50|32|2874.79|2886|

两组损失和梯度均有限，参数更新检查通过。完整机器结果见[资源报告](../reports/r4-resource-probe.json)。耗时包含冷启动、CPU 内容读取及缓存差异，不作为稳定吞吐估计；显存只统计 PyTorch 分配，不包括全部进程与 CPU 内存。

## 当前结论与下一步

当前硬件可以启动本项目完整 R4 适配结构的训练，但尚未完成全轮训练、效果消融或最大数据规模验证。R5 的长历史 ID 路径关闭控制尚待实现。下一步补齐统一训练入口和 R0–R5 对照，再决定是否扩大共享用户子集；测试集继续保留。
