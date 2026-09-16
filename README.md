# TAOBAO-MM · MUSE

基于 TAOBAO-MM 的多模态长历史兴趣建模复现，对应 [A Simple Yet Effective Multimodal Search-Based Framework for Lifelong User Interest Modeling](https://arxiv.org/html/2512.07216v1)。本项目为非官方实现，参考[官方代码](https://github.com/alimama-tech/MUSE)。

研究主线：**长历史处理成本 → 目标相关检索 → 语义增强兴趣建模 → 检索与融合的独立贡献**。

[实施路线](docs/roadmap.md) · [方法覆盖与适配边界](docs/reproduction-scope.md) · [前序内容融合项目](https://github.com/IJustWannaFindMyPeace/taobao-mm-content-integration)

## 当前状态

本仓库已建立实验范围与工程路线，R0/R1 检索对照与完整 R4 适配结构已实现，L=200/1000 的 20 步资源检查通过；R0–R5 的 L=200 单种子全轮验证已完成。前序项目已完成 1% 用户子集上的 C0–C7 核心适配配置及 MAKE 策略对照的单种子验证；多种子、低频分组与最终测试尚待完成。这些结果不代表 MUSE 模型通过验证。

## 方法与消融

GSU（General Search Unit，通用搜索单元）按目标相关性选择 Top-K 历史；ESU（Exact Search Unit，精细搜索单元）对选中历史建模。SA-TA（Semantic-Aware Target Attention，语义感知目标注意力）融合语义与 ID 信息，SimTier 提供相似度分布特征。

|编号|历史选择|长历史建模|检验目标|
|---|---|---|---|
|R0|最近 K 条|TA：目标注意力|提供近期历史对照|
|R1|内容余弦 Top-K|TA|检验目标相关检索|
|R2|内容余弦 Top-K|TA + SimTier|检验统计特征|
|R3|内容余弦 Top-K|SA-TA|检验语义增强注意力|
|R4|内容余弦 Top-K|SA-TA + SimTier|检验完整组合|
|R5|内容余弦 Top-K|SimTier-only|隔离统计分支贡献|

公共分支保持一致；R5 关闭被研究长历史的 ID 注意力及均值旁路。R0 不等于独立 DIN 基线。R0–R5 已实现并通过正确性检查，L=200 单种子效果对照与预定配对分析已完成。

补充对照包括 DIN、SIM-hard、SIM-soft，以及 ID/内容检索评分方案；TWIN、MISS 和可学习检索器需完成专项规格后实施。历史长度扫描同时比较 R1 与 R4，避免将更长历史的影响归因于完整模型。详细方法映射与受限项见[范围说明](docs/reproduction-scope.md)。

## 共享协议与资源

计划继承前序项目验证后的数据清单、mask 规则及标准评估实现，并通过预测一致性检查验证迁移。正式协议尚未冻结。AUC 与 GAUC 统一使用完整评估记录；官方兼容口径如保留，将单列报告。

目标设备为 RTX 5070 12 GB、32 GB RAM。固定内容向量、流式预处理和受控词表用于限制资源开销。B=32、L=200、K=50 为待测起点，显存与吞吐以实际模型测量为准。独立依赖锁和训练入口将在验证后发布。

## 证据与局限

计划报告核心消融、多种子波动、用户历史长度与商品频次分组，以及显存和吞吐。论文描述、官方代码差异和本地适配分别记录。

公开 1,000 长度历史上的实验不等于工业 100,000 长度链路复现。缺少额外表示和原始图文时，不宣称完成 OpenCLIP、I2I 与 SCL 的原文表示对照，也不推断线上收益。前序 MAKE 与 MUSE 的组合仅作为可选扩展。

## 数据来源与统计口径

官方数据定义、p90 覆盖率口径和指定版本的差异核验见[数据集契约](https://github.com/IJustWannaFindMyPeace/taobao-mm-content-integration/blob/main/docs/dataset-contract.md)。

## 检索实现与资源检查

共享 C0 迁移、显式 mask 检索和 L=200/1000 的资源检查见[阶段 1](docs/stage-01-search-foundation.md)。本阶段未产生 MUSE 效果指标。

SA-TA、可学习 SimTier 的公式、初始化差异与完整 R4 资源结果见[阶段 2](docs/stage-02-enhanced-esu.md)。

R0–R5 的固定配置、R5 路径关闭验证及串行训练入口见[阶段 3](docs/stage-03-core-study.md)。

首轮验证中 R2 的指标点估计最高；R4 对 R2、R1 对 R0 的 GAUC 增量区间跨零。结果与局限见阶段 3，不预设完整模型或内容检索必然胜出。

R1/R4 的 L=1000 全轮端点对照和配对分析已完成，尚未确认长度增加的稳定 GAUC 收益，控制变量及分析计划见[阶段 4](docs/stage-04-history-length.md)。

两项目共享的现有频次诊断见[频次报告](https://github.com/IJustWannaFindMyPeace/taobao-mm-content-integration/blob/main/docs/stage-15-frequency-and-budget.md)。5% 数据与资源核验见[共享报告](https://github.com/IJustWannaFindMyPeace/taobao-mm-content-integration/blob/main/docs/stage-16-scale-preparation.md)。该规模 R0–R5 全轮训练与保存预测审计已完成；R4 验证 GAUC 为 0.58922，但相对 R2 的增量区间跨零，结果与局限见[阶段 5](docs/stage-05-five-percent-core-study.md)。R1/R4 的历史长度 1000 全轮对照及审计也已完成，R4 GAUC 为 0.59231；两个长度增量区间均高于零，证据及多种子验证计划见[阶段 6](docs/stage-06-five-percent-history-length.md)。

固定数据的三个种子（42、2026、3407）核心对照与审计均已完成。R4 的 GAUC 均值为 0.589776，样本标准差为 0.000688；三个种子排名一致，但部分模块额外收益较小且用户配对区间仍跨零。均值、标准差、同种子增量及局限见[阶段 7](docs/stage-07-training-seed-validation.md)。长历史端点三种子验证已完成，详见阶段 6。

5% 长历史端点的 seed 2026 训练与审计已完成：R4 的长度增益区间高于零，R1 区间包含零且 LogLoss 恶化；seed 3407 对照也已完成。分种子证据见[阶段 6](docs/stage-06-five-percent-history-length.md)。

三个种子的 R4 长度增益均为正且各自用户配对区间高于零；R1 的 seed 2026 区间包含零。完整长度端点证据与解释边界见[阶段 6](docs/stage-06-five-percent-history-length.md)。

5% 数据上三个种子共 27 组预测的商品频次与内容覆盖率诊断已完成，分组差异与小样本局限见[共享分组分析](https://github.com/IJustWannaFindMyPeace/taobao-mm-content-integration/blob/main/docs/stage-21-frequency-and-coverage.md)。
