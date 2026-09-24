# TAOBAO-MM · MUSE

**最新结果：**主实验54份模型及补充73份检查点的测试评估已完成，追加93项统计比较全部归档。先读[结果与故事线](docs/supplementary-test-results.md)和[当前覆盖范围](docs/current-reproduction-status.md)。追加测试单独标注，不混作新的盲测。


[当前复现状态与两项目故事线](docs/current-reproduction-status.md)汇总已完成证据、适配边界及剩余工作。


基于 TAOBAO-MM 固定内容向量，研究内容如何用于长历史选择和兴趣聚合。对应 [A Simple Yet Effective Multimodal Search-Based Framework for Lifelong User Interest Modeling](https://arxiv.org/html/2512.07216v1)，为非官方适配实现，参考[官方代码](https://github.com/alimama-tech/MUSE)。

**研究主线：长历史处理成本 → 目标相关检索 → 统计与语义聚合 → 检索复杂度和历史长度的受控验证。**

## 阅读入口

- [最新论文覆盖审计：已做、适配与可补缺口](docs/paper-coverage-audit-2026-09-23.md)

- [两项目最终故事线](https://github.com/IJustWannaFindMyPeace/taobao-mm-content-integration/blob/main/docs/final-storyline.md)
- [本项目最终测试结果](docs/final-test-results.md)
- [等权内容＋ID 检索的设计与验证证据](docs/shared-fused-retrieval-v1.md)
- [方法覆盖与适配边界](docs/reproduction-scope.md)
- [前序内容融合项目](https://github.com/IJustWannaFindMyPeace/taobao-mm-content-integration)

## 后续补充研究

[G3 融合检索结果与诊断](docs/post-main-g3-results.md)已归档：未显示稳定 GAUC 优势，保留明显种子差异。[G1 三种子结果](docs/post-main-g1-results.md)已归档：学习后冻结的内容检索在三个种子上均低于原始内容余弦对照。

已补充 [8/9 组事后诊断](docs/posthoc-paper-groups-v1.md)。商品分组保留同频并列限制；用户组大多同为截断长度 1000，不能据此判断真实长度趋势。

第一轮冻结实验完成后，已补[测试集低频分组](docs/posthoc-test-frequency-v1.md)，明确标为事后描述性分析。MAKE 训练策略已补齐三个种子的验证证据；DIN/TWIN/MISS 的本地适配补充已完成，共 21 次验证训练和 12 项配对比较；原始工业完整模型仍不在覆盖范围。见[统一收尾说明](docs/study-completion.md)。

## 完成状态与结论

两项目共同的 54 个模型、三个训练种子、66 项冻结比较已全部完成保留测试评估。每组完整评估 923,211 条曝光；测试前验证预测逐元素回放一致，未利用测试结果调参。

统计特征、单独加入语义注意力、两种聚合方式下增加历史长度，以及 L1000 内容检索相对类目／ID 检索的 GAUC 区间，均在三个种子为正。已有统计后再加语义注意力的增量区间仍均跨零。内容＋商品 ID 等权检索反而弱于纯内容检索，不能据此推断所有复杂检索无用。

|L1000 检索方式，固定完整聚合|三种子平均测试 GAUC|
|---|---:|
|内容余弦|0.595488|
|类目检索|0.589523|
|商品与类目 ID 余弦|0.590532|
|商品 ID 余弦|0.587225|
|内容＋商品 ID 等权余弦|0.590843|

全部十二种 MUSE 配置的 AUC、GAUC、LogLoss、波动和区间见[最终结果](docs/final-test-results.md)。不能把完整模型视为每个消融比较的必然赢家。

## 方法与共同协议

GSU（General Search Unit，通用搜索单元）选出与目标有关的 Top-K 历史；ESU（Exact Search Unit，精细搜索单元）聚合选中行为。SimTier 提供相似度统计，SA-TA（Semantic-Aware Target Attention，语义感知目标注意力）融合内容与 ID 注意力分数。

共同 5% 用户队列，固定内容向量，下游 batch=256、单轮、两类学习率 0.001，种子 42/2026/3407。K=50；主比较覆盖 L200 与 L1000。近期历史对照不等于原始 DIN，类目／ID 检索使用统一增强聚合，是明确的适配基线。

## 代码与产物

- 检索与聚合：`src/retrieval.py`、`src/baseline_retrieval.py`、`src/fused_retrieval.py`、`src/enhanced_esu.py`。
- 模型与训练：`src/muse_model.py`、`src/baseline_model.py`、`scripts/train_muse.py`。
- [冻结清单](reports/final-test-freeze-v1.json)、[逐模型测试指标](reports/final-test-model-metrics.csv)、[66 项配对结果](reports/final-test-paired-comparisons.csv)、[完整汇总](reports/final-results-summary.json)。

单卡 RTX 5070 12 GB、32 GB RAM 完成此范围。共同只读测试入口位于前序仓库，具体用法和准入条件见[评估协议](https://github.com/IJustWannaFindMyPeace/taobao-mm-content-integration/blob/main/docs/final-evaluation-protocol-v1.md)。[实施路线](docs/roadmap.md)保留分阶段配置与产物，早期探索结果不替代最终主表。

## 复现边界

主实验后的 [DIN 局部激活机制补充](docs/post-main-din-results.md)已完成三种子验证，GAUC 配对区间均跨零，未观察到可靠改善。[TWIN 四头适配结果](docs/post-main-twin-results.md)中，内容检索在三个种子的配对比较中均更优，但去重历史覆盖数不同。[MISS 双路适配结果](docs/post-main-miss-results.md)已完成三种子验证，未显示稳定 GAUC 优势，共享参数也无跨种子一致收益。[来源核对](docs/post-main-retrieval-source-audit.md)记录了预算和参数共享歧义及本地适配边界。

5% 队列、L1000 不是全量数据和工业十万历史复现。SCL 编码器重训、OpenCLIP/I2I 对照、原始 DIN/TWIN/MISS、其他可学习检索器和线上收益未覆盖。前序 MAKE 不作为 MUSE 默认前置模块。保留跨零、负向和不同指标冲突的结果。

[统一适配基线比较](docs/adapted-baseline-comparison.md)与[固定抽样匿名检索案例](docs/anonymous-retrieval-cases.md)已归档，分别说明整体验证证据与具体选择行为。

[历史长度三种子结果](docs/post-main-length-seeds-results.md)已完成，保持 K=50 与共同下游协议。

[本轮补充工作收尾核验](docs/reproduction-closeout.md)：96项产物一致性检查通过；完成范围与原论文限制分别说明。

[两论文结果阅读指南](docs/storyline-reading-guide.md)：按问题、方法、证据与局限理解完整迭代路径，无需先记实验代号。

[统一追加测试结果](docs/supplementary-test-results.md)已归档：73份验证重放、73份测试评估与93项比较全部完成。原验证报告保留时点记录，最新结论单独发布。
