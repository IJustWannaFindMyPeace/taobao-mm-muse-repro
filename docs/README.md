# 项目文档

建议先阅读[项目概览](../README.md)，再按方法、数据协议和测试结果查阅。历史阶段记录保留当时的配置与结论，不作为当前完成状态。

## 项目概览与方法

- [SimTier 统计布局与检查点兼容性](simtier-layout.md)

- [从多模态表示到长历史兴趣：研究叙事](research-narrative.md)
- [从多模态表示到长历史兴趣：结果阅读指南](storyline-reading-guide.md)
- [论文术语与实现命名](paper-terminology.md)
- [方法来源、实现差异与覆盖范围](reproduction-scope.md)
- [两篇论文对照与结果差异分析](paper-comparison-and-analysis.md)

## 数据与评估协议

- [TAOBAO-MM 字段字典与模型输入](dataset-fields.md)
- [正式用户抽样与评估划分](sampling-and-splits.md)
- [统一追加测试评估：范围与门槛](supplementary-test-protocol.md)
- [8/9 组补充诊断协议](paper-groups-protocol.md)

## 测试结果与诊断

- [L1000 ESU 四组消融：2026-10-16 补充实验](esu-l1000-completion.md)

- [最终保留测试集结果](final-test-results.md)
- [适配基线：统一测试集比较](adapted-baseline-test-comparison.md)
- [统一追加测试：结果与故事线](supplementary-test-results.md)
- [追加测试：完整配对结果](supplementary-test-comparisons.md)
- [测试集商品频次分组：主实验后的补充分析](posthoc-test-frequency-v1.md)
- [8/9 组事后测试诊断](posthoc-paper-groups-v1.md)
- [匿名检索案例：同一历史，三种评分](anonymous-retrieval-cases.md)
- [故事线图表与原始数值](storyline-figures.md)

## 覆盖状态与实施路线

- [实验覆盖与评估状态](current-reproduction-status.md)
- [实施路线](roadmap.md)

## 审计与历史记录

- [仓库一致性检查 · 2026-10-11](repository-audit-2026-10-11.md)

<details>
<summary>展开协议演进、阶段实验与审计记录</summary>

以下记录用于追踪实验过程。带日期的审计和展示计划只说明对应版本，不代表最新运行环境或在线演示稿状态。

- [两项目整体收尾审计](cross-project-completion-audit.md)
- [研究表达审计](research-expression-audit.md)
- [追加测试前：设计与执行审计](supplementary-pretest-audit.md)
- [适配基线：统一验证集比较](adapted-baseline-comparison.md)
- [批大小与超参数选择：扩容前必做检查](batch-and-hyperparameter-selection.md)
- [研究叙事与展示就绪审计](narrative-readiness-audit.md)
- [论文覆盖审计：2026-09-23](paper-coverage-audit-2026-09-23.md)
- [主实验完成后的基线补充阶段](post-main-baseline-extension.md)
- [DIN mechanism extension: fixed validation protocol](post-main-din-protocol.md)
- [DIN 局部激活机制：三种子验证结果](post-main-din-results.md)
- [G1 多模态可学习评分：本地实现规格](post-main-g1-protocol.md)
- [G1 可学习多模态检索：三种子结果](post-main-g1-results.md)
- [G3：语义相似度与 ID 注意力融合检索](post-main-g3-protocol.md)
- [G3 融合注意力检索：结果与行为诊断](post-main-g3-results.md)
- [历史长度曲线：三种子扩展协议](post-main-length-seeds-protocol.md)
- [历史长度：三种子四点曲线](post-main-length-seeds-results.md)
- [MISS 双路 CTR 适配：固定协议](post-main-miss-protocol.md)
- [MISS 双路 CTR 适配：三种子结果](post-main-miss-results.md)
- [TWIN 与 MISS：来源歧义和实施约束](post-main-retrieval-source-audit.md)
- [四头一致性检索：固定补充协议](post-main-twin-protocol.md)
- [四头一致性检索补充结果](post-main-twin-results.md)
- [展示计划与材料就绪情况](presentation-plan.md)
- [本轮复现补充：收尾核验](reproduction-closeout.md)
- [共同协议下的核心对照矩阵](shared-core-matrix-v1.md)
- [共同协议核心结果：三个种子的证据](shared-core-three-seed-results.md)
- [多模态余弦与多模态表示＋ID 余弦检索对照](shared-fused-retrieval-v1.md)
- [历史长度敏感性：固定检索数量](shared-length-sensitivity-v1.md)
- [共同协议下的长历史与检索策略对照](shared-retrieval-matrix-v1.md)
- [长历史与检索策略：共同协议下的三种子结果](shared-retrieval-three-seed-results.md)
- [共同训练方案：限定范围的学习率确认](shared-training-protocol-check.md)
- [共享主干迁移与 GSU 检索检查](stage-01-search-foundation.md)
- [增强 ESU：SA-TA 与可学习 SimTier](stage-02-enhanced-esu.md)
- [R0–R5 核心效果对照](stage-03-core-study.md)
- [历史长度端点对照：L=200 与 L=1000](stage-04-history-length.md)
- [阶段 5：5% 数据上的核心消融](stage-05-five-percent-core-study.md)
- [阶段 6：5% 数据上的历史长度端点](stage-06-five-percent-history-length.md)
- [阶段 7：固定数据上的训练随机性验证](stage-07-training-seed-validation.md)
- [阶段 8：独立基线的检索契约与规模约束](stage-08-baseline-contracts.md)
- [阶段 9：基线整模型接入与资源准入](stage-09-baseline-resource-gate.md)
- [阶段 10：固定曝光预算的批大小资源扫描](stage-10-batch-resource-scan.md)
- [阶段 11：基于实际显存余量的资源复测](stage-11-device-headroom.md)
- [阶段 12：批大小与学习率的训练质量筛选](stage-12-batch-quality-screen.md)
- [阶段 13：质量筛选审计与跨种子/跨模型确认](stage-13-quality-audit-and-confirmation.md)
- [阶段 14：多模态检索相对类目与 ID 检索的证据](stage-14-retrieval-baseline-evidence.md)
- [两项目故事线审计 · 2026-09-22](storyline-audit-2026-09-22.md)
- [两项目研究收尾：故事线与证据边界](study-completion.md)

</details>
