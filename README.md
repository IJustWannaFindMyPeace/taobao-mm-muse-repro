# TAOBAO-MM上的MUSE长历史机制复现

状态：规划中，未完成环境与训练。非官方复现项目，没有已验证运行命令或成绩。

论文：[MUSE: A Simple Yet Effective Multimodal Search-Based Framework for Lifelong User Interest Modeling](https://arxiv.org/html/2512.07216v1)。参考：[官方实现](https://github.com/alimama-tech/MUSE)。

## 与前一项目的关系

CIKM2024项目研究内容怎样融合进推荐；本项目研究内容怎样帮助选择、使用长历史。共用冻结数据与评估协议，不默认继承MAKE。建议仓库名taobao-mm-muse-repro，远程仓库尚未创建。

## 方法

GSU（General Search Unit，通用搜索单元）用固定内容余弦选择Top-K；ESU（Exact Search Unit，精细搜索单元）用SA-TA（Semantic-Aware Target Attention，语义感知目标注意力）与SimTier统计生成特征。

## 实验计划

R0最近K+TA；R1内容TopK+TA；R2内容TopK+TA+SimTier；R3内容TopK+SA-TA；R4内容TopK+SA-TA+SimTier；R5内容TopK+SimTier-only（关闭研究长历史ID旁路）。公共分支一致。另列B0短历史DIN、SIM-hard/soft对照，不能把R0叫完整DIN。L扫描对照R1/R4；K扫描与低频/长度分组补充诊断。

G0内容余弦、G-ID ID余弦、G1多模态注意力、G2内容余弦+ID余弦、G3内容余弦+ID注意力已纳入路线；G1/G3训练梯度规格未完成，不能直接运行随机检索器作证据。TWIN/MISS为条件项，MAKE组合为新增扩展。OpenCLIP/I2I及原文表示敏感性对照因数据受限暂不运行。所有结果未运行。详见[覆盖与开工审计](论文方案覆盖与开工审计.md)。

## 共享协议和适配

保留官方数据测试边界、共享子集清单/词表/内容版本、完整用户GAUC与统一mask。官方batch内GAUC、并列处理与本项目标准口径分别报告。修正实现需明确版本，不静默混入模型收益。公共代码从CIKM冻结版本带来源哈希迁入，两个项目独立可运行。

## 环境与资源

目标RTX5070 12GB、32GB主存。固定内容向量、小词表、流式预处理；首跑B32/L200/K50为待验证起点。不得承诺官方全量配置直接装入12GB。共享环境版本，项目独立隔离；锁文件及真实命令待验证后添加。

## 局限与验收

公开最长1k的实验不能称10万历史复现。没有原始图文和私有配对，不重现SCL/OpenCLIP/I2I预训练比较或线上收益。验收为可复现运行、核心消融、多种子波动、资源记录和来源/适配清单。

## 2026-09-11 实操进度更新

本地Git骨架已建立。工作区Windows CUDA bootstrap训练检查已通过，实际Python3.11.15/PyTorch2.12.1+cu132；暂复用既有环境，最终独立依赖锁未完成。CIKM目录scripts/smoke_train.py是合成数据诊断，不是C0或论文结果。官方demo键关联不完整，真实训练等待提取一致子集。前文规划状态中的“环境未验证”已被本节更新：仅bootstrap验证完成，正式训练尚未完成。远程仓库未创建。

## 分阶段学习

阅读[学习路线](docs/learning-path.md)。每阶段展示问题、关键代码、验证过的命令、产物与检查问题，通过Git提交记录进展。当前仅完成环境合成训练诊断与官方demo读取，真实数据基线尚未开始。
