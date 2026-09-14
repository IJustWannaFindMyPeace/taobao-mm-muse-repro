# 阶段 5：5% 数据上的核心消融

## 共同实验基础

本轮与内容融合项目使用同一份 5% 特征存储、训练/验证划分和固定内容向量。训练曝光 3,052,475 条，验证曝光 768,773 条；不对保留测试集进行模型评估。

存储 manifest SHA256：`7dd761c48f61b3ae219922f8387f6026eb6eb011ef6ef8050bb1472a52593f59`。准备与资源核验见[共享数据报告](https://github.com/IJustWannaFindMyPeace/taobao-mm-content-integration/blob/main/docs/stage-16-scale-preparation.md)。

模型保持阶段 3 的实现适配：历史长度 200、检索 K=50、短历史 50、embedding 宽度 16、batch size 32、seed 42，每组完整训练一轮。两类优化器学习率均为 0.001，weight decay 为 0。

## 痛点与消融关系

|配置|检索及长历史处理|主要比较|
|---|---|---|
|R0|最近行为 + ID 注意力|检索参照|
|R1|内容余弦 Top-K + ID 注意力|R1−R0：目标相关筛选|
|R2|R1 + SimTier|R2−R1：兴趣分布统计|
|R3|R1 + SA-TA|R3−R1：语义与 ID 注意力融合|
|R4|R1 + SimTier + SA-TA|R4−R2、R4−R3：组合的额外贡献|
|R5|内容 Top-K + SimTier，关闭长历史 ID 聚合|R2−R5：长历史 ID 聚合的额外贡献|

GSU（General Search Unit）负责从完整历史中筛选行为，ESU（Exact Search Unit）负责对选中行为进行精细建模。SA-TA（Semantic-Aware Target Attention）为语义感知目标注意力。R5 仍保留用户/目标 ID 和短历史 ID 分支，不是全模型的内容单模态对照。所有模块的具体公式与掩码适配沿用阶段 2、3。

CIKM 的 C0 与本项目 R0 并非只差一个检索开关；两个骨干的历史处理与 V 投影存在差异。因此跨项目数值可展示共同数据上的系统表现，但检索和模块贡献必须用本表内部的控制变量比较。两个项目的 SimTier 实现也分别保留其定义，不作同名等价处理。

## 运行与状态

```powershell
python scripts/run_core_study.py --store <store-050> --output-dir <fresh-directory>
```

入口按 R0–R5 顺序使用单卡运行，每组保存独立训练日志、配置与代码摘要、验证预测、模型检查点和指标报告。任一组失败即停止后续运行。

六组队列已启动，尚无完整 5% 对照结果。初步按前一项目完整运行估计 4–6 小时，此为排期估算；首组完成后更新。后续审计需要验证完整曝光对齐、重新计算指标，并报告按用户配对的 GAUC bootstrap 区间。单种子区间不反映训练随机性，也不能代替多种子实验。

本轮先固定长度 200 做核心模块消融，长度 1000 的资源探测不能当作该长度的正式效果结果。长历史端点、其他种子及补充基线仍待后续运行。
