# DIN mechanism extension: fixed validation protocol

> 文档性质：阶段记录／历史协议。本文中的规模、配置、验证结果及待办状态属于记录时点；当前结果以[主测试](final-test-results.md)、[补充测试](supplementary-test-results.md)和[覆盖说明](current-reproduction-status.md)为准。

This post-main-study extension tests whether the recent-history attention mechanism affects the common-backbone baseline. The original 54-model test study remains frozen. These additional runs use validation only; their design was specified after the original test results were available.

## Architecture and comparison

The primary comparison is **DIN local activation versus dot-product softmax**, using the same most recent 50 history positions. Both models share the study's ten categorical fields, 16-dimensional ID embeddings, target item/category representation, prediction tower, data order, and optimizer settings. The unused long-interest input to the common prediction tower is zero.

The DIN adaptation concatenates target, history, their difference, and their elementwise product, then applies a 80–40–1 local activation network with Dice hidden activations. Its scalar outputs weight history vectors **without softmax normalization**, preserving interest intensity. Padding is excluded before the activation network and its normalization statistics. A singleton valid behavior uses stored normalization statistics. The dot-product control uses the existing normalized short-attention module.

This is a paper-informed DIN mechanism on the shared experimental backbone. It is not the complete original industrial DIN configuration: the prediction tower and optimizer follow this study, and mini-batch-aware embedding regularization is not implemented. The interaction features and hidden widths are explicit local architecture choices. Consequently, this comparison isolates the attention-module package, not each internal change separately.

## Fixed runs and checks

| Setting | Value |
|---|---|
| Models | DIN local activation; short-only dot-product softmax |
| Seeds | 42, 2026, 3407 |
| Cohort | Existing 5% user cohort, unchanged splits |
| Training | One full epoch; batch 256 |
| Optimizers | SparseAdam and AdamW, both learning rate 0.001 |
| Selection | No validation-based hyperparameter sweep |
| Evaluation | Complete validation predictions; AUC, GAUC, LogLoss |
| Primary contrast | DIN minus dot-product control, paired within seed |

A 100-step training probe checks the fixed batch size, finite gradients, parameter updates, checkpoint reload, and resource limits before full runs. Its validation scores are not effectiveness evidence. Full runs must process 3,052,475 training rows and 768,773 validation rows. Original test predictions and frozen source files are not modified.

Tests check non-normalized interest intensity, padding invariance including normalization statistics, empty and singleton histories, shared initialization, exclusion of older history, gradient flow, and checkpoint round trips. Training reports record configuration, source hashes, feature-store identity, resource measurements, and predictions.

## Interpretation

Comparing this DIN adaptation with a complete MUSE model changes retrieval, history coverage, and aggregation together; that is an overall model comparison, not a single-factor ablation. The paired short-only control is required to interpret the effect of the local activation mechanism. TWIN and MISS remain separate extension work and are not represented by either short-history model.

Reference: [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/html/1706.06978v4).
