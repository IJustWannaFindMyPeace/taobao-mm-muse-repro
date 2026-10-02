"""Publish a descriptive common-test table without mixing validation metrics."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, required=True)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    previous = json.loads((root / 'reports/adapted-baseline-comparison.json').read_text())
    definitions = [
        ('supplement', 'post-main-short-interest-v1--din_lau-seed{}'),
        ('supplement', 'post-main-short-interest-v1--dot_softmax-seed{}'),
        ('main', 'b1-hard-enhanced-seed{}'),
        ('main', 'b2-soft-enhanced-seed{}'),
        ('supplement', 'post-main-twin-extension-v1--consistent-seed{}'),
        ('supplement', 'post-main-twin-extension-v1--content-seed{}'),
        ('supplement', 'post-main-miss-extension-v1--hybrid_separate-seed{}'),
        ('supplement', 'post-main-miss-extension-v1--content_separate-seed{}'),
        ('main', 'r4-l1000-seed{}'),
    ]
    rows = []
    for old, (phase, template) in zip(previous['models'], definitions):
        entries = []
        for seed in (42, 2026, 3407):
            key = template.format(seed)
            directory = a.work / ('final-heldout-test-v1' if phase == 'main' else 'supplementary-heldout-test-v1/models')
            path = directory / key / 'report.json'
            report = json.loads(path.read_text())
            assert report['test_evaluated'] and report['rows'] == 923211
            entries.append(dict(model_id=key, phase=phase, seed=seed, metrics=report['metrics'],
                                report_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        rows.append(dict(label=old['label'], budget=old['budget'], reports=entries,
                         mean_metrics={m: mean(e['metrics'][m] for e in entries) for m in ('auc', 'gauc', 'logloss')}))
    result = dict(scope='descriptive_adapted_baselines_common_test', test_evaluated=True, models=rows,
                  limitations=['Main frozen tests and later supplemental tests are distinguished by phase.',
                               'Different architectures and history budgets; not a causal ranking or equal-compute benchmark.',
                               'Three seeds share the same 5% cohort; arithmetic means have no cross-seed confidence interval.'])
    (root / 'reports/adapted-baseline-test-comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    lines = ['# 适配基线：统一测试集比较', '',
             '全部使用同一923,211条测试曝光，三个种子取算术均值。主实验与后续追加评估的来源阶段分别记录，不能合称一次新的盲测。', '',
             '|模型|历史预算与聚合|AUC|GAUC|LogLoss|', '|---|---|---:|---:|---:|']
    for row in rows:
        m = row['mean_metrics']
        lines.append(f"|{row['label']}|{row['budget']}|{m['auc']:.6f}|{m['gauc']:.6f}|{m['logloss']:.6f}|")
    lines += ['', '不同结构、参数量和候选历史预算同时存在差异；均值用于比较总体效果，配对区间用于判断具体差值的统计证据。原工业模型细节并未完整复现，机制解释应优先参考各方法内部的配对控制。', '',
              '测试耗时不列入性能比较。本次存在并行外部负载，无法把运行记录解释为受控算法效率。', '',
              '[各种子指标与来源指纹](../reports/adapted-baseline-test-comparison.json) · [原验证表与适配边界](adapted-baseline-comparison.md) · [追加测试结论](supplementary-test-results.md)']
    (root / 'docs/adapted-baseline-test-comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('Published 9 baseline families, 27 test reports.')


if __name__ == '__main__':
    main()
