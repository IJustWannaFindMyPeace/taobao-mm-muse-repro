"""Publish audited post-main DIN validation results, never rewrite the test table."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runs',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    status=json.loads((a.runs/'status.json').read_text())
    assert len(status['completed'])==6 and status['failed'] is None and status['active'] is None
    results={};comparisons={}
    for seed in (42,2026,3407):
        c=json.loads((a.runs/f'comparison-seed{seed}.json').read_text())
        for mechanism,role in [('din_lau','candidate'),('dot_softmax','baseline')]:
            job=f'{mechanism}-seed{seed}';folder=a.runs/job
            r=json.loads((folder/'report.json').read_text())
            assert r['steps']==11924 and r['epochs'][0]['rows']==3052475
            assert r['validation']['rows']==768773 and not r['test_evaluated']
            assert r['parameters_updated'] and r['reload_max_abs_error']==0
            assert c[role+'_sha256']==digest(folder/'validation_predictions.npz')
            assert c[role]==r['validation']
            results[job]={'report_sha256':digest(folder/'report.json'),'report':r}
        comparisons[str(seed)]=c
    means={m:{k:statistics.mean(results[f'{m}-seed{s}']['report']['validation'][k]
                for s in (42,2026,3407)) for k in ('auc','gauc','logloss')}
           for m in ('din_lau','dot_softmax')}
    summary={'scope':'post_main_validation_extension','test_evaluated':False,
             'models':results,'paired_comparisons':comparisons,'three_seed_means':means}
    (root/'reports/post-main-din-validation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    text='''# DIN 局部激活机制：三种子验证结果

在相同短历史长度和公共主干下，DIN 局部激活机制未显示可靠的 GAUC 改善。三个种子的点估计均略低于点积 softmax 对照，但各自的 95% 配对置信区间均包含零，不能据此确认 DIN 更差，也不能认定两者等效。

这是主测试结果已查看后设计的验证集补充实验，不属于原冻结测试表，也不是原始工业 DIN 的完整复现。[固定协议与实现边界](post-main-din-protocol.md)记录了结构、训练设置和本地适配。

## 配对结果

差值方向为 DIN − 点积 softmax；GAUC 越大越好。

|种子|DIN GAUC|点积 softmax GAUC|GAUC 差|95% 配对区间|
|---|---:|---:|---:|---|
'''
    for seed,c in comparisons.items():
        lo,hi=c['gauc_delta_user_bootstrap_95_percentile']
        text+=f"|{seed}|{c['candidate']['gauc']:.6f}|{c['baseline']['gauc']:.6f}|{c['candidate_minus_baseline']['gauc']:+.6f}|[{lo:+.6f}, {hi:+.6f}]|\n"
    text+='\n## 三种子均值\n\n|模型|AUC|GAUC|LogLoss|\n|---|---:|---:|---:|\n'
    for m,label in [('din_lau','DIN 局部激活'),('dot_softmax','点积 softmax')]:
        x=means[m];text+=f"|{label}|{x['auc']:.6f}|{x['gauc']:.6f}|{x['logloss']:.6f}|\n"
    text+='''
## 证据与局限

每次完整训练 3,052,475 条样本、11,924 步，完整验证 768,773 条样本；batch=256，单轮，5% 用户队列。六次运行均通过参数更新、有限梯度、资源门槛及检查点重载检查。DIN 训练峰值显存约 7.83 GiB，点积对照约 7.79 GiB。

区间采用 2,000 次用户级配对 bootstrap，条件于各次已训练模型，不包含训练随机性或队列抽样不确定性的完整估计。三个种子单独报告，不把它们拼接为独立用户样本。

该对照只支持关于短历史注意力模块的结论，不能单独证明长历史的价值，也不能证明 MUSE 优于原始 DIN。长历史与多模态聚合的证据仍来自主实验对应消融。AUC 和 LogLoss 的种子间方向也不完全一致，不能表述为全指标胜负。

机器可读结果、配置、源文件指纹及配对统计见 [结果文件](../reports/post-main-din-validation.json)。
'''
    (root/'docs/post-main-din-results.md').write_text(text,encoding='utf-8')
    print(json.dumps(means,indent=2))


if __name__=='__main__':main()
