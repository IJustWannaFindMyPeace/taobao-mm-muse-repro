"""Audit and publish paired four-head retrieval validation results."""
import argparse,json,hashlib,statistics
from pathlib import Path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];models={};comparisons={}
    status=json.loads((a.runs/'status.json').read_text())
    assert len(status['completed'])==6 and not status['failed'] and not status['active']
    for seed in (42,2026,3407):
        c=json.loads((a.runs/f'comparison-seed{seed}.json').read_text())
        for mode,role in [('consistent','candidate'),('content','baseline')]:
            job=f'{mode}-seed{seed}';folder=a.runs/job;r=json.loads((folder/'report.json').read_text())
            assert r['steps']==11924 and r['epochs'][0]['rows']==3052475
            assert r['validation']['rows']==768773 and not r['test_evaluated']
            assert r['parameters_updated'] and r['reload_max_abs_error']==0
            assert c[role]==r['validation'] and c[role+'_sha256']==sha(folder/'validation_predictions.npz')
            models[job]={'report_sha256':sha(folder/'report.json'),'report':r}
        comparisons[str(seed)]=c
    means={m:{k:statistics.mean(models[f'{m}-seed{s}']['report']['validation'][k] for s in (42,2026,3407))
              for k in ('auc','gauc','logloss')} for m in ('consistent','content')}
    summary={'scope':'post_main_validation_extension','test_evaluated':False,'models':models,
             'paired_comparisons':comparisons,'three_seed_means':means}
    (root/'reports/post-main-twin-validation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    text='''# 四头一致性检索补充结果

在固定公共主干中，内容余弦检索优于四头 ID 一致性检索适配。三个种子的 GAUC 差值区间均不包含零，AUC 与 LogLoss 方向也一致。这是验证集上的本地适配结果，不代表原始工业 TWIN 的完整复现。

差值方向为“一致性检索 − 内容检索”。

|种子|一致性 GAUC|内容 GAUC|GAUC 差|95% 用户配对区间|
|---|---:|---:|---:|---|
'''
    for seed,c in comparisons.items():
        lo,hi=c['gauc_delta_user_bootstrap_95_percentile'];assert hi<0
        text+=f"|{seed}|{c['candidate']['gauc']:.6f}|{c['baseline']['gauc']:.6f}|{c['candidate_minus_baseline']['gauc']:+.6f}|[{lo:+.6f}, {hi:+.6f}]|\n"
    text+='\n|三种子均值|AUC|GAUC|LogLoss|\n|---|---:|---:|---:|\n'
    for m,label in [('consistent','四头一致性检索'),('content','内容检索＋相同四头聚合')]:
        v=means[m];text+=f"|{label}|{v['auc']:.6f}|{v['gauc']:.6f}|{v['logloss']:.6f}|\n"
    text+='''
## 预算与解释

每头 K=50，两模型每例最多处理 200 个头—位置组合。内容检索实际平均覆盖约 49.89 个不同历史位置；一致性检索随种子约覆盖 128.64、149.62、175.51 个位置。因此，这不是相同去重覆盖预算的消融。SimTier 对头间位置并集计数，分布也随检索变化。

结果支持：在当前特征、数据和训练协议下，检索与聚合评分一致不保证获得更好的最终排序指标。它不能单独识别参数共享、覆盖数量或直方图变化的因果贡献，也不能直接与原单路 K=50 主表作等预算排名。

六次运行均完成 3,052,475 条训练样本及 768,773 条验证样本，固定 batch=256、单轮、三个种子。各次训练约 22–23 分钟。配对区间使用 2,000 次用户 bootstrap，条件于各次已训练模型；不包含完整的种子和队列抽样不确定性。这是主测试已查看后的补充，不改写原冻结测试表。

见[固定协议](post-main-twin-protocol.md)及[机器可读结果](../reports/post-main-twin-validation.json)。
'''
    (root/'docs/post-main-twin-results.md').write_text(text,encoding='utf-8')
    print(json.dumps(means,indent=2))


if __name__=='__main__':main()
