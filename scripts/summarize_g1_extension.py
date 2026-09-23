"""Validate and publish the fixed two-stage G1 validation experiment."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];runs=a.runs
    status=json.loads((runs/'status.json').read_text());plan=json.loads((runs/'plan.json').read_text())
    assert status['phase']=='completed' and len(status['completed'])==6 and not status['failed']
    models={};comparisons={};seeds=(42,2026,3407)
    for seed in seeds:
        c=json.loads((runs/f'primary-seed{seed}.json').read_text())
        control=plan['reused_controls'][str(seed)];base=Path(control['run'])
        assert sha(base/'report.json')==control['report_sha256']
        assert sha(base/'validation_predictions.npz')==control['predictions_sha256']==c['baseline_sha256']
        assert json.loads((base/'report.json').read_text())['validation']==c['baseline']
        for stage in ('scorer','downstream'):
            job=f'{stage}-seed{seed}';folder=runs/job;r=json.loads((folder/'report.json').read_text())
            assert r['config']==plan['configs'][job] and r['store_manifest_sha256']==plan['manifest_sha256']
            expected={k:v for k,v in plan['source_hashes'].items() if not k.startswith('train_g1_') or k==f'train_g1_{stage}.py'}
            assert r['implementation_sha256']==expected
            assert r['steps']==11924 and r['epochs'][0]['rows']==3052475 and not r['test_evaluated']
            assert r['validation']['rows']==768773 and r['reload_max_abs_error']==0 and r['parameters_updated']
            assert r['projection_parameters_updated'] if stage=='scorer' else r['projection_unchanged']
            if stage=='downstream':
                assert c['candidate']==r['validation'] and c['candidate_sha256']==sha(folder/'validation_predictions.npz')
                assert r['scorer_provenance']['checkpoint_sha256']==sha(runs/f'scorer-seed{seed}/model.pt')
                assert r['scorer_provenance']['report_sha256']==sha(runs/f'scorer-seed{seed}/report.json')
            models[job]={'report':r,'report_sha256':sha(folder/'report.json')}
        comparisons[str(seed)]=c
    means={role:{k:statistics.mean(c[role][k] for c in comparisons.values()) for k in ('auc','gauc','logloss')} for role in ('baseline','candidate')}
    result=dict(scope='post_main_validation_extension',test_evaluated=False,models=models,comparisons=comparisons,three_seed_means=means,reused_controls=plan['reused_controls'])
    (root/'reports/post-main-g1-validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    text='''# G1 可学习内容检索：三种子结果

本地两阶段方案未改善检索效果：三个种子的 GAUC 均低于原始内容余弦对照，95% 用户级配对区间均完全低于零；AUC 下降，LogLoss 上升。额外评分器预训练没有转化为下游收益。

[实验协议](post-main-g1-protocol.md)固定训练内容投影一轮，再冻结投影用于 Top-50 检索。下游 ESU、公共训练设置和输入保持一致；对照复用已核验的同种子内容余弦模型。该方案是监督和梯度路径明确的本地适配，不是原论文未公开训练细节的唯一还原。

|种子|内容余弦 GAUC|学习后检索 GAUC|学习后 − 原始|95% 配对区间|
|---|---:|---:|---:|---|
'''
    for seed,c in comparisons.items():
        lo,hi=c['gauc_delta_user_bootstrap_95_percentile']
        text+=f"|{seed}|{c['baseline']['gauc']:.6f}|{c['candidate']['gauc']:.6f}|{c['candidate_minus_baseline']['gauc']:+.6f}|[{lo:+.6f}, {hi:+.6f}]|\n"
    text+='\n|三种子均值|AUC|GAUC|LogLoss|\n|---|---:|---:|---:|\n'
    for role,label in [('baseline','原始内容余弦'),('candidate','冻结学习后投影')]:
        m=means[role];text+=f"|{label}|{m['auc']:.6f}|{m['gauc']:.6f}|{m['logloss']:.6f}|\n"
    text+='''
## 训练成本与核验

|种子|评分器训练分钟|下游训练分钟|评分器峰值 GiB|下游峰值 GiB|
|---|---:|---:|---:|---:|
'''
    for seed in seeds:
        s=models[f'scorer-seed{seed}']['report'];d=models[f'downstream-seed{seed}']['report']
        text+=f"|{seed}|{s['training_seconds']/60:.2f}|{d['training_seconds']/60:.2f}|{s['training_peak_allocated_mib']/1024:.2f}|{d['training_peak_allocated_mib']/1024:.2f}|\n"
    text+='''
表中耗时只计训练，不含验证和文件保存。六次运行的训练、验证及保存累计约 2.30 小时；不包含独立配对统计和前置资源探针。完整运行比短探针外推更快，预算估计不是实际耗时。

每次训练使用 3,052,475 条曝光、一轮 11,924 步，验证 768,773 条曝光。已核验源码、配置、数据清单、对照文件与预测指纹；评分器投影确实更新，下游投影保持冻结，公共参数正常更新，保存重载预测误差均为零。

## 故事线与局限

痛点是原始余弦没有针对当前点击任务学习评分；改进尝试是用内容分支的点击监督学习投影，再用于检索。证据表明，这一尝试在当前共同协议下反而损害排序与概率预测。辅助分支训练目标与离散 Top-K 下游收益可能不一致，但本实验不能单独证明下降原因。

因此保留原始内容余弦作为该对照中的优胜方案，不为制造正收益选择种子或追加调参。结论限于固定向量、固定预训练目标、一轮预训练和当前用户子集；不能外推为所有可学习内容检索无效。

每种子区间来自 2,000 次用户级配对 bootstrap，条件于已训练模型，无多重比较校正。当前为验证集补充证据，不是新的盲测，也不改写原冻结测试结果。

完整指标与来源指纹见[机器可读报告](../reports/post-main-g1-validation.json)。
'''
    (root/'docs/post-main-g1-results.md').write_text(text,encoding='utf-8')
    print(json.dumps(means,indent=2))


if __name__=='__main__':
    main()
