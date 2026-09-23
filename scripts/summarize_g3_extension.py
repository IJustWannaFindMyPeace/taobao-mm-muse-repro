"""Publish fixed G3 comparisons and final-checkpoint behavioral diagnosis."""
import argparse,hashlib,json,statistics
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    status=json.loads((a.runs/'status.json').read_text());plan=json.loads((a.runs/'plan.json').read_text())
    behavior=json.loads((a.runs/'behavior-audit.json').read_text())
    assert status['phase']=='completed' and len(status['completed'])==6 and not status['failed']
    models={};comparisons={}
    for seed in (42,2026,3407):
        c=json.loads((a.runs/f'primary-seed{seed}.json').read_text())
        for mode,role in [('fused_attention','candidate'),('content','baseline')]:
            job=f'{mode}-seed{seed}';folder=a.runs/job;r=json.loads((folder/'report.json').read_text())
            assert r['config']==plan['configs'][job] and r['implementation_sha256']==plan['source_hashes']
            assert r['steps']==11924 and r['epochs'][0]['rows']==3052475 and not r['test_evaluated']
            assert r['validation']['rows']==768773 and r['reload_max_abs_error']==0 and r['parameters_updated']
            assert c[role]==r['validation'] and c[role+'_sha256']==sha(folder/'validation_predictions.npz')
            models[job]={'report':r,'report_sha256':sha(folder/'report.json')}
        assert behavior['results'][str(seed)]['baseline_predictions_exact_original']
        comparisons[str(seed)]=c
    means={m:{k:statistics.mean(models[f'{m}-seed{s}']['report']['validation'][k] for s in (42,2026,3407))
              for k in ('auc','gauc','logloss')} for m in ('fused_attention','content')}
    result={'scope':'post_main_validation_extension','test_evaluated':False,'models':models,'comparisons':comparisons,
            'three_seed_means':means,'behavior_audit':behavior}
    (root/'reports/post-main-g3-validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    text='''# G3 融合注意力检索：结果与行为诊断

共享 SA-TA 评分用于检索未显示稳定优势：三个种子的 GAUC 点估计均低于内容余弦对照，一个区间跨零、两个完全低于零。种子 2026 的下降远大于另外两次，必须单列解释，不能仅以均值掩盖不稳定性。

这是固定公共协议的验证集适配实验，非原论文唯一实现；[协议](post-main-g3-protocol.md)记录评分、初始化与梯度路径。三个内容对照与此前同种子的完整 L1000 MUSE 验证预测逐元素完全相同。

|种子|内容检索 GAUC|融合检索 GAUC|融合 − 内容|95% 配对区间|
|---|---:|---:|---:|---|
'''
    for seed,c in comparisons.items():
        lo,hi=c['gauc_delta_user_bootstrap_95_percentile']
        text+=f"|{seed}|{c['baseline']['gauc']:.6f}|{c['candidate']['gauc']:.6f}|{c['candidate_minus_baseline']['gauc']:+.6f}|[{lo:+.6f}, {hi:+.6f}]|\n"
    text+='\n|三种子均值|AUC|GAUC|LogLoss|\n|---|---:|---:|---:|\n'
    for m,label in [('fused_attention','融合检索'),('content','内容检索')]:
        x=means[m];text+=f"|{label}|{x['auc']:.6f}|{x['gauc']:.6f}|{x['logloss']:.6f}|\n"
    text+='''
## 最终检查点行为

固定抽取 1024 个等间隔验证曝光位置，读取最终权重，不更新模型。对内容有效的历史，保持 ID 分数不变时，融合评分对内容相似度的斜率是 `gamma[1] + gamma[2] * ID_score`。该斜率可以为负；不能仅凭一个系数为负判断整体排序。

|种子|负斜率比例|与内容 Top-50 重叠率|融合选中平均内容余弦|内容对照平均余弦|
|---|---:|---:|---:|---:|
'''
    for seed,b in behavior['results'].items():
        text+=f"|{seed}|{b['negative_semantic_slope_fraction']:.2%}|{b['content_topk_overlap_fraction']:.2%}|{b['mean_selected_content_cosine_fused']:.6f}|{b['mean_selected_content_cosine_reference']:.6f}|\n"
    text+='''
2026 的最终检索明显偏向内容相似度较低的行为，与其 GAUC 下降相一致。诊断描述最终模型的评分行为，不建立训练过程的因果解释，也不证明给融合系数加正约束一定有效。另两个种子 AUC/LogLoss 有改善而 GAUC 不占优，不能将结果简化为全指标一致退步。

六次完整训练均通过行数、源文件、配置、预测对齐及重载检查。区间采用每种子 2,000 次用户级配对 bootstrap，条件于已训练模型，无多重校正。原结果保留；未根据诊断调整权重、补选种子或重写冻结测试表。缺失目标内容的回退样本不计入上述检索行为统计。

全部指标、指纹、权重与诊断样本位置见[机器可读报告](../reports/post-main-g3-validation.json)。G1 多模态可学习评分仍是独立缺口，不能用本结果替代。
'''
    (root/'docs/post-main-g3-results.md').write_text(text,encoding='utf-8')
    print(json.dumps(means,indent=2))


if __name__=='__main__':main()
