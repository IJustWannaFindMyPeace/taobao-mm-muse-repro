"""Publish verified three-seed four-point history curves."""
import argparse,json,hashlib,statistics
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);a=p.parse_args();root=Path(__file__).resolve().parents[1]
 refs=read(a.runs/'verified-runs.json');state=read(a.runs/'status.json');plan=read(a.runs/'plan.json')
 assert state['phase']=='completed' and not state['failed'] and len(state['completed'])==26 and len(refs)==24
 models={};comparisons={};means={}
 for key,ref in refs.items():
  folder=Path(ref['run']);assert sha(folder/'report.json')==ref['report_sha256'];assert sha(folder/'validation_predictions.npz')==ref['predictions_sha256']
  r=read(folder/'report.json');assert not r['test_evaluated'] and r['steps']==11924 and r['epochs'][0]['rows']==3052475
  assert r['store_manifest_sha256']==plan['manifest_sha256'] and r['parameters_updated'] and r['reload_max_abs_error']==0
  models[key]=dict(report=r,report_sha256=ref['report_sha256'],predictions_sha256=ref['predictions_sha256'])
 for f in ('r1','r4'):
  means[f]={}
  for l in (50,200,500,1000):means[f][str(l)]={m:statistics.mean(models[f'{f}-l{l}-seed{s}']['report']['validation'][m] for s in (42,2026,3407)) for m in ('auc','gauc','logloss')}
  for seed in (42,2026,3407):
   for lo,hi in ((50,200),(200,500),(500,1000)):
    key=f'{f}-l{hi}-vs-l{lo}-seed{seed}';c=read(a.runs/f'{key}.json')
    for role,l in (('baseline',lo),('candidate',hi)):
     model=models[f'{f}-l{l}-seed{seed}'];assert c[role]==model['report']['validation'] and c[role+'_sha256']==model['predictions_sha256']
    comparisons[key]=c
 result=dict(scope='three_seed_four_point_validation_length_extension',models=models,comparisons=comparisons,three_seed_means=means,test_evaluated=False,source_exception=plan['source_exception'],new_run_seconds=sum(x['seconds'] for x in state['completed']))
 (root/'reports/post-main-length-seeds-validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 text='''# 历史长度：三种子四点曲线

完整 MUSE 从 L50→L200、L200→L500 的 GAUC 差值在三个种子上均有正向配对区间；L500→L1000 的三个区间均跨零，不能确认进一步收益，也不能宣称统计等效或 L500 普遍最优。普通目标注意力的提升更不稳定。

[固定协议](post-main-length-seeds-protocol.md)保持 K=50、batch=256、单轮及共同数据。R1 为内容检索＋普通目标注意力，R4 为内容检索＋SimTier＋SA-TA。增加的是 GSU 可检索的历史上限，不是 ESU 的选中预算。SimTier 的聚合输入也随具体模型路径变化，不能把整条曲线解释为独立检索器的纯效应。

## 三种子均值

|模型|历史上限|AUC|GAUC|LogLoss|
|---|---:|---:|---:|---:|
'''
 for f in ('r1','r4'):
  for l,m in means[f].items():text+=f"|{f.upper()}|{l}|{m['auc']:.6f}|{m['gauc']:.6f}|{m['logloss']:.6f}|\n"
 text+='\n## 固定相邻长度比较\n\n|模型|种子|长度变化|GAUC 差|95% 用户配对区间|\n|---|---:|---|---:|---|\n'
 for f in ('r1','r4'):
  for seed in (42,2026,3407):
   for lo,hi in ((50,200),(200,500),(500,1000)):
    c=comparisons[f'{f}-l{hi}-vs-l{lo}-seed{seed}'];lower,upper=c['gauc_delta_user_bootstrap_95_percentile'];text+=f"|{f.upper()}|{seed}|{lo}→{hi}|{c['candidate_minus_baseline']['gauc']:+.6f}|[{lower:+.6f}, {upper:+.6f}]|\n"
 text+='''
## 故事线与局限

更多历史提供额外可检索信息，但收益取决于模型如何利用它。本地完整 MUSE 的前三点呈现较稳定的相邻改善，最后一步未显示明确收益。普通目标注意力仅在 200→500 的两个种子上有正向区间。两条曲线不能替代正式的“模型×长度”交互检验，也不能证明差异由某个单独模块造成。

8 次新增训练、18 项配对统计及 24 份模型产物核验均完成，新增工作累计约 87 分钟。验证指标、报告与预测指纹逐项对应；未重新选择种子、长度或主实验检查点。源码差异只涉及本轮不调用的基线分支，详情见协议。

每项区间采用 2,000 次用户级配对 bootstrap，条件于已训练模型及同一队列，未作多重比较校正。所有点均为验证集证据，不能与原冻结测试曲线混称。该数据历史最多1000条，不能外推更长历史或不同数据规模。

[完整指标和来源指纹](../reports/post-main-length-seeds-validation.json)。
'''
 (root/'docs/post-main-length-seeds-results.md').write_text(text,encoding='utf-8');print('Verified 24 models and 18 comparisons')
if __name__=='__main__':main()
