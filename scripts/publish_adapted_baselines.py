"""Publish descriptive, validation-only adapted baseline comparisons."""
import argparse,json,hashlib,statistics
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path,required=True);a=p.parse_args()
 root=Path(__file__).resolve().parents[1];inventory={x['id']:x for x in json.loads(a.inventory.read_text())['models']};seeds=(42,2026,3407)
 specs=[('DIN 局部激活适配','din','din_lau','短历史50；无长历史检索'),('短历史点积对照','din','dot_softmax','短历史50；无长历史检索'),('SIM-hard 类目检索适配',None,'b1-hard-enhanced','L1000→K50；增强 ESU'),('SIM-soft ID检索适配',None,'b2-soft-enhanced','L1000→K50；增强 ESU'),('TWIN 一致评分适配','twin','consistent','L1000；4头各50，至多200个位置'),('TWIN 内容检索对照','twin','content','L1000；4头共用内容Top50'),('MISS 双路分离适配','miss','hybrid_separate','L1000；ID25＋内容25，可能重叠'),('MISS 内容双路对照','miss','content_separate','L1000；内容Top50分成两路各25'),('MUSE 完整适配',None,'r4-l1000','L1000→K50；SimTier＋SA-TA')]
 rows=[]
 for label,source,mode,budget in specs:
  rs=[];fingerprints={}
  for seed in seeds:
   key=f'{mode}-seed{seed}'
   if source:
    path=root/f'reports/post-main-{source}-validation.json';entry=json.loads(path.read_text())['models'][key];r=entry['report'];fingerprints[str(seed)]=entry['report_sha256']
   else:
    entry=inventory[key];path=Path(entry['run'])/'report.json';assert sha(path)==entry['report_sha256'];r=json.loads(path.read_text());fingerprints[str(seed)]=sha(path)
   assert not r['test_evaluated'] and r['config']['seed']==seed and r['config']['batch_size']==256
   assert r['validation']['rows']==768773 and r['epochs'][0]['rows']==3052475 and r['steps']==11924
   rs.append(r)
  rows.append(dict(label=label,budget=budget,reports=rs,report_sha256=fingerprints,mean={m:statistics.mean(r['validation'][m] for r in rs) for m in ('auc','gauc','logloss')},mean_training_minutes=statistics.mean(r['training_seconds']/60 for r in rs),max_training_allocated_gib=max(r['training_peak_allocated_mib']/1024 for r in rs)))
 manifests={r['store_manifest_sha256'] for row in rows for r in row['reports']};assert len(manifests)==1
 result=dict(scope='descriptive_adapted_baselines_common_validation',test_evaluated=False,models=rows,store_manifest_sha256=manifests.pop())
 (root/'reports/adapted-baseline-comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 text='''# 适配基线：统一验证集比较

本表统一使用 768,773 条验证曝光，三种子取算术均值；没有把冻结测试指标与新增验证指标混在一起。训练均为同一 5% 用户队列、batch=256、一轮，但结构、历史预算与参数量不同，因此是描述性比较，不是严格等预算排行榜，也不是原论文表 5 的数值复刻。

|模型|历史预算与聚合|AUC|GAUC|LogLoss|平均训练分钟|最高训练显存 GiB|
|---|---|---:|---:|---:|---:|---:|
'''
 for r in rows:
  m=r['mean'];text+=f"|{r['label']}|{r['budget']}|{m['auc']:.6f}|{m['gauc']:.6f}|{m['logloss']:.6f}|{r['mean_training_minutes']:.2f}|{r['max_training_allocated_gib']:.2f}|\n"
 text+='''
## 如何解读

DIN 只处理短历史；与长历史模型的差异同时包含输入范围和架构。SIM 两行使用本地检索与增强 ESU，不能称原始 SIM。TWIN 四头可覆盖更多位置，省略工业交叉特征和缓存；MISS 使用本地双路适配，未还原原树检索和 MMoE。MUSE 使用公开固定内容向量，未重训 SCL 编码器。

因此，上表不能证明所有差异仅由检索方法造成。机制证据应优先阅读各模型内部的配对控制：短历史 DIN 与点积、TWIN 一致评分与内容检索、MISS 双路与内容双路，以及 MUSE 的 GSU/ESU 消融。

均值未附跨种子置信区间，不用于宣称显著优胜。各模型的用户级配对区间保留在原报告；不同种子共享数据，不是独立数据集重复。耗时来自各次真实训练，不含预处理、验证、历史内容编码成本，受运行时系统负载影响；显存为框架记录的训练峰值分配量，不是总设备占用。

[完整指标与来源报告指纹](../reports/adapted-baseline-comparison.json)。主实验冻结测试结果仍单独保留在 [final-test-results](final-test-results.md)。
'''
 (root/'docs/adapted-baseline-comparison.md').write_text(text,encoding='utf-8');print('Published 9 model families, 27 validation reports')
if __name__=='__main__':main()
