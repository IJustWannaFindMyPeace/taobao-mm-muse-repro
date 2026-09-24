"""Fixed-row anonymous retrieval examples; no labels, updates, or test access."""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'src'),str(root/'extensions')]
from feature_store import FeatureStore
from learned_content_search import ContentProjection
from enhanced_esu import semantic_scores,fuse_scores
from twin import masked_topk
from retrieval import select_history

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args();w=a.work
 torch.set_num_threads(4);store=FeatureStore(w/'shared-feature-store-050','validation')
 rows=np.linspace(0,len(store)-1,8,dtype=np.int64)
 g1path=w/'post-main-g1-extension-v1/downstream-seed42/model.pt';g3path=w/'post-main-g3-extension-v1/fused_attention-seed42/model.pt'
 g1=torch.load(g1path,map_location='cpu',mmap=True,weights_only=True);g3=torch.load(g3path,map_location='cpu',mmap=True,weights_only=True)
 projection=ContentProjection();projection.load_state_dict({k.removeprefix('projection.'):v for k,v in g1.items() if k.startswith('projection.')});projection.eval()
 cases=[]
 with torch.no_grad():
  for number,row in enumerate(rows,1):
   b=store.batch(np.array([row]),1000,with_content=True)
   features=torch.as_tensor(b['features'],dtype=torch.long);item=torch.as_tensor(b['history_item'],dtype=torch.long);cat=torch.as_tensor(b['history_category'],dtype=torch.long)
   valid=torch.as_tensor(b['history_valid']);tc=torch.as_tensor(b['target_content']);hc=torch.as_tensor(b['history_content']);ta=torch.as_tensor(b['target_content_available']);ha=torch.as_tensor(b['history_content_available'])
   cosine,eligible=semantic_scores(tc,hc,valid,ta,ha)
   case=dict(case=number,selection_index=int(row),valid_history_positions=int(valid.sum()),content_eligible_positions=int(eligible.sum()))
   if not bool(ta[0]) or tc.norm().item()==0:
    case['status']='target content unavailable; retained in sample, score comparison not applicable';cases.append(case);continue
   target=torch.cat([F.embedding(features[:,6],g3['embeddings.205.weight']),F.embedding(features[:,7],g3['embeddings.206.weight'])],-1)
   history=torch.cat([F.embedding(item,g3['embeddings.205.weight']),F.embedding(cat,g3['embeddings.206.weight'])],-1)
   q=F.linear(target,g3['long_attention.query.weight'],g3['long_attention.query.bias']);k=F.linear(history,g3['long_attention.key.weight'],g3['long_attention.key.bias'])
   ids=(q[:,None]*k).sum(-1)/history.shape[-1]**.5
   scores={'content':cosine,'g1':projection.scores(tc,hc),'g3':fuse_scores(ids,cosine,g3['gamma'])}
   selections={}
   for mode,score in scores.items():
    indices,mask=masked_topk(score,eligible,50);selections[mode]=indices[0][mask[0]].tolist()
   ci,cm=select_history(tc,hc,valid,ta,ha,50);assert selections['content']==ci[0][cm[0]].tolist()
   names={int(features[0,6]):'Target'};categories={int(features[0,7]):'TargetCategory'}
   for pos in range(item.shape[1]):
    if not valid[0,pos]:continue
    key=int(item[0,pos]);category=int(cat[0,pos])
    if key not in names:names[key]=f'Item{len(names):03d}'
    if category not in categories:categories[category]=f'Category{len(categories):03d}'
   case['status']='scored';case['routes']={}
   for mode,positions in selections.items():
    case['routes'][mode]=dict(top50_positions=positions,top10=[dict(rank=rank+1,position=pos,item=names[int(item[0,pos])],category=categories[int(cat[0,pos])],retrieval_score=float(scores[mode][0,pos]),content_cosine=float(cosine[0,pos])) for rank,pos in enumerate(positions[:10])])
   case['overlap_with_content']={mode:dict(shared_positions=len(set(positions)&set(selections['content'])),selected_positions=len(positions)) for mode,positions in selections.items() if mode!='content'}
   cases.append(case)
 result=dict(scope='posthoc_anonymous_retrieval_examples',split='validation',seed=42,sampling='8 evenly spaced exposure indices fixed before score computation; no label or performance filtering',history_length=1000,k=50,store_manifest_sha256=sha(w/'shared-feature-store-050/manifest.json'),checkpoint_sha256={'g1':sha(g1path),'g3':sha(g3path)},implementation_sha256=sha(Path(__file__)),cases=cases,test_evaluated=False)
 (root/'reports/anonymous-retrieval-cases.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 text='''# 匿名检索案例：同一历史，三种评分

固定取验证集 8 个等间隔曝光位置，使用种子 42 的最终检查点；不按标签、指标或展示效果选例。比较原始内容余弦、G1 预训练后冻结的内容评分、G3 共享融合评分，均从最多 1000 条历史中选 50 个位置。

这些案例只解释“评分规则改变了哪些历史被选中”，不能仅凭个例判断模型优劣或商品真实语义。没有标题和图片，不能把匿名类目命名为跑鞋、咖啡机等实际品类。评分不是 pCTR，也不是经过 softmax 的注意力权重；不同方法的分值尺度不能直接横比。

物品和类目均在每个案例内重新编号，跨案例同名不表示同一物品。Target 表示与目标相同的词表索引；匿名标识来源于模型词表，OOV 可能合并多个原始商品。位置是长度 1000 输入数组的零基位置，不是时间戳；重叠按历史位置计算，重复点击不合并成一个位置。

## Top-50 重叠概览

|案例|有效历史|内容有效位置|G1 与内容重叠|G3 与内容重叠|
|---|---:|---:|---:|---:|
'''
 for c in cases:
  if c['status']!='scored':text+=f"|{c['case']}|{c['valid_history_positions']}|{c['content_eligible_positions']}|不适用|不适用|\n";continue
  overlap=c['overlap_with_content'];text+=f"|{c['case']}|{c['valid_history_positions']}|{c['content_eligible_positions']}|{overlap['g1']['shared_positions']}/{overlap['g1']['selected_positions']}|{overlap['g3']['shared_positions']}/{overlap['g3']['selected_positions']}|\n"
 for c in cases:
  text+=f"\n## 案例 {c['case']}\n\n"
  if c['status']!='scored':text+='目标内容缺失，保留该抽样位置但不比较语义评分。\n';continue
  text+='各方法前 5 名如下，完整 Top-50 位置与前 10 名分数见机器可读报告。\n\n|方法|名次|历史位置|物品|类目|检索分数|原内容余弦|\n|---|---:|---:|---|---|---:|---:|\n'
  for mode,route in c['routes'].items():
   for x in route['top10'][:5]:text+=f"|{mode}|{x['rank']}|{x['position']}|{x['item']}|{x['category']}|{x['retrieval_score']:.5f}|{x['content_cosine']:.5f}|\n"
 text+='''
## 证据边界

G1 与 G3 是不同训练方案的最终模型，本表不是受控因果归因。只展示一个种子和固定八个位置；不能外推整体分布或宣称重复实验稳定性。G1/G3 的三种子整体效果分别见 [G1](post-main-g1-results.md) 和 [G3](post-main-g3-results.md)。本程序没有重训参数、读取测试集或输出用户 ID、标签及原始商品 ID。

[抽样规则、检查点指纹与完整案例](../reports/anonymous-retrieval-cases.json)。
'''
 (root/'docs/anonymous-retrieval-cases.md').write_text(text,encoding='utf-8');print('Published',len(cases),'fixed cases')
if __name__=='__main__':main()
