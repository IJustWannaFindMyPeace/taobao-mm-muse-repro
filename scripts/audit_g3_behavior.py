"""Read-only final-checkpoint diagnostic; fixed validation rows, no refitting."""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'src'),str(root/'extensions')]
from feature_store import FeatureStore
from enhanced_esu import semantic_scores,fuse_scores
from retrieval import select_history
from twin import masked_topk


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--store',type=Path,required=True);p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--inventory',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    store=FeatureStore(a.store,'validation');rows=np.linspace(0,len(store)-1,1024,dtype=np.int64)
    inventory=json.loads(a.inventory.read_text())['models'];results={}
    torch.set_num_threads(4)
    with torch.no_grad():
        for seed in (42,2026,3407):
            old=Path(next(x['run'] for x in inventory if x['id']==f'r4-l1000-seed{seed}'))
            with np.load(old/'validation_predictions.npz') as x,np.load(a.runs/f'content-seed{seed}/validation_predictions.npz') as y:
                for key in ('source_rows','labels','user_rows','logits'):np.testing.assert_array_equal(x[key],y[key])
            checkpoint=a.runs/f'fused_attention-seed{seed}/model.pt'
            s=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
            totals=dict(eligible=0,negative_semantic_slope=0,selected=0,overlap=0,fused_cos_sum=0.,content_cos_sum=0.,content_selected=0)
            for start in range(0,len(rows),64):
                b=store.batch(rows[start:start+64],1000,with_content=True)
                features=torch.as_tensor(b['features'],dtype=torch.long)
                item=torch.as_tensor(b['history_item'],dtype=torch.long);cat=torch.as_tensor(b['history_category'],dtype=torch.long)
                valid=torch.as_tensor(b['history_valid']);tc=torch.as_tensor(b['target_content']);hc=torch.as_tensor(b['history_content'])
                ta=torch.as_tensor(b['target_content_available']);ha=torch.as_tensor(b['history_content_available'])
                target=torch.cat([F.embedding(features[:,6],s['embeddings.205.weight']),F.embedding(features[:,7],s['embeddings.206.weight'])],-1)
                history=torch.cat([F.embedding(item,s['embeddings.205.weight']),F.embedding(cat,s['embeddings.206.weight'])],-1)
                q=F.linear(target,s['long_attention.query.weight'],s['long_attention.query.bias'])
                k=F.linear(history,s['long_attention.key.weight'],s['long_attention.key.bias'])
                ids=(q[:,None]*k).sum(-1)/history.shape[-1]**.5
                cosine,eligible=semantic_scores(tc,hc,valid,ta,ha)
                fi,fm=masked_topk(fuse_scores(ids,cosine,s['gamma']),eligible,50)
                ci,cm=select_history(tc,hc,valid,ta,ha,50)
                cm &= eligible.gather(1,ci)
                membership=torch.zeros_like(valid);membership.scatter_(1,ci,cm)
                totals['eligible']+=int(eligible.sum())
                totals['negative_semantic_slope']+=int(((s['gamma'][1]+s['gamma'][2]*ids<0)&eligible).sum())
                totals['selected']+=int(fm.sum());totals['content_selected']+=int(cm.sum())
                totals['overlap']+=int((membership.gather(1,fi)&fm).sum())
                totals['fused_cos_sum']+=float((cosine.gather(1,fi)*fm).sum())
                totals['content_cos_sum']+=float((cosine.gather(1,ci)*cm).sum())
            results[str(seed)]={'gamma':s['gamma'].tolist(),'baseline_predictions_exact_original':True,
                'negative_semantic_slope_fraction':totals['negative_semantic_slope']/totals['eligible'],
                'content_topk_overlap_fraction':totals['overlap']/totals['selected'],
                'mean_selected_content_cosine_fused':totals['fused_cos_sum']/totals['selected'],
                'mean_selected_content_cosine_reference':totals['content_cos_sum']/totals['content_selected'],
                'counts':totals,'checkpoint_sha256':sha(checkpoint)}
            print(seed,json.dumps(results[str(seed)]),flush=True)
            del s
    a.report.write_text(json.dumps({'scope':'posthoc_final_checkpoint_behavior_diagnostic','split':'validation',
        'sample':'1024 evenly spaced exposure row indices, fixed before score computation','sample_rows':rows.tolist(),
        'implementation_sha256':sha(Path(__file__)),'results':results,
        'limitations':['descriptive final-checkpoint evidence, not causal attribution or training trajectory',
                       'only content-eligible positions enter behavioral statistics; missing-target fallback excluded',
                       'semantic slope holds ID score fixed; coefficient signs alone are not causal evidence',
                       'no tuning, test prediction, or parameter updates']},indent=2),encoding='utf-8')


if __name__=='__main__':main()
