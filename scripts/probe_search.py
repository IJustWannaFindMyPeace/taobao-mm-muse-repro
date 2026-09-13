"""Verify companion prediction parity and profile R0/R1 training steps."""
import argparse,gc,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from c0 import C0
from feature_store import FeatureStore
from search_model import SearchModel


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensors(batch):
    return [torch.as_tensor(batch[key],device='cuda',dtype=torch.long if key in ['features','history_item','history_category'] else None)
            for key in ['features','history_item','history_category','history_valid','target_content','history_content','target_content_available','history_content_available']]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--store',type=Path,required=True)
    parser.add_argument('--c0-run',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise ValueError('Fresh output directory required')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    provenance=json.loads((root/'reports/shared-code-provenance.json').read_text())
    for name,expected in provenance['files'].items():
        if digest(root/name)!=expected:raise ValueError('Shared source digest mismatch: '+name)
    train=FeatureStore(args.store,'train');val=FeatureStore(args.store,'validation')
    original=json.loads((args.c0_run/'report.json').read_text())
    if original['store_manifest_sha256']!=digest(args.store/'manifest.json'):raise ValueError('Store mismatch')
    model=C0(train.manifest['vocab_sizes']).cuda()
    model.load_state_dict(torch.load(args.c0_run/'model.pt',map_location='cpu',weights_only=True));model.eval()
    batch=val.batch(np.arange(32),200,with_content=True)
    with torch.no_grad():pred=model(*tensors(batch)[:4]).cpu().numpy()
    with np.load(args.c0_run/'validation_predictions.npz') as old:
        np.testing.assert_array_equal(batch['source_rows'],old['source_rows'][:32])
        np.testing.assert_allclose(pred,old['logits'][:32],rtol=0,atol=1e-6)
        parity=float(np.max(np.abs(pred-old['logits'][:32])))
    del model,batch;gc.collect();torch.cuda.empty_cache()
    rows=np.random.default_rng(42).permutation(len(train))[:640]
    results=[]
    for length in [200,1000]:
        for mode in ['recent','cosine']:
            torch.manual_seed(42)
            model=SearchModel(train.manifest['vocab_sizes'],k=50,mode=mode).cuda()
            sparse=list(model.embeddings.parameters());sparse_ids={id(p) for p in sparse}
            dense=[p for p in model.parameters() if id(p) not in sparse_ids]
            optimizers=[torch.optim.SparseAdam(sparse,lr=.001),torch.optim.AdamW(dense,lr=.001,weight_decay=0)]
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
            before=model.tower[-1].weight.detach().clone()
            for offset in range(0,len(rows),32):
                batch=train.batch(rows[offset:offset+32],length,with_content=True)
                for optimizer in optimizers:optimizer.zero_grad(set_to_none=True)
                logits=model(*tensors(batch))
                loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.as_tensor(batch['labels'],device='cuda'))
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                loss.backward()
                for p in model.parameters():
                    if p.grad is not None:
                        gradient=p.grad.coalesce().values() if p.grad.is_sparse else p.grad
                        if not torch.isfinite(gradient).all():raise RuntimeError('Nonfinite gradient')
                for optimizer in optimizers:optimizer.step()
            torch.cuda.synchronize()
            record={'mode':mode,'history_length':length,'k':50,'batch_size':32,'steps':20,'rows':640,
                    'seconds':time.perf_counter()-start,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                    'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
                    'parameters_updated':not torch.equal(before,model.tower[-1].weight)}
            assert record['parameters_updated'];results.append(record);print(json.dumps(record),flush=True)
            del model,sparse,dense,optimizers,optimizer,p,gradient,loss,logits,batch,before
            gc.collect();torch.cuda.empty_cache()
    report={'scope':'training_resource_probe_not_effectiveness_evaluation','store_manifest_sha256':digest(args.store/'manifest.json'),
            'c0_prediction_parity':{'rows':32,'max_abs_logit_error':parity},'profiles':results,
            'implementation_sha256':{p.name:digest(p) for p in [Path(__file__),root/'src/retrieval.py',root/'src/search_model.py']},
            'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'test_evaluated':False,
            'limitations':['20-step probes include startup and CPU feature lookup; not steady-state benchmarks',
                           'R0/R1 only; SA-TA and learned SimTier not yet implemented',
                           'current 1% vocabulary; not a maximum-scale guarantee']}
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
