"""Train one controlled MUSE configuration on the shared validation protocol."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from muse_model import MUSEModel
from baseline_model import BaselineModel
from feature_store import FeatureStore
from metrics import evaluate
from probe_search import tensors,digest as sha256


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--store',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--max-steps',type=int)
    args=parser.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    if args.max_steps is not None and args.max_steps<1:raise ValueError('Positive step limit required')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise ValueError('Fresh directory required')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(args.config.read_text())
    limits=cfg.get('resource_limits')
    resource_samples=[]
    if limits:
        import psutil
        process=psutil.Process()
        def check_resources(step):
            sample={'step':step,'allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                    'process_rss_gib':process.memory_info().rss/2**30,
                    'system_available_gib':psutil.virtual_memory().available/2**30}
            resource_samples.append(sample)
            if (sample['allocated_gib']>limits['max_allocated_gib'] or
                sample['process_rss_gib']>limits['max_process_rss_gib'] or
                sample['system_available_gib']<limits['min_system_available_gib']):
                (args.output_dir/'resource_failure.json').write_text(json.dumps({'limits':limits,'sample':sample},indent=2))
                raise RuntimeError('Resource gate exceeded; batch and data scale were not changed')
        check_resources(0)
    if not 1<=cfg['k']<=cfg['history_length']<=1000:raise ValueError('Invalid history limits')
    root=Path(__file__).resolve().parents[1]
    implementation_hashes={p.name:sha256(p) for p in [Path(__file__),root/'scripts/probe_search.py',*sorted((root/'src').glob('*.py'))]}
    torch.manual_seed(cfg['seed']);rng=np.random.default_rng(cfg['seed'])
    train=FeatureStore(args.store,'train');validation=FeatureStore(args.store,'validation')
    cls=MUSEModel if cfg.get('model_variant','muse')=='muse' else BaselineModel
    extra={} if cls is MUSEModel else {'variant':cfg['model_variant']}
    model=cls(train.manifest['vocab_sizes'],**extra,width=cfg['embedding_width'],short_length=cfg['short_length'],
        k=cfg['k'],mode=cfg['mode'],semantic=cfg['semantic'],histogram=cfg['histogram'],long_id=cfg['long_id']).cuda()
    def get_batch(store,split,indices):
        return store.batch(indices,cfg['history_length'],with_content=True)
    sparse=list(model.embeddings.parameters());sparse_ids={id(p) for p in sparse}
    dense=[p for p in model.parameters() if id(p) not in sparse_ids and p.requires_grad]
    optimizers=[torch.optim.SparseAdam(sparse,lr=cfg['sparse_learning_rate']),
        torch.optim.AdamW(dense,lr=cfg['dense_learning_rate'],weight_decay=cfg['weight_decay'])]
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
    steps=0;curves=[];update_verified=False
    for epoch in range(cfg['epochs']):
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); epoch_started=time.perf_counter()
        model.train();seen=0;loss_sum=0.
        order=rng.permutation(len(train))
        for start in range(0,len(order),cfg['batch_size']):
            batch=get_batch(train,'train',order[start:start+cfg['batch_size']])
            for optimizer in optimizers:optimizer.zero_grad(set_to_none=True)
            inputs=tensors(batch)
            before=model.tower[-1].weight.detach().clone() if steps==0 else None
            logits=model(*inputs)
            loss=nn.functional.binary_cross_entropy_with_logits(logits,torch.as_tensor(batch['labels'],device='cuda'))
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward()
            for param in model.parameters():
                if param.grad is not None:
                    grad=param.grad.coalesce().values() if param.grad.is_sparse else param.grad
                    if not torch.isfinite(grad).all():raise RuntimeError('Nonfinite gradient')
            for optimizer in optimizers:optimizer.step()
            if before is not None:update_verified=not torch.equal(before,model.tower[-1].weight)
            seen+=len(batch['labels']);loss_sum+=loss.item()*len(batch['labels']);steps+=1
            if limits and (steps==1 or steps%100==0):check_resources(steps)
            if steps%100==0:
                print(f'epoch={epoch+1} steps={steps} rows={seen} logloss={loss_sum/seen:.6f}',flush=True)
                (args.output_dir/'progress.json').write_text(json.dumps({'epoch':epoch+1,'steps':steps,'rows':seen}))
            if args.max_steps and steps>=args.max_steps:break
        torch.cuda.synchronize()
        record={'epoch':epoch+1,'rows':seen,'online_train_logloss':loss_sum/seen,
                'training_seconds':time.perf_counter()-epoch_started,
                'training_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                'training_peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20}
        model.eval();epoch_predictions=[]
        with torch.no_grad():
            for start in range(0,len(validation),cfg['batch_size']):
                if limits and start%(100*cfg['batch_size'])==0:check_resources(steps)
                b=get_batch(validation,'validation',np.arange(start,min(start+cfg['batch_size'],len(validation))))
                epoch_predictions.append(model(*tensors(b)).cpu().numpy())
        epoch_logits=np.concatenate(epoch_predictions)
        record['validation']=evaluate(validation.labels,epoch_logits,validation.user_rows)
        np.savez(args.output_dir/f'validation-epoch-{epoch+1}.npz',logits=epoch_logits,labels=validation.labels,
                 source_rows=validation.source_rows,user_rows=validation.user_rows)
        curves.append(record)
        (args.output_dir/'epoch_metrics.json').write_text(json.dumps(curves,indent=2))
        print(json.dumps(record),flush=True)
        if args.max_steps and steps>=args.max_steps:break
    torch.cuda.synchronize()
    training_seconds=sum(record['training_seconds'] for record in curves)
    peak=max(record['training_peak_allocated_mib'] for record in curves)
    reserved=max(record['training_peak_reserved_mib'] for record in curves)
    assert update_verified
    # Record train-only memory before validation. No test dataset is instantiated.
    logits=epoch_logits
    metrics=evaluate(validation.labels,logits,validation.user_rows)
    np.savez(args.output_dir/'validation_predictions.npz',logits=logits,labels=validation.labels,source_rows=validation.source_rows,user_rows=validation.user_rows)
    torch.save(model.state_dict(),args.output_dir/'model.pt')
    # Reload into the existing model to avoid duplicating the large vocabulary on GPU.
    probe=get_batch(validation,'validation',np.arange(4))
    with torch.no_grad():before=model(*tensors(probe)).cpu()
    model.load_state_dict(torch.load(args.output_dir/'model.pt',map_location='cpu',weights_only=True))
    with torch.no_grad():error=(before-model(*tensors(probe)).cpu()).abs().max().item()
    assert error==0
    result={'scope':'partial_training_resource_probe' if args.max_steps else 'one_seed_validation_run',
        'config':cfg,'store_manifest_sha256':sha256(args.store/'manifest.json'),
        'implementation_sha256':implementation_hashes,
        'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'steps':steps,'epochs':curves,
        'training_seconds':training_seconds,'training_peak_allocated_mib':peak,'training_peak_reserved_mib':reserved,
        'parameters_updated':update_verified,'reload_max_abs_error':error,'validation':metrics,
        'test_evaluated':False,'limitations':['single seed; no comparative model result','user-disjoint validation','local architecture and optimizer adaptation']}
    if limits:
        result['resource_monitor']={'limits':limits,'sample_count':len(resource_samples),
            'max_process_rss_gib':max(s['process_rss_gib'] for s in resource_samples),
            'min_system_available_gib':min(s['system_available_gib'] for s in resource_samples),
            'policy':'checked at first and every 100 training steps, every 100 validation batches; stop without automatic changes'}
    (args.output_dir/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
