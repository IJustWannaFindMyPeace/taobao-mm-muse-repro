"""Resource-gated, fixed dual-route validation study; no test evaluation."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--store',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--max-estimated-hours',type=float,default=6)
    a=p.parse_args();root=Path(__file__).resolve().parents[1];out=a.output_dir.resolve();store=a.store.resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Fresh output required')
    out.mkdir(parents=True,exist_ok=True)
    modes=('hybrid_separate','content_separate','hybrid_shared')
    jobs=[f'{m}-seed{s}' for s in (42,2026,3407) for m in modes]
    state={'phase':'probes','planned':jobs,'completed':[],'probes_completed':[],
           'active':None,'failed':None,'test_evaluated':False}
    sources=[root/'scripts/train_miss_extension.py',root/'scripts/probe_search.py',
             *sorted((root/'src').glob('*.py')),*sorted((root/'extensions').glob('*.py'))]
    hashes={p.name:sha(p) for p in sources};manifest_hash=sha(store/'manifest.json')
    configs={job:json.loads((root/f'configs/post-main-miss-{job}.json').read_text()) for job in jobs}
    (out/'plan.json').write_text(json.dumps({'configs':configs,'source_hashes':hashes,
        'manifest_sha256':manifest_hash,'max_estimated_hours':a.max_estimated_hours,
        'primary':'hybrid_separate minus content_separate','sensitivity':'hybrid_shared minus hybrid_separate'},indent=2))
    def save():
        tmp=out/'status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(out/'status.json')
    def execute(command,log):
        with log.open('w',encoding='utf-8') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
    def train(job,probe=False):
        name=('probe-' if probe else '')+job
        state['active']=name;save()
        cfg=root/f'configs/post-main-miss-{job}.json'
        assert json.loads(cfg.read_text())==configs[job]
        assert all(sha(p)==hashes[p.name] for p in sources)
        command=[sys.executable,str(root/'scripts/train_miss_extension.py'),'--store',str(store),
                 '--config',str(cfg),'--output-dir',str(out/name)]
        if probe:command+=['--max-steps','100']
        execute(command,out/f'{name}.log')
        r=json.loads((out/name/'report.json').read_text())
        assert r['config']==configs[job] and r['implementation_sha256']==hashes
        assert r['store_manifest_sha256']==manifest_hash and not r['test_evaluated']
        assert r['steps']==(100 if probe else 11924)
        assert r['epochs'][0]['rows']==(25600 if probe else 3052475)
        assert r['parameters_updated'] and r['reload_max_abs_error']==0
        with np.load(out/name/'validation_predictions.npz') as pred:
            assert len(pred['logits'])==768773 and np.isfinite(pred['logits']).all()
            for key in ('labels','source_rows','user_rows'):
                np.testing.assert_array_equal(pred[key],np.load(store/f'validation.{key}.npy',mmap_mode='r'))
        budget=r['epochs'][0]['validation_budget']
        assert budget['rows']==768773
        assert 0<=budget['mean_unique_positions']<=budget['mean_route_positions']<=50
        assert abs(budget['mean_route_positions']-budget['mean_unique_positions']-budget['mean_overlap_positions'])<1e-8
        if job.startswith('content_'):assert budget['mean_overlap_positions']==0
        state['probes_completed' if probe else 'completed'].append(job);state['active']=None;save()
        return r
    try:
        probes=[train(f'{m}-seed42',probe=True) for m in modes]
        estimate=9*max(r['training_seconds']*11924/100+r['epochs'][0]['validation_and_prediction_save_seconds'] for r in probes)
        state['estimated_training_hours']=estimate/3600
        if estimate>a.max_estimated_hours*3600:
            state['phase']='throughput_review_required';save();return
        state['phase']='training';save()
        for job in jobs:train(job)
        state['phase']='paired_comparisons';save()
        for seed in (42,2026,3407):
            for name,baseline,candidate in [('primary','content_separate','hybrid_separate'),('sharing','hybrid_separate','hybrid_shared')]:
                state['active']=f'{name}-seed{seed}';save()
                execute([sys.executable,str(root/'scripts/compare_validation.py'),
                         '--baseline',str(out/f'{baseline}-seed{seed}/validation_predictions.npz'),
                         '--candidate',str(out/f'{candidate}-seed{seed}/validation_predictions.npz'),
                         '--report',str(out/f'{name}-seed{seed}.json')],out/f'{name}-seed{seed}.log')
        state['phase']='completed';state['active']=None;save()
    except Exception as exc:
        state['phase']='failed';state['failed']={'job':state['active'],'error':str(exc)};save();raise


if __name__=='__main__':main()
