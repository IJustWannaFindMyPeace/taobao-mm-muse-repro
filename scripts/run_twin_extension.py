"""Run the fixed six-run, validation-only four-head search extension serially."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--store',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--consistent-probe',type=Path,required=True)
    p.add_argument('--content-probe',type=Path,required=True)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    for mode,probe in [('consistent',a.consistent_probe),('content',a.content_probe)]:
        report=json.loads((probe/'report.json').read_text())
        assert report['scope']=='partial_training_resource_probe' and report['steps']==100
        assert report['config']==json.loads((root/f'configs/post-main-twin-{mode}-seed42.json').read_text())
        assert report['parameters_updated'] and report['reload_max_abs_error']==0
        assert not report['test_evaluated']
        assert report['store_manifest_sha256']==hashlib.sha256((a.store/'manifest.json').read_bytes()).hexdigest()
        for folder in ('src','extensions'):
            for source in (root/folder).glob('*.py'):
                assert report['implementation_sha256'][source.name]==hashlib.sha256(source.read_bytes()).hexdigest()
        source=root/'scripts/train_twin_extension.py'
        assert report['implementation_sha256'][source.name]==hashlib.sha256(source.read_bytes()).hexdigest()
    out=a.output_dir.resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Fresh output required')
    out.mkdir(parents=True,exist_ok=True)
    jobs=[f'{m}-seed{s}' for s in (42,2026,3407) for m in ('consistent','content')]
    state=dict(planned=jobs,completed=[],active=None,failed=None,test_evaluated=False)
    def save():
        tmp=out/'status.tmp'
        tmp.write_text(json.dumps(state,indent=2))
        tmp.replace(out/'status.json')
    expected_hash=hashlib.sha256((a.store/'manifest.json').read_bytes()).hexdigest()
    for job in jobs:
        state['active']=job;save()
        try:
            cfg=root/f'configs/post-main-twin-{job}.json'
            with (out/f'{job}.log').open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,str(root/'scripts/train_twin_extension.py'),
                    '--store',str(a.store.resolve()),'--config',str(cfg),
                    '--output-dir',str(out/job)],check=True,stdout=log,stderr=subprocess.STDOUT)
            r=json.loads((out/job/'report.json').read_text())
            assert r['config']==json.loads(cfg.read_text())
            assert r['store_manifest_sha256']==expected_hash
            assert r['scope']=='one_seed_validation_run' and not r['test_evaluated']
            assert r['steps']==11924 and r['epochs'][0]['rows']==3052475
            assert r['parameters_updated'] and r['reload_max_abs_error']==0
            budget=r['epochs'][0]['validation_budget']
            assert budget['rows']==768773
            assert 0<=budget['mean_unique_positions']<=budget['mean_head_positions']<=200
            if job.startswith('content-'):assert budget['mean_unique_positions']<=50
            import numpy as np
            with np.load(out/job/'validation_predictions.npz') as pred:
                assert len(pred['logits'])==768773 and np.isfinite(pred['logits']).all()
            state['completed'].append(job);state['active']=None;save()
            print('Completed and checked '+job,flush=True)
        except Exception as exc:
            state['failed']={'job':job,'error':str(exc)};state['active']=None;save()
            raise


if __name__=='__main__':main()

