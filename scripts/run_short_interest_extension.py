"""Run the fixed six-run, validation-only short-interest extension serially."""
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
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    out=a.output_dir.resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Fresh output required')
    out.mkdir(parents=True,exist_ok=True)
    jobs=[f'{m}-seed{s}' for s in (42,2026,3407) for m in ('din_lau','dot_softmax')]
    state=dict(planned=jobs,completed=[],active=None,failed=None,test_evaluated=False)
    def save():
        tmp=out/'status.tmp'
        tmp.write_text(json.dumps(state,indent=2))
        tmp.replace(out/'status.json')
    expected_hash=hashlib.sha256((a.store/'manifest.json').read_bytes()).hexdigest()
    for job in jobs:
        state['active']=job;save()
        try:
            cfg=root/f'configs/post-main-short-{job}.json'
            with (out/f'{job}.log').open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,str(root/'scripts/train_short_interest.py'),
                    '--store',str(a.store.resolve()),'--config',str(cfg),
                    '--output-dir',str(out/job)],check=True,stdout=log,stderr=subprocess.STDOUT)
            r=json.loads((out/job/'report.json').read_text())
            assert r['config']==json.loads(cfg.read_text())
            assert r['store_manifest_sha256']==expected_hash
            assert r['scope']=='one_seed_validation_run' and not r['test_evaluated']
            assert r['steps']==11924 and r['epochs'][0]['rows']==3052475
            assert r['parameters_updated'] and r['reload_max_abs_error']==0
            import numpy as np
            with np.load(out/job/'validation_predictions.npz') as pred:
                assert len(pred['logits'])==768773 and np.isfinite(pred['logits']).all()
            state['completed'].append(job);state['active']=None;save()
            print('Completed and checked '+job,flush=True)
        except Exception as exc:
            state['failed']={'job':job,'error':str(exc)};state['active']=None;save()
            raise


if __name__=='__main__':main()
