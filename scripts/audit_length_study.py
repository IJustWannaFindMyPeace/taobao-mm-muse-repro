"""Verify length-only controls and source-aligned predictions before paired comparisons."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from metrics import evaluate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--store',type=Path,required=True)
    parser.add_argument('--core',type=Path,required=True)
    parser.add_argument('--length',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Fresh output directory required')
    manifest=json.loads((args.store/'manifest.json').read_text())
    expected={key:np.load(args.store/('validation.'+key+'.npy')) for key in ['labels','source_rows','user_rows']}
    paths={name:args.core/name for name in ['r1','r4']}
    paths.update({name:args.length/name for name in ['r1-l1000','r4-l1000']})
    reports={}
    checks={}
    for name,path in paths.items():
        r=json.loads((path/'report.json').read_text());cfg=r['config']
        if r['store_manifest_sha256']!=digest(args.store/'manifest.json') or r['test_evaluated'] or r['reload_max_abs_error']!=0 or not r['parameters_updated']:
            raise ValueError('Run integrity failed: '+name)
        if cfg['epochs']!=1 or len(r['epochs'])!=1 or r['epochs'][0]['rows']!=manifest['splits']['train']['rows']:
            raise ValueError('Incomplete training: '+name)
        if r['steps']!=(manifest['splits']['train']['rows']+cfg['batch_size']-1)//cfg['batch_size']:
            raise ValueError('Incomplete training steps')
        with np.load(path/'validation_predictions.npz') as saved:
            for key,value in expected.items():np.testing.assert_array_equal(saved[key],value)
            logits=saved['logits']
        measured=evaluate(expected['labels'],logits,expected['user_rows'])
        if measured!=r['validation']:raise ValueError('Metric mismatch: '+name)
        reports[name]=r
        checks[name]={'report_sha256':digest(path/'report.json'),'prediction_sha256':digest(path/'validation_predictions.npz'),
            'all_rows_and_recomputed_metrics_equal':True,'metrics':measured,
            'training_seconds':r['training_seconds'],'peak_allocated_mib':r['training_peak_allocated_mib']}
        print('Audited '+name,flush=True)
    for name in ['r1','r4']:
        a=reports[name];b=reports[name+'-l1000']
        if a['config']['history_length']!=200 or b['config']['history_length']!=1000:
            raise ValueError('Unexpected length endpoints')
        normalize=lambda cfg:{k:v for k,v in cfg.items() if k not in ['history_length','experiment']}
        if normalize(a['config'])!=normalize(b['config']) or a['implementation_sha256']!=b['implementation_sha256']:
            raise ValueError('Changes beyond history length: '+name)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    comparisons={}
    for baseline,candidate in [('r1','r1-l1000'),('r4','r4-l1000'),('r1-l1000','r4-l1000')]:
        name=candidate+'-vs-'+baseline
        output=args.output_dir/(name+'.json')
        subprocess.run([sys.executable,str(Path(__file__).with_name('compare_validation.py')),
            '--baseline',str(paths[baseline]/'validation_predictions.npz'),
            '--candidate',str(paths[candidate]/'validation_predictions.npz'),'--report',str(output)],check=True,stdout=subprocess.DEVNULL)
        comparisons[name]=json.loads(output.read_text())
        print('Compared '+name,flush=True)
    result={'scope':'single_seed_length_endpoint_validation_audit','store_manifest_sha256':digest(args.store/'manifest.json'),
        'audit_source_sha256':digest(Path(__file__)),'runs':checks,'comparisons':comparisons,'test_evaluated':False,
        'limitations':['fixed trained models and cohort; not training-seed variability',
            'three intervals without multiplicity correction','two length endpoints, not a complete length sweep']}
    (args.output_dir/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()
