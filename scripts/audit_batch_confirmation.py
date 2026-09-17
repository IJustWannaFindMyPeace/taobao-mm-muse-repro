"""Audit selected training protocols; comparisons are validation diagnostics."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from metrics import evaluate


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for name in ['store','study','screen-audit','legacy-r4','legacy-c6','output-dir']:
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists():raise ValueError('Fresh audit directory required')
    state=json.loads((args.study/'status.json').read_text())
    if state['completed']!=state['planned'] or state['active'] or state['failed']:
        raise ValueError('Confirmation incomplete')
    if digest(args.screen_audit)!=state['selection_audit_sha256']:raise ValueError('Selection audit changed')
    paths={name:args.study/name for name in state['planned']}
    paths.update({'r4-b32-seed2026':args.legacy_r4,'c6-b32-seed42':args.legacy_c6})
    manifest=json.loads((args.store/'manifest.json').read_text())
    expected={k:np.load(args.store/('validation.'+k+'.npy')) for k in ['labels','source_rows','user_rows']}
    reports={};checks={};source_checks={}
    for name,path in paths.items():
        r=json.loads((path/'report.json').read_text());c=r['config']
        if r['store_manifest_sha256']!=digest(args.store/'manifest.json') or r['test_evaluated'] or r['reload_max_abs_error']!=0 or not r['parameters_updated']:
            raise ValueError('Run integrity failed')
        if c['epochs']!=1 or len(r['epochs'])!=1 or r['epochs'][0]['rows']!=manifest['splits']['train']['rows']:
            raise ValueError('Incomplete epoch')
        if r['steps']!=(manifest['splits']['train']['rows']+c['batch_size']-1)//c['batch_size']:
            raise ValueError('Step count mismatch')
        if not (path/'model.pt').exists():raise ValueError('Checkpoint missing')
        if name in state['planned'] and c!=json.loads((args.study/'configs'/(name+'.json')).read_text()):
            raise ValueError('Config snapshot mismatch')
        with np.load(path/'validation_predictions.npz') as saved:
            for k,v in expected.items():np.testing.assert_array_equal(saved[k],v)
            metrics=evaluate(expected['labels'],saved['logits'],expected['user_rows'])
        if metrics!=r['validation'] or metrics!=r['epochs'][0]['validation']:raise ValueError('Metric mismatch')
        reports[name]=r
        checks[name]={'metrics':metrics,'config':c,'report_sha256':digest(path/'report.json'),
            'prediction_sha256':digest(path/'validation_predictions.npz'),'all_rows_and_metrics_equal':True,
            'training_seconds':r['training_seconds'],'peak_allocated_mib':r['training_peak_allocated_mib']}
        print('Audited '+name,flush=True)
    for family,seed in [('r4',2026),('c6',42)]:
        old=reports[f'{family}-b32-seed{seed}']
        for batch in [256,512]:
            name=f'{family}-b{batch}-seed{seed}';new=reports[name]
            normalize=lambda c:{k:v for k,v in c.items() if k not in ['experiment','batch_size','dense_learning_rate','resource_limits']}
            if normalize(old['config'])!=normalize(new['config']):raise ValueError('Unexpected protocol changes')
            if new['config']['dense_learning_rate']!=.0002 or old['config']['dense_learning_rate']!=.001:
                raise ValueError('Unexpected learning rate')
            common=set(old['implementation_sha256'])&set(new['implementation_sha256'])-{'train_muse.py','train_c0.py'}
            required={'c0.py','muse_model.py','retrieval.py','enhanced_esu.py'} if family=='r4' else {'c0.py','make.py','make_fusion.py','simtier.py'}
            if not required<=common:raise ValueError('Missing model provenance')
            if any(old['implementation_sha256'][k]!=new['implementation_sha256'][k] for k in common):
                raise ValueError('Model source mismatch')
            source_checks[name]={'matching_recorded_dependencies':sorted(common),
                'training_entrypoint_changed':'resource monitoring and reporting; model dependencies unchanged'}
    args.output_dir.mkdir()
    comparisons={}
    for family,seed in [('r4',2026),('c6',42)]:
        for baseline,candidate in [(32,256),(32,512),(256,512)]:
            a=f'{family}-b{baseline}-seed{seed}';b=f'{family}-b{candidate}-seed{seed}'
            name=b+'-vs-'+a;out=args.output_dir/(name+'.json')
            subprocess.run([sys.executable,str(Path(__file__).with_name('compare_validation.py')),
                '--baseline',str(paths[a]/'validation_predictions.npz'),'--candidate',str(paths[b]/'validation_predictions.npz'),
                '--report',str(out)],check=True,stdout=subprocess.DEVNULL)
            comparisons[name]=json.loads(out.read_text());print('Compared '+name,flush=True)
    result={'scope':'validation_confirmation_of_selected_training_protocols','runs':checks,'comparisons':comparisons,
        'source_checks':source_checks,'selection_audit_sha256':digest(args.screen_audit),
        'store_manifest_sha256':digest(args.store/'manifest.json'),'audit_sha256':digest(Path(__file__)),
        'test_evaluated':False,'final_protocol_frozen':False,
        'limitations':['batch32 comparisons change both batch and dense learning rate',
            'same validation population used for tuning; different seed is not independent data confirmation',
            '2000 user bootstrap samples conditional on fitted models; six comparisons not multiplicity adjusted',
            'C6 uses unchanged batch32 content pretraining; downstream protocol only']}
    (args.output_dir/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()
