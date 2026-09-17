"""Audit shared batch-256 protocol against lower LR and legacy batch-32 controls."""
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
    for name in ['store','study','reference-audit','reference-study','legacy-r4','legacy-c6','output-dir']:
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists():raise ValueError('Fresh audit directory required')
    state=json.loads((args.study/'status.json').read_text())
    if state['completed']!=state['planned'] or state['active'] or state['failed']:
        raise ValueError('Confirmation incomplete')
    if digest(args.reference_audit)!=state['reference_audit_sha256']:raise ValueError('Selection audit changed')
    paths={name:args.study/name for name in state['planned']}
    paths.update({'r4-b32-seed2026':args.legacy_r4,'c6-b32-seed42':args.legacy_c6})
    paths.update({name:args.reference_study/name for name in ['r4-b256-seed2026','c6-b256-seed42']})
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
    pairs=[]
    for family,seed in [('r4',2026),('c6',42)]:
        candidate=f'{family}-b256-d001-seed{seed}'
        for baseline,variable in [(f'{family}-b32-seed{seed}','batch_size'),
                                  (f'{family}-b256-seed{seed}','dense_learning_rate')]:
            old,new=reports[baseline],reports[candidate]
            normalize=lambda c:{k:v for k,v in c.items() if k not in ['experiment',variable,'resource_limits']}
            if normalize(old['config'])!=normalize(new['config']):raise ValueError('Unexpected protocol difference')
            common=set(old['implementation_sha256'])&set(new['implementation_sha256'])-{'train_muse.py','train_c0.py'}
            required={'c0.py','muse_model.py','retrieval.py','enhanced_esu.py'} if family=='r4' else {'c0.py','make.py','make_fusion.py','simtier.py'}
            if not required<=common or any(old['implementation_sha256'][k]!=new['implementation_sha256'][k] for k in common):
                raise ValueError('Model provenance mismatch')
            key=candidate+'-vs-'+baseline
            source_checks[key]={'intended_variable':variable,'matching_recorded_dependencies':sorted(common)}
            pairs.append((baseline,candidate))
    args.output_dir.mkdir()
    comparisons={}
    for a,b in pairs:
        name=b+'-vs-'+a;out=args.output_dir/(name+'.json')
        subprocess.run([sys.executable,str(Path(__file__).with_name('compare_validation.py')),
            '--baseline',str(paths[a]/'validation_predictions.npz'),'--candidate',str(paths[b]/'validation_predictions.npz'),
            '--report',str(out)],check=True,stdout=subprocess.DEVNULL)
        comparisons[name]=json.loads(out.read_text());print('Compared '+name,flush=True)
    result={'scope':'shared_training_protocol_validation_audit','runs':checks,'comparisons':comparisons,
        'source_checks':source_checks,'selection_audit_sha256':digest(args.reference_audit),
        'store_manifest_sha256':digest(args.store/'manifest.json'),'audit_sha256':digest(Path(__file__)),
        'test_evaluated':False,'final_protocol_frozen':False,
        'limitations':['batch32 comparisons change batch only; batch256 comparisons change dense LR only',
            'same validation population used for tuning; different seed is not independent data confirmation',
            '2000 user bootstrap samples conditional on fitted models; four comparisons not multiplicity adjusted',
            'C6 uses unchanged batch32 content pretraining; downstream protocol only']}
    (args.output_dir/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()
