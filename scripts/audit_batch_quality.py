"""Recompute the full validation screen before selecting one candidate per batch."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from metrics import evaluate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--store',type=Path,required=True)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    args=p.parse_args()
    if args.report.exists():raise ValueError('Fresh audit report required')
    protocol=json.loads(args.protocol.read_text())
    state=json.loads((args.study/'status.json').read_text())
    if state['planned']!=protocol['configs'] or state['completed']!=state['planned'] or state['active'] or state['failed']:
        raise ValueError('Screen incomplete or differs from protocol')
    manifest=json.loads((args.store/'manifest.json').read_text())
    expected={key:np.load(args.store/('validation.'+key+'.npy')) for key in ['labels','source_rows','user_rows']}
    results={}; common=None; implementation=None; grid=set()
    for name in protocol['configs']:
        run=args.study/name; r=json.loads((run/'report.json').read_text());cfg=r['config']
        if cfg!=json.loads((args.study/'configs'/(name+'.json')).read_text()):raise ValueError('Configuration snapshot mismatch')
        if r['store_manifest_sha256']!=digest(args.store/'manifest.json') or r['test_evaluated'] or r['reload_max_abs_error']!=0 or not r['parameters_updated']:
            raise ValueError('Run integrity failure')
        if cfg['seed']!=protocol['seed'] or cfg['epochs']!=1 or len(r['epochs'])!=1:
            raise ValueError('Unexpected seed/epoch protocol')
        if r['epochs'][0]['rows']!=manifest['splits']['train']['rows'] or r['steps']!=(manifest['splits']['train']['rows']+cfg['batch_size']-1)//cfg['batch_size']:
            raise ValueError('Incomplete training')
        if not (run/'model.pt').is_file():raise ValueError('Missing checkpoint')
        fixed={k:v for k,v in cfg.items() if k not in ['experiment','batch_size','dense_learning_rate','sparse_learning_rate']}
        if common is None:common=fixed;implementation=r['implementation_sha256']
        if fixed!=common or r['implementation_sha256']!=implementation:raise ValueError('Changes beyond grid variables')
        grid.add((cfg['batch_size'],cfg['sparse_learning_rate'],cfg['dense_learning_rate']))
        with np.load(run/'validation_predictions.npz') as saved, np.load(run/'validation-epoch-1.npz') as epoch:
            for key,value in expected.items():
                np.testing.assert_array_equal(saved[key],value)
                np.testing.assert_array_equal(epoch[key],value)
            np.testing.assert_array_equal(saved['logits'],epoch['logits'])
            measured=evaluate(expected['labels'],saved['logits'],expected['user_rows'])
        if measured!=r['validation'] or measured!=r['epochs'][0]['validation']:raise ValueError('Metric mismatch')
        results[name]={'config':cfg,'metrics':measured,'report_sha256':digest(run/'report.json'),
            'predictions_sha256':digest(run/'validation_predictions.npz'),'all_rows_and_metrics_equal':True,
            'training_seconds':r['training_seconds'],'run_seconds_since_training_start':r['run_seconds_since_training_start'],
            'peak_allocated_mib':r['training_peak_allocated_mib'],'resource_monitor':r['resource_monitor']}
        print('Audited '+name,flush=True)
    required={(b,s,d) for b in protocol['batch_sizes'] for s in protocol['sparse_learning_rates'] for d in protocol['dense_learning_rates']}
    if grid!=required or len(results)!=len(required):raise ValueError('Grid mismatch')
    selected={str(b):min((n for n,r in results.items() if r['config']['batch_size']==b),
                        key=lambda n:(-results[n]['metrics']['gauc'],results[n]['metrics']['logloss'],n))
              for b in protocol['batch_sizes']}
    output={'scope':'audited_validation_hyperparameter_screen','store_manifest_sha256':digest(args.store/'manifest.json'),
        'protocol_sha256':digest(args.protocol),'audit_source_sha256':digest(Path(__file__)),
        'implementation_sha256':implementation,'runs':results,'selected_for_confirmation':selected,
        'selection_rule':'maximum GAUC within each batch; ties lower LogLoss then configuration name',
        'test_evaluated':False,'final_batch_selected':False,
        'limitations':['validation reused for selection; best values have selection bias',
                       'single seed; independent seed and companion-model confirmation pending',
                       'checkpoint presence checked; weights not rehashed or replayed by this audit']}
    args.report.write_text(json.dumps(output,indent=2),encoding='utf-8')


if __name__=='__main__':main()
