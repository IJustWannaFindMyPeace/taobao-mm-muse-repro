"""Shared-protocol long-history and retrieval controls with audited reuse."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from run_shared_core import digest, read


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--store',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--reuse',type=Path,required=True)
    parser.add_argument('--reuse-audit',type=Path,required=True)
    parser.add_argument('--seed',type=int,help='Override training seed only; data split remains fixed')
    args=parser.parse_args()
    args.configs=['r1-l1000','r4-l1000','b1-hard-enhanced','b2-soft-enhanced','g-id-item']
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise ValueError('Fresh directory required')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    if len(set(args.configs))!=len(args.configs):raise ValueError('Duplicate configurations')
    for name in args.configs:
        if Path(name).name!=name or not (root/'configs'/(name+'.json')).is_file():raise ValueError('Unknown configuration: '+name)
    if args.seed is not None and args.seed<0:raise ValueError('Nonnegative seed required')
    protocol=read(root/'configs/shared-downstream-protocol-v1.json')
    if args.seed not in protocol['training_seeds'] or digest(args.store/'manifest.json')!=protocol['store_manifest_sha256']:raise ValueError('Protocol mismatch')
    snapshot=args.output_dir/'configs';snapshot.mkdir()
    configs={}
    for name in args.configs:
        cfg=json.loads((root/'configs'/(name+'.json')).read_text())
        cfg.update({k:protocol[k] for k in ['batch_size','dense_learning_rate','sparse_learning_rate','weight_decay','epochs','embedding_width','resource_limits']})
        cfg['seed']=args.seed
        configs[name]=cfg
        (snapshot/(name+'.json')).write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    previous=read(args.reuse/'report.json');audit=read(args.reuse_audit)
    normalize=lambda c:{k:v for k,v in c.items() if k!='experiment'}
    if normalize(previous['config'])!=normalize(configs['r4-l1000']):raise ValueError('Reuse config mismatch')
    entries=[v for v in audit['runs'].values() if v['report_sha256']==digest(args.reuse/'report.json')]
    if len(entries)!=1 or not entries[0]['all_rows_and_metrics_equal'] or entries[0].get('prediction_sha256',entries[0].get('predictions_sha256'))!=digest(args.reuse/'validation_predictions.npz'):
        raise ValueError('Reuse audit mismatch')
    if audit['test_evaluated'] or audit['store_manifest_sha256']!=protocol['store_manifest_sha256'] or not (args.reuse/'model.pt').exists():raise ValueError('Reuse provenance mismatch')
    for name,value in previous['implementation_sha256'].items():
        path=root/('scripts' if name in ['train_muse.py','probe_search.py'] else 'src')/name
        if digest(path)!=value:raise ValueError('Implementation changed: '+name)
    (args.output_dir/'plan.json').write_text(json.dumps({'protocol':protocol,'seed':args.seed,'reuse':{'r4-l1000':str(args.reuse.resolve())},'reuse_audit_sha256':digest(args.reuse_audit),'configs':configs},indent=2),encoding='utf-8')
    state={'reused':[], 'planned':args.configs,'completed':[],'active':None,'failed':None}
    def save():
        tmp=args.output_dir/'status.tmp'
        tmp.write_text(json.dumps(state,indent=2));tmp.replace(args.output_dir/'status.json')
    for name in state['planned']:
        state['active']=name;save();print('Starting '+name,flush=True)
        if name=='r4-l1000':
            state['reused'].append(name);state['completed'].append(name);state['active']=None;save();continue
        with (args.output_dir/(name+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,str(root/'scripts/train_muse.py'),
                '--store',str(args.store.resolve()),'--config',str((snapshot/(name+'.json')).resolve()),
                '--output-dir',str((args.output_dir/name).resolve())],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            state['failed']=name;state['active']=None;save()
            raise RuntimeError(name+' failed; later runs were not started')
        report=read(args.output_dir/name/'report.json')
        if report['test_evaluated'] or report['config']!=configs[name] or report['reload_max_abs_error']!=0 or report['store_manifest_sha256']!=protocol['store_manifest_sha256']:raise ValueError('Completed run verification failed')
        state['completed'].append(name);state['active']=None;save()
        print('Completed '+name,flush=True)


if __name__=='__main__':main()
