"""Confirm audited candidates on a second R4 seed and the companion C6 model."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for name in ['audit','screen','store','output-dir','content-project','c6-config','content-checkpoint','hist-cache']:
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists():raise ValueError('Fresh directory required')
    audit=json.loads(args.audit.read_text())
    if audit['store_manifest_sha256']!=digest(args.store/'manifest.json') or audit['test_evaluated']:
        raise ValueError('Audit data mismatch')
    selected=audit['selected_for_confirmation']
    if set(selected)!={'256','512'}:raise ValueError('Unexpected candidates')
    configs={};kinds={}
    base=json.loads(args.c6_config.read_text())
    if digest(args.content_checkpoint)!=base['content_checkpoint_sha256']:
        raise ValueError('Content checkpoint mismatch')
    for batch in [256,512]:
        source=selected[str(batch)];entry=audit['runs'][source]
        if not entry['all_rows_and_metrics_equal'] or digest(args.screen/source/'report.json')!=entry['report_sha256']:
            raise ValueError('Selected report changed after audit')
        original=entry['config']
        name=f'r4-b{batch}-seed2026'
        configs[name]=dict(original,seed=2026,experiment=name);kinds[name]='muse'
    for batch in [256,512]:
        original=audit['runs'][selected[str(batch)]]['config']
        name=f'c6-b{batch}-seed42'
        configs[name]=dict(base,experiment=name,batch_size=batch,
            sparse_learning_rate=original['sparse_learning_rate'],dense_learning_rate=original['dense_learning_rate'],
            resource_limits=original['resource_limits'])
        if configs[name]['seed']!=42:raise ValueError('Expected companion seed 42')
        kinds[name]='content'
    args.output_dir.mkdir();(args.output_dir/'configs').mkdir()
    for name,cfg in configs.items():
        (args.output_dir/'configs'/(name+'.json')).write_text(json.dumps(cfg,indent=2))
    state={'planned':list(configs),'completed':[],'active':None,'failed':None,
           'selection_audit_sha256':digest(args.audit),'test_evaluated':False}
    def save():
        tmp=args.output_dir/'status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(args.output_dir/'status.json')
    save()
    for name in configs:
        state['active']=name;save()
        script=Path(__file__).with_name('train_muse.py') if kinds[name]=='muse' else args.content_project/'scripts/train_c0.py'
        command=[sys.executable,str(script.resolve()),'--store',str(args.store.resolve()),
                 '--config',str((args.output_dir/'configs'/(name+'.json')).resolve()),
                 '--output-dir',str((args.output_dir/name).resolve())]
        if kinds[name]=='content':command+=['--hist-cache',str(args.hist_cache.resolve()),'--content-checkpoint',str(args.content_checkpoint.resolve())]
        with (args.output_dir/(name+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            state.update(active=None,failed=name);save();raise RuntimeError(name+' failed')
        state['completed'].append(name);state['active']=None;save()


if __name__=='__main__':main()
