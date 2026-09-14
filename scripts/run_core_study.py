"""Serial R0-R5 controlled full-epoch validation runs."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--store',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--configs',nargs='+',default=['r0','r1','r2','r3','r4','r5'])
    parser.add_argument('--seed',type=int,help='Override training seed only; data split remains fixed')
    args=parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise ValueError('Fresh directory required')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    if len(set(args.configs))!=len(args.configs):raise ValueError('Duplicate configurations')
    for name in args.configs:
        if Path(name).name!=name or not (root/'configs'/(name+'.json')).is_file():raise ValueError('Unknown configuration: '+name)
    if args.seed is not None and args.seed<0:raise ValueError('Nonnegative seed required')
    snapshot=args.output_dir/'configs';snapshot.mkdir()
    for name in args.configs:
        cfg=json.loads((root/'configs'/(name+'.json')).read_text())
        if args.seed is not None:cfg['seed']=args.seed
        (snapshot/(name+'.json')).write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    state={'planned':args.configs,'completed':[],'active':None,'failed':None}
    def save():
        tmp=args.output_dir/'status.tmp'
        tmp.write_text(json.dumps(state,indent=2));tmp.replace(args.output_dir/'status.json')
    for name in state['planned']:
        state['active']=name;save();print('Starting '+name,flush=True)
        with (args.output_dir/(name+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,str(root/'scripts/train_muse.py'),
                '--store',str(args.store.resolve()),'--config',str((snapshot/(name+'.json')).resolve()),
                '--output-dir',str((args.output_dir/name).resolve())],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            state['failed']=name;state['active']=None;save()
            raise RuntimeError(name+' failed; later runs were not started')
        state['completed'].append(name);state['active']=None;save()
        print('Completed '+name,flush=True)


if __name__=='__main__':main()
