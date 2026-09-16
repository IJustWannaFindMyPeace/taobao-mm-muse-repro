"""Queue isolated resource probes after a successful existing training study."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import psutil


def training_active():
    for process in psutil.process_iter(['pid','cmdline']):
        try:
            if process.pid != psutil.Process().pid and any(Path(x).name in ['train_muse.py','train_c0.py','train_content.py'] for x in (process.info['cmdline'] or [])):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--after-study', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--store', type=Path, required=True)
    p.add_argument('--content-project', type=Path, required=True)
    p.add_argument('--hist-cache', type=Path, required=True)
    p.add_argument('--content-checkpoint', type=Path, required=True)
    p.add_argument('--c6-config', type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists(): raise ValueError('Fresh directory required')
    args.output_dir.mkdir(parents=True)
    state = {'status':'waiting_for_baselines','active':None,'completed':[], 'resource_rejected':[],
             'models':['c6','r4','id_item_category'],'batch_sizes':[32,64,128,256],
             'quality_evaluation':False,'test_evaluated':False}
    def save():
        tmp=args.output_dir/'status.tmp'
        tmp.write_text(json.dumps(state,indent=2),encoding='utf-8');tmp.replace(args.output_dir/'status.json')
    save()
    try:
        while True:
            before=json.loads((args.after_study/'status.json').read_text())
            if before.get('failed'): raise RuntimeError('Preceding baseline study failed')
            if before['completed']==before['planned'] and before['active'] is None:
                break
            # A stale study file must not silently authorize overlapping work.
            if not training_active():
                time.sleep(5)
                if not training_active(): raise RuntimeError('Preceding study incomplete without a live training process')
            time.sleep(30)
        if training_active(): raise RuntimeError('Another training process is active')
        common=[]
        for name in ['store','content_project','hist_cache','content_checkpoint','c6_config']:
            common += ['--'+name.replace('_','-'),str(getattr(args,name).resolve())]
        results={}
        for model in state['models']:
            for batch in state['batch_sizes']:
                if training_active(): raise RuntimeError('Concurrent training detected; stopping batch scan')
                key=f'{model}-b{batch}'
                state.update(status='running',active=key);save()
                path=args.output_dir/(key+'.json')
                with (args.output_dir/(key+'.log')).open('w',encoding='utf-8') as out:
                    command=[sys.executable,str(Path(__file__).with_name('probe_batch_size.py')),*common,
                             '--model',model,'--batch-size',str(batch),'--report',str(path.resolve())]
                    subprocess.run(command,check=True,stdout=out,stderr=subprocess.STDOUT)
                result=json.loads(path.read_text())
                results[key]=result
                if result['status']=='resource_rejected':
                    state['resource_rejected'].append(key);save();break
                if result['status']!='passed': raise RuntimeError('Unexpected probe status')
                state['completed'].append(key);save()
        state.update(status='completed',active=None);save()
        (args.output_dir/'summary.json').write_text(json.dumps({'scope':'resource_screen_only',
            'runs':results,'next_step':'repeat feasible finalists, then fixed-validation learning-rate/quality evaluation',
            'final_batch_selected':False,'test_evaluated':False},indent=2),encoding='utf-8')
    except Exception as exc:
        state.update(status='failed',error=str(exc));save();raise


if __name__=='__main__':
    main()
