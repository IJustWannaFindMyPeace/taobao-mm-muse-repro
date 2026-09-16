"""Repeat batch finalists with longer timing, then conditionally explore larger batches."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--initial-scan',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    for name in ['store','content-project','hist-cache','content-checkpoint','c6-config']:
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if args.output_dir.exists():raise ValueError('Fresh output directory required')
    args.output_dir.mkdir(parents=True)
    state={'status':'waiting','active':None,'completed':[],'skipped':[],
           'resource_rejected':[],'final_batch_selected':False}
    def save():
        temp=args.output_dir/'status.tmp'
        temp.write_text(json.dumps(state,indent=2));temp.replace(args.output_dir/'status.json')
    save()
    try:
        while True:
            first=json.loads((args.initial_scan/'status.json').read_text())
            if first['status']=='failed':raise RuntimeError('Initial scan failed')
            if first['status']=='completed':break
            time.sleep(30)
        initial=json.loads((args.initial_scan/'summary.json').read_text())['runs']
        common=[]
        for name in ['store','content_project','hist_cache','content_checkpoint','c6_config']:
            common.extend(['--'+name.replace('_','-'),str(getattr(args,name).resolve())])
        from run_batch_scan import training_active
        results={}
        for model in ['c6','r4','id_item_category']:
            for batch in [256,128,512,1024]:
                key=f'{model}-b{batch}'
                if batch in [256,128]:
                    prior=initial.get(key,{})
                    if prior.get('status')!='passed':
                        state['skipped'].append({'run':key,'reason':'initial resource screen did not pass'});save();continue
                else:
                    previous=results.get(f'{model}-b{batch//2}',{})
                    lower=results.get(f'{model}-b{batch//4}',{})
                    if (previous.get('status')!='passed' or lower.get('status')!='passed' or
                        previous['measured_rows_per_second'] < lower['measured_rows_per_second']*1.05 or
                        previous['peak_allocated_gib']>=8.8):
                        state['skipped'].append({'run':key,'reason':'previous resource headroom or throughput gain insufficient'});save();continue
                if training_active():raise RuntimeError('Concurrent training detected')
                state.update(status='running',active=key);save()
                report=args.output_dir/(key+'.json')
                with (args.output_dir/(key+'.log')).open('w',encoding='utf-8') as log:
                    subprocess.run([sys.executable,str(Path(__file__).with_name('probe_batch_size.py')),
                        *common,'--model',model,'--batch-size',str(batch),'--measured-rows','262144',
                        '--report',str(report.resolve())],check=True,stdout=log,stderr=subprocess.STDOUT)
                result=json.loads(report.read_text());results[key]=result
                if result['status']=='resource_rejected':state['resource_rejected'].append(key)
                elif result['status']=='passed':state['completed'].append(key)
                else:raise RuntimeError('Unexpected probe status')
                save()
        summary={'scope':'resource_repeat_and_conditional_expansion_not_quality_selection','runs':results,
                 'initial_scan':str(args.initial_scan),'skipped':state['skipped'],
                 'final_batch_selected':False,'validation_evaluated':False,'test_evaluated':False}
        (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2))
        state.update(status='completed',active=None);save()
    except Exception as exc:
        state.update(status='failed',error=str(exc));save();raise


if __name__=='__main__':main()
