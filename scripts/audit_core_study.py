"""Validate completed core runs before fixed paired validation comparisons."""
import argparse,json,subprocess,sys,hashlib
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    names=[f'r{i}' for i in range(6)]
    status=json.loads((args.study/'status.json').read_text())
    if status['active'] or status['failed'] or set(status['completed'])!=set(names):
        raise ValueError('All six runs must complete successfully')
    reports={name:json.loads((args.study/name/'report.json').read_text()) for name in names}
    base=reports['r0']
    switches=[('recent',False,False,True),('cosine',False,False,True),('cosine',False,True,True),
              ('cosine',True,False,True),('cosine',True,True,True),('cosine',False,True,False)]
    for name,expected in zip(names,switches):
        r=reports[name];cfg=r['config']
        if tuple(cfg[k] for k in ['mode','semantic','histogram','long_id'])!=expected:
            raise ValueError('Ablation switches mismatch: '+name)
        if r['scope']!='one_seed_validation_run' or r['test_evaluated'] or not r['parameters_updated'] or r['reload_max_abs_error']!=0:
            raise ValueError('Run verification failed: '+name)
        if r['store_manifest_sha256']!=base['store_manifest_sha256'] or r['implementation_sha256']!=base['implementation_sha256']:
            raise ValueError('Source or data version mismatch: '+name)
        for key in ['seed','embedding_width','history_length','short_length','batch_size','epochs','k','dense_learning_rate','sparse_learning_rate','weight_decay']:
            if cfg[key]!=base['config'][key]:raise ValueError('Common configuration mismatch: '+name)
        if len(r['epochs'])!=1 or r['epochs'][0]['rows']!=609902 or r['validation']['rows']!=151947 or r['steps']!=19060:
            raise ValueError('Incomplete fixed-cohort run: '+name)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise ValueError('Fresh output directory required')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    pairs=[('r0','r1'),('r1','r2'),('r3','r4'),('r1','r3'),('r2','r4'),('r5','r2')]
    comparisons={}
    for baseline,candidate in pairs:
        key=candidate+'-vs-'+baseline;output=args.output_dir/(key+'.json')
        subprocess.run([sys.executable,str(Path(__file__).with_name('compare_validation.py')),
            '--baseline',str(args.study/baseline/'validation_predictions.npz'),
            '--candidate',str(args.study/candidate/'validation_predictions.npz'),'--report',str(output)],check=True,stdout=subprocess.DEVNULL)
        comparisons[key]=json.loads(output.read_text())
        print('Completed '+key,flush=True)
    result={'scope':'fixed_1_percent_single_seed_core_validation_audit',
            'store_manifest_sha256':base['store_manifest_sha256'],
            'source_report_sha256':{name:hashlib.sha256((args.study/name/'report.json').read_bytes()).hexdigest() for name in names},
            'metrics':{name:r['validation'] for name,r in reports.items()},'comparisons':comparisons,
            'test_evaluated':False,'limitations':['fixed fitted models and cohort; no seed variability',
                                               'six exploratory intervals without multiplicity correction',
                                               'fixed 1% cohort audit; not a general dataset-size checker']}
    (args.output_dir/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()
