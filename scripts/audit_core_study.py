"""Validate completed core runs before fixed paired validation comparisons."""
import argparse,json,subprocess,sys,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from metrics import evaluate


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--store',type=Path,help='Verify source-aligned saved predictions; required for cohorts other than legacy 1 percent')
    args=parser.parse_args()
    names=[f'r{i}' for i in range(6)]
    status=json.loads((args.study/'status.json').read_text())
    if status['active'] or status['failed'] or set(status['completed'])!=set(names):
        raise ValueError('All six runs must complete successfully')
    reports={name:json.loads((args.study/name/'report.json').read_text()) for name in names}
    base=reports['r0']
    train_rows,val_rows=609902,151947
    prediction_checks={}
    if args.store:
        path=args.store/'manifest.json'
        manifest=json.loads(path.read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest()!=base['store_manifest_sha256']:
            raise ValueError('Feature store manifest mismatch')
        train_rows=manifest['splits']['train']['rows'];val_rows=manifest['splits']['validation']['rows']
        expected_arrays={key:np.load(args.store/('validation.'+key+'.npy')) for key in ['labels','source_rows','user_rows']}
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
        if len(r['epochs'])!=1 or cfg['epochs']!=1 or r['epochs'][0]['rows']!=train_rows or r['validation']['rows']!=val_rows or r['steps']!=(train_rows+cfg['batch_size']-1)//cfg['batch_size']:
            raise ValueError('Incomplete fixed-cohort run: '+name)
        if args.store:
            prediction_path=args.study/name/'validation_predictions.npz'
            with np.load(prediction_path) as saved:
                for key,expected_array in expected_arrays.items():
                    np.testing.assert_array_equal(saved[key],expected_array)
                logits=saved['logits']
            if not np.isfinite(logits).all():raise ValueError('Nonfinite saved predictions')
            measured=evaluate(expected_arrays['labels'],logits,expected_arrays['user_rows'])
            if measured!=r['validation']:raise ValueError('Recomputed metric mismatch: '+name)
            prediction_checks[name]={'all_saved_rows_match_store':True,'all_metrics_recomputed_equal':True,
                'prediction_sha256':hashlib.sha256(prediction_path.read_bytes()).hexdigest()}
            print('Audited '+name,flush=True)
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
    result={'scope':'single_seed_core_validation_audit',
            'store_manifest_sha256':base['store_manifest_sha256'],
            'train_rows':train_rows,'validation_rows':val_rows,'saved_prediction_checks':prediction_checks,
            'audit_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'source_report_sha256':{name:hashlib.sha256((args.study/name/'report.json').read_bytes()).hexdigest() for name in names},
            'metrics':{name:r['validation'] for name,r in reports.items()},'comparisons':comparisons,
            'test_evaluated':False,'limitations':['fixed fitted models and cohort; no seed variability',
                                               'six exploratory intervals without multiplicity correction',
                                               'full source-aligned prediction verification requires --store',
                                               'validation and checkpoint selection uncertainty is not modeled']}
    (args.output_dir/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()
