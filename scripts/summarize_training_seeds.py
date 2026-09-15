"""Summarize audited fixed-data training seeds without selecting a best seed."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--studies',nargs=3,type=Path,required=True)
    p.add_argument('--audits',nargs=3,type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    args=p.parse_args();runs={};provenance={}
    for study,audit_path in zip(args.studies,args.audits):
        audit=json.loads(audit_path.read_text());reports={}
        for i in range(6):
            name=f'r{i}';path=study/name/'report.json';r=json.loads(path.read_text())
            if digest(path)!=audit['source_report_sha256'][name]:raise ValueError('Report changed after audit')
            check=audit['saved_prediction_checks'][name]
            if not check['all_saved_rows_match_store'] or not check['all_metrics_recomputed_equal']:raise ValueError('Incomplete audit')
            if r['store_manifest_sha256']!=audit['store_manifest_sha256']:raise ValueError('Store mismatch')
            reports[name]=r
        seed=reports['r0']['config']['seed']
        if seed in runs or any(r['config']['seed']!=seed for r in reports.values()):raise ValueError('Duplicate/inconsistent seed')
        runs[seed]=reports;provenance[str(seed)]={'audit_sha256':digest(audit_path),'reports':audit['source_report_sha256']}
    if set(runs)!={42,2026,3407}:raise ValueError('Expected prespecified seeds 42, 2026, 3407')
    seeds=sorted(runs);base=runs[42]
    for reports in runs.values():
        for name,r in reports.items():
            strip=lambda cfg:{k:v for k,v in cfg.items() if k!='seed'}
            if strip(r['config'])!=strip(base[name]['config']) or r['implementation_sha256']!=base[name]['implementation_sha256'] or r['store_manifest_sha256']!=base[name]['store_manifest_sha256']:
                raise ValueError('Changes beyond training seed: '+name)
    def stats(values):return {'values':values,'mean':float(np.mean(values)),'sample_std':float(np.std(values,ddof=1))}
    models={name:{metric:stats([runs[s][name]['validation'][metric] for s in seeds]) for metric in ['auc','gauc','logloss']} for name in base}
    pairs=[('r1','r0'),('r2','r1'),('r3','r1'),('r4','r3'),('r4','r2'),('r2','r5')]
    deltas={a+'-minus-'+b:{metric:stats([runs[s][a]['validation'][metric]-runs[s][b]['validation'][metric] for s in seeds]) for metric in ['auc','gauc','logloss']} for a,b in pairs}
    result={'scope':'three_training_seed_fixed_cohort_summary','seeds':seeds,'models':models,'paired_seed_deltas':deltas,
        'provenance':provenance,'source_sha256':digest(Path(__file__)),'store_manifest_sha256':base['r0']['store_manifest_sha256'],
        'limitations':['three training seeds only; sample standard deviation is not a confidence interval',
        'same validation population across seeds; no cohort or test generalization evidence',
        'no significance test on three seeds; no best-seed selection'],'test_evaluated':False}
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({name:v['gauc'] for name,v in deltas.items()},indent=2))

if __name__=='__main__':main()
