"""Run the fixed three-seed cosine-fusion control and audit full validation predictions."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from probe_search import digest
from metrics import evaluate


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser()
    for name in ('store','references','resource-report','output-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    protocol=read(root/'configs/shared-downstream-protocol-v1.json')
    resource=read(args.resource_report)
    if resource['status']!='passed' or resource['model']!='content_id_cosine' or resource['batch_size']!=protocol['batch_size']:
        raise ValueError('Resource probe required')
    for name,value in resource['implementation_sha256'].items():
        if digest(root/name)!=value:raise ValueError('Implementation changed since probe: '+name)
    if digest(args.store/'manifest.json')!=protocol['store_manifest_sha256']:
        raise ValueError('Data protocol mismatch')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Fresh output required')
    refs=read(args.references)
    configs={}
    for seed in protocol['training_seeds']:
        cfg=read(root/f'configs/shared-g-content-id-cosine-seed{seed}.json')
        for key in ('batch_size','epochs','dense_learning_rate','sparse_learning_rate','weight_decay','embedding_width','resource_limits'):
            if cfg[key]!=protocol[key]:raise ValueError('Common protocol changed')
        baseline=Path(refs[str(seed)]['run'])
        old=read(baseline/'report.json')
        audit=read(Path(refs[str(seed)]['audit']))
        entries=[r for r in audit['runs'].values() if r['report_sha256']==digest(baseline/'report.json')]
        if len(entries)!=1 or entries[0].get('prediction_sha256',entries[0].get('predictions_sha256'))!=digest(baseline/'validation_predictions.npz'):
            raise ValueError('Reference provenance mismatch')
        if old['test_evaluated'] or old['store_manifest_sha256']!=protocol['store_manifest_sha256']:
            raise ValueError('Reference data mismatch')
        common=lambda c:{k:v for k,v in c.items() if k not in ('experiment','model_variant')}
        if old['config'].get('model_variant','muse')!='muse' or common(old['config'])!=common(cfg):
            raise ValueError('Reference configuration mismatch')
        # Original reference runs instantiate MUSEModel, not the extended baseline class.
        # All old source files remain exact except the unused baseline dispatcher.
        for name,value in old['implementation_sha256'].items():
            source=root/('scripts' if name in ('train_muse.py','probe_search.py') else 'src')/name
            if name!='baseline_model.py' and digest(source)!=value:
                raise ValueError('Reference implementation changed: '+name)
        configs[str(seed)]=cfg
    args.output_dir.mkdir(parents=True)
    (args.output_dir/'plan.json').write_text(json.dumps({'configs':configs,'references':refs,
        'protocol':protocol,'resource_report_sha256':digest(args.resource_report),
        'source_hash_exception':'baseline_model.py adds unused branch for original MUSEModel; prediction/gradient equivalence tested'},indent=2))
    state={'planned':protocol['training_seeds'],'completed':[],'active':None,'failed':None,'test_evaluated':False}
    def save():
        tmp=args.output_dir/'status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(args.output_dir/'status.json')
    expected={k:np.load(args.store/f'validation.{k}.npy',mmap_mode='r') for k in ('labels','source_rows','user_rows')}
    comparisons={}
    for seed in state['planned']:
        state['active']=seed;save()
        try:
            output=args.output_dir/f'seed{seed}'
            config=args.output_dir/f'config-seed{seed}.json'
            config.write_text(json.dumps(configs[str(seed)],indent=2))
            with (args.output_dir/f'seed{seed}.log').open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,str(root/'scripts/train_muse.py'),'--store',str(args.store.resolve()),
                    '--config',str(config.resolve()),'--output-dir',str(output.resolve())],check=True,stdout=log,stderr=subprocess.STDOUT)
            r=read(output/'report.json')
            rows=read(args.store/'manifest.json')['splits']['train']['rows']
            if (r['config']!=configs[str(seed)] or r['test_evaluated'] or r['epochs'][0]['rows']!=rows
                or len(r['epochs'])!=1 or r['steps']!=(rows+255)//256 or r['reload_max_abs_error']!=0
                or not r['parameters_updated'] or r['store_manifest_sha256']!=protocol['store_manifest_sha256']):
                raise ValueError('Training integrity failed')
            with np.load(output/'validation_predictions.npz') as saved:
                for key,value in expected.items():np.testing.assert_array_equal(saved[key],value)
                if evaluate(expected['labels'],saved['logits'],expected['user_rows'])!=r['validation']:
                    raise ValueError('Saved metrics mismatch')
            comparison=args.output_dir/f'seed{seed}-vs-content.json'
            subprocess.run([sys.executable,str(root/'scripts/compare_validation.py'),
                '--baseline',str(Path(refs[str(seed)]['run'])/'validation_predictions.npz'),
                '--candidate',str(output/'validation_predictions.npz'),'--report',str(comparison)],check=True,stdout=subprocess.DEVNULL)
            comparisons[str(seed)]={'comparison':read(comparison),'report_sha256':digest(output/'report.json'),
                'all_rows_and_metrics_equal':True,'training_seconds':r['training_seconds'],
                'peak_allocated_mib':r['training_peak_allocated_mib']}
            state['completed'].append(seed);state['active']=None;save()
            print('Completed and audited seed '+str(seed),flush=True)
        except Exception as exc:
            state['failed']={'seed':seed,'error':str(exc)};state['active']=None;save();raise
    (args.output_dir/'summary.json').write_text(json.dumps({'scope':'fixed_cosine_fusion_three_seed_validation',
        'comparisons':comparisons,'test_evaluated':False,'plan_sha256':digest(args.output_dir/'plan.json'),
        'limitations':['one fixed fusion and missingness policy; not all complex retrievers',
            'no multiplicity correction; validation only; fixed user cohort']},indent=2))


if __name__=='__main__':main()
