"""Complete fixed four-point history curves for three seeds, validation only."""
import argparse,json,subprocess,sys,time,hashlib
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
 a=p.parse_args();w=a.work.resolve();out=a.output_dir.resolve();root=Path(__file__).resolve().parents[1];store=w/'shared-feature-store-050'
 if out.exists():raise ValueError('Fresh output required')
 out.mkdir();(out/'configs').mkdir();manifest=sha(store/'manifest.json')
 files=[root/'scripts/train_muse.py',root/'scripts/probe_search.py',*sorted((root/'src').glob('*.py'))];hashes={p.name:sha(p) for p in files}
 # The only historical change is an unused BaselineModel branch added in f36e51b.
 old_baseline='351a8650722209658e75baa28a54e2e6e554fd5b505a5e025a950bdc34625f61'
 new_baseline='1960031fd791cee17d6a6ae0ada983a70962e4cd045f632bf852fd4a0c608690'
 assert hashes['baseline_model.py']==new_baseline
 def source_check(r):
  h=r['implementation_sha256']
  for k,v in h.items():
   if k=='baseline_model.py':assert v in (old_baseline,new_baseline)
   else:assert hashes[k]==v,(k,'source mismatch')
  assert set(hashes)-set(h)<= {'fused_retrieval.py'}
  assert r['config'].get('model_variant','muse')=='muse'
 inventory={x['id']:x for x in read(w/'final-evaluation-inventory-v1.json')['models']}
 paths={};configs={};reference={}
 for seed in (42,2026,3407):
  for family in ('r1','r4'):
   for length in (200,1000):
    oldkey=f'{family}-seed{seed}' if length==200 else f'{family}-l1000-seed{seed}'
    entry=inventory[oldkey];folder=Path(entry['run']);assert sha(folder/'report.json')==entry['report_sha256']
    key=f'{family}-l{length}-seed{seed}';paths[key]=folder
   for length in (50,500):
    key=f'{family}-l{length}-seed{seed}'
    if seed==42:paths[key]=w/f'shared-length-sensitivity-v1-seed42/shared-{family}-l{length}'
    else:
     base=read(paths[f'{family}-l200-seed{seed}']/'report.json')['config']
     configs[key]=dict(base,history_length=length,experiment=f'post-main-{key}');paths[key]=out/key
 state=dict(phase='preflight',completed=[],active=None,failed=None,test_evaluated=False)
 def save():
  t=out/'status.tmp';t.write_text(json.dumps(state,indent=2));t.replace(out/'status.json')
 def verify(key):
  folder=paths[key];r=read(folder/'report.json');source_check(r)
  assert r['store_manifest_sha256']==manifest and not r['test_evaluated'] and r['steps']==11924
  assert r['parameters_updated'] and r['reload_max_abs_error']==0 and r['epochs'][0]['rows']==3052475
  assert r['config']['batch_size']==256 and r['config']['k']==50
  with np.load(folder/'validation_predictions.npz') as pred:
   assert len(pred['logits'])==768773 and np.isfinite(pred['logits']).all()
   for k in ('labels','source_rows','user_rows'):np.testing.assert_array_equal(pred[k],np.load(store/f'validation.{k}.npy',mmap_mode='r'))
  reference[key]=dict(report_sha256=sha(folder/'report.json'),predictions_sha256=sha(folder/'validation_predictions.npz'),run=str(folder))
  return r
 def execute(name,command):
  assert all(sha(f)==hashes[f.name] for f in files)
  state['active']=name;save();started=time.time()
  with (out/f'{name}.log').open('w',encoding='utf-8') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
  state['completed'].append(dict(job=name,seconds=time.time()-started));state['active']=None;save()
 try:
  for key in paths:
   if key not in configs:verify(key)
  estimate=2*sum(read(paths[f'{f}-l{l}-seed42']/'report.json')['run_seconds_since_training_start'] for f in ('r1','r4') for l in (50,500))/3600
  assert estimate<6,'Throughput review required'
  plan=dict(configs=configs,reused=reference.copy(),source_hashes=hashes,manifest_sha256=manifest,estimated_hours=estimate,source_exception='baseline_model.py unused content_id_cosine branch plus fused_retrieval.py import; MUSEModel execution source unchanged',test_evaluated=False)
  (out/'plan.json').write_text(json.dumps(plan,indent=2));state['phase']='training';save()
  for key,cfg in configs.items():
   config=out/'configs'/f'{key}.json';config.write_text(json.dumps(cfg,indent=2))
   execute(key,[sys.executable,str(root/'scripts/train_muse.py'),'--store',str(store),'--config',str(config),'--output-dir',str(paths[key])])
   assert verify(key)['config']==cfg
  state['phase']='comparisons';save()
  for seed in (42,2026,3407):
   for family in ('r1','r4'):
    reports=[read(paths[f'{family}-l{l}-seed{seed}']/'report.json') for l in (50,200,500,1000)]
    normalize=lambda c:{k:v for k,v in c.items() if k not in ('experiment','history_length')}
    assert all(normalize(r['config'])==normalize(reports[0]['config']) for r in reports)
    for lo,hi in ((50,200),(200,500),(500,1000)):
     left=f'{family}-l{lo}-seed{seed}';right=f'{family}-l{hi}-seed{seed}';name=f'{family}-l{hi}-vs-l{lo}-seed{seed}'
     for key in (left,right):assert sha(paths[key]/'validation_predictions.npz')==reference[key]['predictions_sha256']
     execute(name,[sys.executable,str(root/'scripts/compare_validation.py'),'--baseline',str(paths[left]/'validation_predictions.npz'),'--candidate',str(paths[right]/'validation_predictions.npz'),'--report',str(out/f'{name}.json')])
  (out/'verified-runs.json').write_text(json.dumps(reference,indent=2));state['phase']='completed';save()
 except Exception as e:
  state.update(phase='failed',failed=dict(job=state['active'],error=repr(e)));save();raise
if __name__=='__main__':main()
