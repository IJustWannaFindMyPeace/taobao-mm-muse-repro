"""Fixed two-stage G1 validation extension with audited reusable controls."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    for name in ('store', 'output-dir', 'scorer-probe', 'downstream-probe', 'controls'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--max-estimated-hours', type=float, default=6)
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = a.output_dir.resolve()
    store = a.store.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Fresh output required')
    out.mkdir(parents=True, exist_ok=True)
    seeds = (42, 2026, 3407)
    stages = ('scorer', 'downstream')
    jobs = [f'{stage}-seed{seed}' for seed in seeds for stage in stages]
    configs = {job: json.loads((root/f'configs/post-main-g1-{job}.json').read_text()) for job in jobs}
    files = [root/f'scripts/train_g1_{s}.py' for s in stages]
    files += [root/'scripts/probe_search.py', *sorted((root/'src').glob('*.py')), *sorted((root/'extensions').glob('*.py'))]
    hashes = {f.name: sha(f) for f in files}
    manifest = sha(store/'manifest.json')
    state = dict(phase='preflight', planned=jobs, completed=[], active=None, failed=None, test_evaluated=False)

    def save():
        tmp = out/'status.tmp'
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(out/'status.json')

    def check(run, stage, seed, probe=False, control=False):
        r = json.loads((run/'report.json').read_text())
        assert r['store_manifest_sha256'] == manifest and not r['test_evaluated']
        assert r['steps'] == (100 if probe else 11924)
        assert r['epochs'][0]['rows'] == (25600 if probe else 3052475)
        assert r['scope'] == ('partial_training_resource_probe' if probe else 'one_seed_validation_run')
        assert r['parameters_updated'] and r['reload_max_abs_error'] == 0
        expected = configs[f'{stage}-seed{seed}']
        if control:
            assert {k:v for k,v in r['config'].items() if k not in ('variant','experiment')} == {k:v for k,v in expected.items() if k not in ('variant','experiment')}
            assert r['config']['variant'] == 'content'
            for f in sorted((root/'src').glob('*.py')):
                assert r['implementation_sha256'][f.name] == hashes[f.name]
        else:
            assert r['config'] == expected
            relevant = {k:v for k,v in hashes.items() if not k.startswith('train_g1_') or k == f'train_g1_{stage}.py'}
            assert r['implementation_sha256'] == relevant
            assert r['projection_parameters_updated'] if stage == 'scorer' else r['projection_unchanged']
        with np.load(run/'validation_predictions.npz') as pred:
            assert len(pred['logits']) == 768773 and np.isfinite(pred['logits']).all()
            for key in ('labels','source_rows','user_rows'):
                np.testing.assert_array_equal(pred[key], np.load(store/f'validation.{key}.npy', mmap_mode='r'))
        return r

    def execute(command, log):
        with log.open('w', encoding='utf-8') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)

    save()
    try:
        probes = [check(a.scorer_probe, 'scorer', 42, True), check(a.downstream_probe, 'downstream', 42, True)]
        controls = {}
        for seed in seeds:
            run = a.controls.resolve()/f'content-seed{seed}'
            check(run, 'downstream', seed, control=True)
            controls[str(seed)] = {'run':str(run), 'report_sha256':sha(run/'report.json'), 'predictions_sha256':sha(run/'validation_predictions.npz')}
        estimate = 3 * sum(r['training_seconds']*11924/100 + r['epochs'][0]['validation_and_prediction_save_seconds'] for r in probes)
        state['estimated_training_hours'] = estimate/3600
        (out/'plan.json').write_text(json.dumps(dict(configs=configs, source_hashes=hashes, manifest_sha256=manifest, reused_controls=controls, primary='frozen learned content retrieval minus raw content cosine retrieval', pretraining_budget='one additional epoch per seed, reported separately', max_estimated_hours=a.max_estimated_hours), indent=2))
        if estimate > a.max_estimated_hours*3600:
            state['phase'] = 'throughput_review_required'; save(); return
        state['phase'] = 'training'; save()
        for seed in seeds:
            for stage in stages:
                job = f'{stage}-seed{seed}'
                state['active'] = job; save()
                assert all(sha(f) == hashes[f.name] for f in files)
                config = root/f'configs/post-main-g1-{job}.json'
                assert json.loads(config.read_text()) == configs[job]
                command = [sys.executable, str(root/f'scripts/train_g1_{stage}.py'), '--store', str(store), '--config', str(config), '--output-dir', str(out/job)]
                if stage == 'downstream':
                    command += ['--scorer-run', str(out/f'scorer-seed{seed}')]
                execute(command, out/f'{job}.log')
                report = check(out/job, stage, seed)
                if stage == 'downstream':
                    provenance = report['scorer_provenance']
                    assert provenance['checkpoint_sha256'] == sha(out/f'scorer-seed{seed}/model.pt')
                    assert provenance['report_sha256'] == sha(out/f'scorer-seed{seed}/report.json')
                state['completed'].append(job); state['active'] = None; save()
        state['phase'] = 'paired_comparisons'; save()
        for seed in seeds:
            state['active'] = f'primary-seed{seed}'; save()
            control = controls[str(seed)]
            baseline = Path(control['run'])/'validation_predictions.npz'
            assert sha(baseline) == control['predictions_sha256']
            execute([sys.executable, str(root/'scripts/compare_validation.py'), '--baseline', str(baseline), '--candidate', str(out/f'downstream-seed{seed}/validation_predictions.npz'), '--report', str(out/f'primary-seed{seed}.json')], out/f'primary-seed{seed}.log')
        state['phase'] = 'completed'; state['active'] = None; save()
    except Exception as exc:
        state['phase'] = 'failed'; state['failed'] = {'job':state['active'], 'error':repr(exc)}; save(); raise


if __name__ == '__main__':
    main()
