"""Bounded check of dense LR 0.001 at batch 256 on two companion models."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for key in ['audit', 'store', 'output-dir', 'content-project',
                'content-checkpoint', 'hist-cache']:
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError('Fresh output directory required')
    audit = json.loads(args.audit.read_text(encoding='utf-8'))
    if audit['test_evaluated'] or audit['store_manifest_sha256'] != digest(args.store/'manifest.json'):
        raise ValueError('Data provenance mismatch')
    configurations = {}
    for family, seed in [('r4', 2026), ('c6', 42)]:
        entry = audit['runs'][f'{family}-b256-seed{seed}']
        cfg = entry['config'].copy()
        if not entry['all_rows_and_metrics_equal'] or cfg['dense_learning_rate'] != 0.0002:
            raise ValueError('Unexpected reference protocol')
        if cfg['batch_size'] != 256 or cfg['sparse_learning_rate'] != 0.001:
            raise ValueError('Unexpected batch or embedding LR')
        name = f'{family}-b256-d001-seed{seed}'
        cfg.update(experiment=name, dense_learning_rate=0.001)
        configurations[name] = cfg
    if digest(args.content_checkpoint) != configurations['c6-b256-d001-seed42']['content_checkpoint_sha256']:
        raise ValueError('Content checkpoint mismatch')
    args.output_dir.mkdir()
    (args.output_dir/'configs').mkdir()
    for name, cfg in configurations.items():
        (args.output_dir/'configs'/f'{name}.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    state = dict(planned=list(configurations), completed=[], active=None, failed=None,
                 reference_audit_sha256=digest(args.audit), test_evaluated=False)

    def save():
        temp = args.output_dir/'status.tmp'
        temp.write_text(json.dumps(state, indent=2), encoding='utf-8')
        temp.replace(args.output_dir/'status.json')

    save()
    for name in configurations:
        state['active'] = name
        save()
        content = name.startswith('c6')
        script = (args.content_project/'scripts/train_c0.py' if content
                  else Path(__file__).with_name('train_muse.py'))
        command = [sys.executable, str(script.resolve()), '--store', str(args.store.resolve()),
                   '--config', str((args.output_dir/'configs'/f'{name}.json').resolve()),
                   '--output-dir', str((args.output_dir/name).resolve())]
        if content:
            command += ['--hist-cache', str(args.hist_cache.resolve()),
                        '--content-checkpoint', str(args.content_checkpoint.resolve())]
        with (args.output_dir/f'{name}.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            state.update(active=None, failed=name)
            save()
            raise RuntimeError(name + ' failed')
        state['completed'].append(name)
        state['active'] = None
        save()


if __name__ == '__main__':
    main()
