"""Serial remaining-seed matrices, with an independent audit after each seed."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from run_shared_core import digest, read


def main():
    parser = argparse.ArgumentParser()
    for name in ['store', 'content-project', 'hist-cache', 'output-dir', 'first-audit']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--content-runs', type=Path, nargs='+', required=True)
    args = parser.parse_args()
    audit = read(args.first_audit)
    root = Path(__file__).resolve().parent
    if (audit['seed'] != 42 or audit['test_evaluated']
            or audit['store_manifest_sha256'] != digest(args.store/'manifest.json')
            or set(audit['models']) != {'c0','c2','c3','c5','c6','c7','r0','r1','r2','r3','r4','r5'}
            or not all(m['saved_rows_and_metrics_verified'] for m in audit['models'].values())):
        raise ValueError('First-seed audit prerequisite failed')
    seeds = [read(path/'report.json')['config']['seed'] for path in args.content_runs]
    if seeds != [2026, 3407]:
        raise ValueError('Expected remaining seeds in fixed order')
    if args.output_dir.exists():
        raise ValueError('Fresh queue directory required')
    args.output_dir.mkdir()
    state = dict(planned=seeds, completed=[], active=None, phase=None, failed=None,
                 first_audit_sha256=digest(args.first_audit), test_evaluated=False)

    def save():
        temporary = args.output_dir/'status.tmp'
        temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
        temporary.replace(args.output_dir/'status.json')

    save()
    for seed, content_run in zip(seeds, args.content_runs):
        study = args.output_dir/f'seed{seed}'
        commands = [
            ('training', [sys.executable, str(root/'run_shared_core.py'),
                          '--store', str(args.store.resolve()), '--content-project', str(args.content_project.resolve()),
                          '--hist-cache', str(args.hist_cache.resolve()), '--content-run', str(content_run.resolve()),
                          '--output-dir', str(study.resolve())]),
            ('audit', [sys.executable, str(root/'audit_shared_core.py'),
                       '--store', str(args.store.resolve()), '--runs', str(study.resolve()),
                       '--report', str((args.output_dir/f'seed{seed}-audit.json').resolve())])]
        for phase, command in commands:
            state.update(active=seed, phase=phase)
            save()
            with (args.output_dir/f'seed{seed}-{phase}.log').open('w', encoding='utf-8') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                state.update(failed=seed, active=None)
                save()
                raise RuntimeError(f'Seed {seed} failed during {phase}; queue stopped')
        state['completed'].append(seed)
        state.update(active=None, phase=None)
        save()


if __name__ == '__main__':
    main()
