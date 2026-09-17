"""Serial remaining-seed retrieval matrices and source-aligned audits."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from run_shared_core import digest, read


def main():
    parser = argparse.ArgumentParser()
    for name in ['store', 'core-root', 'output-dir', 'first-audit','reuse-2026','reuse-audit']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    audit = read(args.first_audit)
    root = Path(__file__).resolve().parent
    if (audit['seed'] != 42 or audit['test_evaluated']
            or audit['store_manifest_sha256'] != digest(args.store/'manifest.json')
            or len(audit['runs']) != 7
            or not all(m['all_rows_and_recomputed_metrics_equal'] for m in audit['runs'].values())):
        raise ValueError('First-seed audit prerequisite failed')
    seeds = [2026, 3407]
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
    for seed in seeds:
        study = args.output_dir/f'seed{seed}'
        training=[sys.executable,str(root/'run_shared_retrieval.py'),
                  '--store',str(args.store.resolve()),'--seed',str(seed),'--output-dir',str(study.resolve())]
        if seed==2026:
            training+=['--reuse',str(args.reuse_2026.resolve()),'--reuse-audit',str(args.reuse_audit.resolve())]
        commands=[('training',training),('audit',[sys.executable,str(root/'audit_shared_retrieval.py'),
            '--store',str(args.store.resolve()),'--core',str((args.core_root/f'seed{seed}').resolve()),
            '--length',str(study.resolve()),'--output-dir',str((args.output_dir/f'seed{seed}-audit').resolve())])]
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
