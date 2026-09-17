"""Run the two projects' core matrix under one explicit downstream protocol."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    for key in ['store', 'content-project', 'content-run', 'hist-cache', 'output-dir']:
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--reuse-c6', type=Path)
    parser.add_argument('--reuse-audit', type=Path)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol_path = root/'configs/shared-downstream-protocol-v1.json'
    protocol = read(protocol_path)
    if protocol != read(args.content_project/'configs/shared-downstream-protocol-v1.json'):
        raise ValueError('Project protocols differ')
    if digest(args.store/'manifest.json') != protocol['store_manifest_sha256']:
        raise ValueError('Unexpected data store')
    manifest = read(args.store/'manifest.json')
    content = read(args.content_run/'report.json')
    seed = content['config']['seed']
    if seed not in protocol['training_seeds'] or content['test_evaluated']:
        raise ValueError('Unexpected pretraining seed or test evaluation')
    if content['store_manifest_sha256'] != protocol['store_manifest_sha256']:
        raise ValueError('Pretraining data mismatch')
    if len(content['epochs']) != content['config']['epochs']:
        raise ValueError('Incomplete pretraining')
    for epoch in content['epochs']:
        if epoch['rows'] != manifest['splits']['train']['rows'] or epoch['reload_max_abs_error'] != 0:
            raise ValueError('Invalid pretraining epoch')
    best = max(content['epochs'], key=lambda e: (e['validation']['gauc'], -e['epoch']))
    checkpoint = args.content_run/f"epoch-{best['epoch']}.pt"
    if digest(checkpoint) != best['checkpoint_sha256']:
        raise ValueError('Pretraining checkpoint mismatch')
    cache = read(args.hist_cache/'manifest.json')
    if cache['store_manifest_sha256'] != protocol['store_manifest_sha256']:
        raise ValueError('Histogram cache data mismatch')
    fields = ['batch_size', 'dense_learning_rate', 'sparse_learning_rate', 'epochs',
              'embedding_width', 'weight_decay', 'resource_limits']
    tasks = {}
    for family, repo, names in [
        ('content', args.content_project, ['c0', 'c2', 'c3', 'c5', 'c6', 'c7']),
        ('muse', root, ['r0', 'r1', 'r2', 'r3', 'r4', 'r5'])]:
        for name in names:
            cfg = read(repo/'configs'/f'{name}.json')
            cfg.update({key: protocol[key] for key in fields})
            cfg['seed'] = seed
            if family == 'content' and cfg['history_length'] != content['config']['history_length']:
                raise ValueError('Pretraining history mismatch')
            if cfg.get('simtier_bins') and (cfg['simtier_bins'] != cache['bins'] or cfg['history_length'] != cache['history_length']):
                raise ValueError('Histogram shape mismatch')
            if cfg.get('pretrained_content'):
                cfg['content_checkpoint_sha256'] = best['checkpoint_sha256']
            tasks[name] = dict(family=family, config=cfg)
    if bool(args.reuse_c6) != bool(args.reuse_audit):
        raise ValueError('Reuse requires an audited run and audit report')
    if args.reuse_c6:
        previous = read(args.reuse_c6/'report.json')
        audit = read(args.reuse_audit)
        entry = audit['runs'][f'c6-b256-d001-seed{seed}']
        normalize = lambda cfg: {k: v for k, v in cfg.items() if k != 'experiment'}
        if normalize(previous['config']) != normalize(tasks['c6']['config']):
            raise ValueError('Reuse config mismatch')
        if (audit['test_evaluated'] or audit['store_manifest_sha256'] != protocol['store_manifest_sha256']
                or not entry['all_rows_and_metrics_equal']
                or entry['report_sha256'] != digest(args.reuse_c6/'report.json')
                or entry['prediction_sha256'] != digest(args.reuse_c6/'validation_predictions.npz')
                or not (args.reuse_c6/'model.pt').is_file()):
            raise ValueError('Reuse evidence mismatch')
        for name, expected in previous['implementation_sha256'].items():
            path = args.content_project/('scripts' if name == 'train_c0.py' else 'src')/name
            if not path.is_file() or digest(path) != expected:
                raise ValueError('Reuse implementation changed: ' + name)
        tasks['c6']['reuse'] = str(args.reuse_c6.resolve())
    if args.plan_only:
        print(json.dumps(dict(protocol=protocol, seed=seed, tasks=tasks), indent=2))
        return
    if args.output_dir.exists():
        raise ValueError('Fresh matrix directory required')
    args.output_dir.mkdir()
    (args.output_dir/'configs').mkdir()
    for name, task in tasks.items():
        (args.output_dir/'configs'/f'{name}.json').write_text(json.dumps(task['config'], indent=2), encoding='utf-8')
    plan = dict(protocol=protocol, protocol_sha256=digest(protocol_path), seed=seed,
                content_report_sha256=digest(args.content_run/'report.json'), tasks=tasks,
                reuse_audit_sha256=digest(args.reuse_audit) if args.reuse_audit else None)
    (args.output_dir/'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    state = dict(planned=list(tasks), completed=[], reused=[], active=None, failed=None, test_evaluated=False)

    def save():
        temporary = args.output_dir/'status.tmp'
        temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
        temporary.replace(args.output_dir/'status.json')

    save()
    for name, task in tasks.items():
        state['active'] = name
        save()
        try:
            if 'reuse' in task:
                state['reused'].append(name)
            else:
                script = (args.content_project/'scripts/train_c0.py' if task['family'] == 'content'
                          else root/'scripts/train_muse.py')
                command = [sys.executable, '-u', str(script.resolve()), '--store', str(args.store.resolve()),
                           '--config', str((args.output_dir/'configs'/f'{name}.json').resolve()),
                           '--output-dir', str((args.output_dir/name).resolve())]
                if task['config'].get('simtier_bins'):
                    command += ['--hist-cache', str(args.hist_cache.resolve())]
                if task['config'].get('pretrained_content'):
                    command += ['--content-checkpoint', str(checkpoint.resolve())]
                with (args.output_dir/f'{name}.log').open('w', encoding='utf-8') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                report = read(args.output_dir/name/'report.json')
                if (report['config'] != task['config'] or report['test_evaluated']
                        or report['store_manifest_sha256'] != protocol['store_manifest_sha256']
                        or report['reload_max_abs_error'] != 0
                        or report['validation']['rows'] != manifest['splits']['validation']['rows']
                        or sum(e['rows'] for e in report['epochs']) != manifest['splits']['train']['rows'] * task['config']['epochs']):
                    raise ValueError('Completed run integrity failure')
            state['completed'].append(name)
            state['active'] = None
            save()
        except Exception as exc:
            state.update(active=None, failed=name, error=str(exc))
            save()
            raise


if __name__ == '__main__':
    main()
