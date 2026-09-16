"""One isolated fixed-exposure batch-size resource probe; no validation tuning."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--store', type=Path, required=True)
    p.add_argument('--model', choices=['c6', 'r4', 'id_item_category'], required=True)
    p.add_argument('--batch-size', type=int, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--content-project', type=Path, required=True)
    p.add_argument('--hist-cache', type=Path, required=True)
    p.add_argument('--content-checkpoint', type=Path, required=True)
    p.add_argument('--c6-config', type=Path, required=True)
    p.add_argument('--warmup-rows', type=int, default=4096)
    p.add_argument('--measured-rows', type=int, default=65536)
    args = p.parse_args()
    if args.report.exists() or args.batch_size < 1:
        raise ValueError('Fresh report and positive batch size required')
    if min(args.warmup_rows, args.measured_rows) < 1 or any(n % args.batch_size for n in [args.warmup_rows, args.measured_rows]):
        raise ValueError('Row budgets must be positive multiples of batch size')
    import numpy as np
    import psutil
    import torch
    root = Path(__file__).resolve().parents[1]
    selected_root = args.content_project if args.model == 'c6' else root
    sys.path.insert(0, str(selected_root/'src'))
    from feature_store import FeatureStore
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    sources = {str(f.relative_to(selected_root)): digest(f) for f in sorted((selected_root/'src').glob('*.py'))}
    report = {'scope':'fixed_exposure_batch_resource_probe_not_effectiveness', 'model':args.model,
              'batch_size':args.batch_size,'warmup_rows':args.warmup_rows,'measured_rows':args.measured_rows,
              'store_manifest_sha256':digest(args.store/'manifest.json'),
              'implementation_sha256':sources,'probe_sha256':digest(Path(__file__)),
              'test_evaluated':False,'validation_evaluated':False,'seed':42,
              'dense_learning_rate':.001,'sparse_learning_rate':.001,'status':'running'}
    process = psutil.Process()
    max_rss, min_ram, max_reserved = 0., float('inf'), 0.

    def guard():
        nonlocal max_rss, min_ram, max_reserved
        max_rss = max(max_rss, process.memory_info().rss/2**30)
        min_ram = min(min_ram, psutil.virtual_memory().available/2**30)
        max_reserved = max(max_reserved, torch.cuda.memory_reserved()/2**30)
        if torch.cuda.max_memory_allocated()/2**30 > 9 or max_rss > 24 or min_ram < 4:
            raise MemoryError('Resource gate exceeded; no automatic batch reduction')

    try:
        guard()
        torch.manual_seed(42)
        store = FeatureStore(args.store, 'train')
        if len(store) < args.warmup_rows + args.measured_rows:
            raise ValueError('Insufficient training exposures')
        hist = None
        if args.model == 'c6':
            from make_fusion import C6
            cfg = json.loads(args.c6_config.read_text())
            cache = json.loads((args.hist_cache/'manifest.json').read_text())
            if cache['store_manifest_sha256'] != report['store_manifest_sha256'] or cache['history_length'] != 200 or cache['bins'] != 20:
                raise ValueError('Histogram metadata mismatch')
            if digest(args.hist_cache/'train.npy') != cache['splits']['train']['sha256']:
                raise ValueError('Histogram digest mismatch')
            if digest(args.content_checkpoint) != cfg['content_checkpoint_sha256']:
                raise ValueError('Content checkpoint mismatch')
            model = C6(store.manifest['vocab_sizes'], freeze_content=True).cuda()
            model.content_branch.load_state_dict(torch.load(args.content_checkpoint, map_location='cpu', weights_only=True))
            hist = np.load(args.hist_cache/'train.npy', mmap_mode='r')
            length = 200
            report['content_checkpoint_sha256'] = cfg['content_checkpoint_sha256']
            report['hist_cache_sha256'] = digest(args.hist_cache/'manifest.json')
        else:
            from baseline_model import BaselineModel
            model = BaselineModel(store.manifest['vocab_sizes'], 'content_reference' if args.model == 'r4' else args.model, k=50).cuda()
            length = 1000
        report['history_length'] = length
        sparse = list(model.embeddings.parameters())
        ids = {id(x) for x in sparse}
        dense = [x for x in model.parameters() if id(x) not in ids and x.requires_grad]
        optimizers = [torch.optim.SparseAdam(sparse, lr=.001), torch.optim.AdamW(dense, lr=.001, weight_decay=0)]
        order = np.random.default_rng(42).permutation(len(store))[:args.warmup_rows+args.measured_rows]
        report['exposure_order_sha256'] = hashlib.sha256(order.tobytes()).hexdigest()
        before = model.tower[-1].weight.detach().clone()
        torch.cuda.reset_peak_memory_stats()
        durations = {'lookup':0., 'transfer':0., 'compute_and_update':0.}
        started = measured_start = None
        for offset in range(0, len(order), args.batch_size):
            torch.cuda.synchronize()
            if offset == 0: started = time.perf_counter()
            if offset == args.warmup_rows: measured_start = time.perf_counter()
            measured = offset >= args.warmup_rows
            t = time.perf_counter()
            selection = order[offset:offset+args.batch_size]
            batch = store.batch(selection, length, with_content=True)
            if hist is not None: batch['histogram'] = hist[selection]
            t1 = time.perf_counter()
            names = ['features','history_item','history_category','history_valid']
            if hist is not None: names.append('histogram')
            names += ['target_content','history_content','target_content_available','history_content_available']
            inputs = [torch.as_tensor(batch[k],device='cuda',dtype=torch.long if k in names[:3] else None) for k in names]
            labels = torch.as_tensor(batch['labels'],device='cuda')
            torch.cuda.synchronize(); t2 = time.perf_counter()
            for opt in optimizers: opt.zero_grad(set_to_none=True)
            logits = model(*inputs)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            loss.backward()
            for param in model.parameters():
                if param.grad is not None:
                    gradient = param.grad.coalesce().values() if param.grad.is_sparse else param.grad
                    if not torch.isfinite(gradient).all(): raise RuntimeError('Nonfinite gradient')
            for opt in optimizers: opt.step()
            torch.cuda.synchronize(); t3 = time.perf_counter()
            if measured:
                durations['lookup'] += t1-t
                durations['transfer'] += t2-t1
                durations['compute_and_update'] += t3-t2
            guard()
            if (offset+args.batch_size) % 16384 == 0:
                print(f'{args.model} B{args.batch_size}: {offset+args.batch_size}/{len(order)} rows',flush=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-measured_start
        assert not torch.equal(before, model.tower[-1].weight)
        report.update(status='passed', measured_seconds=elapsed, measured_rows_per_second=args.measured_rows/elapsed,
                      total_seconds=time.perf_counter()-started, measured_stage_seconds=durations,
                      enough_measurement_time=elapsed>=30, parameter_update_verified=True)
    except (torch.cuda.OutOfMemoryError, MemoryError) as exc:
        report.update(status='resource_rejected',reason=str(exc))
    report.update(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  peak_reserved_gib=max_reserved, sampled_peak_rss_gib=max_rss,
                  sampled_min_available_ram_gib=min_ram,
                  limitations=['fixed order, single timing run; repeat finalists in reverse order',
                               'synchronized timing includes instrumentation; not full-epoch guarantee',
                               'different batch sizes imply different optimizer step counts; no quality conclusion'])
    args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(report['status'],flush=True)


if __name__ == '__main__':
    main()
