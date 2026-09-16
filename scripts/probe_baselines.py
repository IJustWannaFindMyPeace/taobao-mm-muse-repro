"""One isolated real-data probe; stop at resource gates, never auto-shrink batches."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import psutil
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from baseline_model import BaselineModel
from feature_store import FeatureStore
from probe_search import tensors, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--store', type=Path, required=True)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--reference-run', type=Path)
    args = parser.parse_args()
    if args.steps <= 20 or args.report.exists():
        raise ValueError('Need >20 steps and fresh report')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    root = Path(__file__).resolve().parents[1]
    hashes = {str(p.relative_to(root)): digest(p) for p in [Path(__file__), root/'scripts/probe_search.py', *sorted((root/'src').glob('*.py'))]}
    torch.manual_seed(42)
    store = FeatureStore(args.store, 'train')
    model = BaselineModel(store.manifest['vocab_sizes'], args.variant, k=50).cuda()
    parity = None
    if args.reference_run:
        if args.variant != 'content_reference':
            raise ValueError('Replay requires unchanged content reference')
        old = json.loads((args.reference_run/'report.json').read_text())
        if old['store_manifest_sha256'] != digest(args.store/'manifest.json'):
            raise ValueError('Reference store mismatch')
        model.load_state_dict(torch.load(args.reference_run/'model.pt', map_location='cpu', weights_only=True))
        val = FeatureStore(args.store, 'validation')
        model.eval()
        with torch.no_grad():
            batch = val.batch(np.arange(32), 1000, with_content=True)
            pred = model(*tensors(batch)).cpu().numpy()
        with np.load(args.reference_run/'validation_predictions.npz') as saved:
            np.testing.assert_array_equal(saved['source_rows'][:32], batch['source_rows'])
            np.testing.assert_allclose(pred, saved['logits'][:32], rtol=0, atol=1e-6)
            parity = float(np.abs(pred-saved['logits'][:32]).max())
        # This branch only replays the checkpoint; resource probes start fresh.
        args.report.write_text(json.dumps({'scope':'validation_replay_only', 'rows':32,
            'max_abs_error':parity,'test_evaluated':False,'implementation_sha256':hashes}, indent=2))
        print('Reference replay passed', flush=True)
        return
    model.train()
    sparse = list(model.embeddings.parameters())
    sparse_ids = {id(p) for p in sparse}
    dense = [p for p in model.parameters() if id(p) not in sparse_ids and p.requires_grad]
    opts = [torch.optim.SparseAdam(sparse, lr=.001), torch.optim.AdamW(dense, lr=.001, weight_decay=0)]
    rows = np.random.default_rng(42).permutation(len(store))[:args.steps*32]
    process = psutil.Process()
    max_rss = process.memory_info().rss
    min_available = psutil.virtual_memory().available
    warm_seconds = None
    tracked = None
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); started = time.perf_counter()
    before = model.tower[-1].weight.detach().clone()
    for step, start in enumerate(range(0, len(rows), 32), 1):
        batch = store.batch(rows[start:start+32], 1000, with_content=True)
        for opt in opts: opt.zero_grad(set_to_none=True)
        if step == 1:
            observed = np.unique(np.r_[batch['features'][:,6], batch['history_item'][:,-50:].ravel()])
            observed = torch.as_tensor(observed[observed>1], device='cuda', dtype=torch.long)
            tracked = model.embeddings['205'].weight[observed].detach().clone()
        logits = model(*tensors(batch))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.as_tensor(batch['labels'], device='cuda'))
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                g = p.grad.coalesce().values() if p.grad.is_sparse else p.grad
                if not torch.isfinite(g).all(): raise RuntimeError('Nonfinite gradient')
        for opt in opts: opt.step()
        torch.cuda.synchronize()
        if step == 1:
            item_updated = not torch.equal(tracked, model.embeddings['205'].weight[observed])
        max_rss = max(max_rss, process.memory_info().rss)
        min_available = min(min_available, psutil.virtual_memory().available)
        if torch.cuda.max_memory_allocated() > 9*2**30 or max_rss > 24*2**30 or min_available < 4*2**30:
            raise RuntimeError('Resource gate exceeded; do not reduce batch automatically')
        if step == 20: warm_seconds = time.perf_counter()-started
        if step % 20 == 0: print(f'{args.variant}: {step}/{args.steps}', flush=True)
    seconds = time.perf_counter()-started
    updated = not torch.equal(before, model.tower[-1].weight)
    assert updated and item_updated
    result = {'scope':'short_resource_probe_not_effectiveness_evaluation','variant':args.variant,
        'store_manifest_sha256':digest(args.store/'manifest.json'),'implementation_sha256':hashes,
        'batch_size':32,'history_length':1000,'k':50,'steps':args.steps,'rows':len(rows),
        'seconds':seconds,'first_20_steps_seconds':warm_seconds,
        'post_warmup_rows_per_second':(args.steps-20)*32/(seconds-warm_seconds),
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
        'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
        'sampled_peak_process_rss_gib':max_rss/2**30,'sampled_min_system_available_gib':min_available/2**30,
        'tower_updated':updated,'shared_item_embedding_updated':item_updated,'resource_gate_passed':True,
        'gpu':torch.cuda.get_device_name(),'test_evaluated':False,
        'limitations':['100-step estimate is not a full-epoch throughput guarantee',
                       'RSS and available RAM sampled after each step; transient peaks may be missed',
                       'PyTorch allocation is not total device usage; no automatic batch reduction']}
    args.report.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
