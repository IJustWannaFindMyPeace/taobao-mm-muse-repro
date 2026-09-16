"""Isolated baseline selectors; encoded 0=padding and 1=unknown.

These implement retrieval contracts, not complete DIN or SIM models.
"""
import torch
from torch.nn import functional as F


def _rank(scores, eligible, k):
    if scores.ndim != 2 or scores.shape != eligible.shape:
        raise ValueError('Expected equally shaped batch/history arrays')
    if not 1 <= k <= scores.shape[1]:
        raise ValueError('Require 1 <= K <= L')
    order = torch.argsort(scores.masked_fill(~eligible, -torch.inf).flip(1),
                         dim=1, descending=True, stable=True)[:, :k]
    indices = scores.shape[1] - 1 - order
    return indices, eligible.gather(1, indices)


@torch.no_grad()
def category_history(target_category, history_category, valid, k=50):
    """Most recent known same-category behaviors; never equate unknown IDs.

    Unknown target category or no match returns no valid selection. Insufficient
    matches remain masked; no cross-category backfill. This is a local policy.
    """
    eligible = valid & (history_category > 1) & (target_category[:, None] > 1)
    eligible &= history_category == target_category[:, None]
    recency = torch.arange(valid.shape[1], device=valid.device).float().expand_as(valid)
    return _rank(recency, eligible, k)


@torch.no_grad()
def id_cosine_history(target_embedding, history_embedding, target_id, history_id, valid, k=50):
    """G-ID cosine on shared item embeddings, not a complete SIM-soft GSU.

    Caller must reuse embeddings trained by the CTR paths and re-lookup selected
    IDs outside no-grad for ESU training. Unknown/zero targets use recent valid
    history. Known targets exclude unknown or zero-vector candidate IDs.
    """
    target_ok = (target_id > 1) & (target_embedding.norm(dim=-1) > 0)
    candidate_ok = valid & (history_id > 1) & (history_embedding.norm(dim=-1) > 0)
    scores = (F.normalize(target_embedding, dim=-1)[:, None] *
              F.normalize(history_embedding, dim=-1)).sum(-1).clamp(-1, 1)
    recency = torch.arange(valid.shape[1], device=valid.device).float().expand_as(valid)
    scores = torch.where(target_ok[:, None], scores, recency)
    eligible = torch.where(target_ok[:, None], candidate_ok, valid)
    return _rank(scores, eligible, k)
