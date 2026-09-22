"""Equal-weight content/item-ID cosine retrieval with content-reference eligibility."""
import torch
from torch.nn import functional as F
from baseline_retrieval import _rank


@torch.no_grad()
def fused_cosine_history(target, history, valid, target_available, history_available,
                         target_id_embedding, history_id_embedding, target_id, history_id, k=50):
    target_ok = target_available & (target.norm(dim=-1) > 0)
    eligible = valid & history_available & (history.norm(dim=-1) > 0) & target_ok[:, None]
    content_score = (F.normalize(target, dim=-1)[:, None] *
                     F.normalize(history, dim=-1)).sum(-1).clamp(-1, 1)
    id_ok = ((target_id > 1) & (target_id_embedding.norm(dim=-1) > 0))[:, None]
    id_ok = id_ok & (history_id > 1) & (history_id_embedding.norm(dim=-1) > 0)
    id_score = (F.normalize(target_id_embedding, dim=-1)[:, None] *
                F.normalize(history_id_embedding, dim=-1)).sum(-1).clamp(-1, 1)
    scores = content_score + id_score.masked_fill(~id_ok, 0)
    recency = torch.arange(valid.shape[1], device=valid.device).float().expand_as(valid)
    scores = torch.where(target_ok[:, None], scores, recency)
    eligible = torch.where(target_ok[:, None], eligible, valid)
    return _rank(scores, eligible, k)
