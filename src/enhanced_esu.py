"""Paper SA-TA score fusion and masked MUSE-code-style learned log counts."""
import torch
from torch import nn


def semantic_scores(target,history,valid,target_available,history_available):
    mask=valid & history_available & target_available[:,None]
    mask=mask & (target.norm(dim=-1)>0)[:,None] & (history.norm(dim=-1)>0)
    q=nn.functional.normalize(target.detach(),dim=-1)
    h=nn.functional.normalize(history.detach(),dim=-1)
    return (q[:,None]*h).sum(-1).clamp(-1,1)*mask,mask


def fuse_scores(id_scores,similarity,gamma):
    return gamma[0]*id_scores+gamma[1]*similarity+gamma[2]*id_scores*similarity


def semantic_attention(attention,value,target,history,valid,similarity,gamma):
    scores=(attention.query(target)[:,None]*attention.key(history)).sum(-1)/history.shape[-1]**.5
    if gamma is not None:scores=fuse_scores(scores,similarity,gamma)
    # Mask after fusion: learned coefficients must never revive padded slots.
    weights=scores.masked_fill(~valid,torch.finfo(scores.dtype).min).softmax(-1)*valid
    weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
    return (value(history)*weights[...,None]).sum(1)


class LearnedSimTier(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale=nn.Parameter(torch.ones(21))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        key = prefix + 'scale'
        if key in state_dict and state_dict[key].shape == (22,):
            state_dict[key] = state_dict[key][:21].clone()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self,similarity,valid):
        # Scores map to 0..20: one endpoint bin and 20 right-closed intervals.
        indices=(torch.ceil(similarity.clamp(-1,1)*10).long()+10).clamp(0,20)
        counts=torch.zeros((len(similarity),21),device=similarity.device)
        counts.scatter_add_(1,indices,valid.float())
        return torch.log1p(counts)*self.scale


class SimTierProjection(nn.Linear):
    """Accept legacy weights whose last input column multiplied an unused bin."""
    def __init__(self):
        super().__init__(21, 128, bias=False)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        key = prefix + 'weight'
        if key in state_dict and state_dict[key].shape == (128, 22):
            state_dict[key] = state_dict[key][:, :21].clone()
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
