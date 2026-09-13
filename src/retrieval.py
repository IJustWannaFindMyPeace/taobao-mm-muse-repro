"""Deterministic masked selection over behavior positions, not unique items."""
import torch
from torch.nn import functional as F


@torch.no_grad()
def select_history(target,history,valid,target_available,history_available,k=50,mode='cosine'):
    if mode not in ['recent','cosine']:raise ValueError('Unknown selection mode')
    if not 1<=k<=history.shape[1]:raise ValueError('Require 1 <= K <= L')
    positions=torch.arange(history.shape[1],device=history.device).expand_as(valid)
    target_ok=target_available & (target.norm(dim=-1)>0)
    eligible=valid & history_available & (history.norm(dim=-1)>0) & target_ok[:,None]
    cosine=(F.normalize(target,dim=-1)[:,None]*F.normalize(history,dim=-1)).sum(-1).clamp(-1,1)
    if mode=='recent':
        mask=valid; scores=positions.float()
    else:
        # An unavailable target has no semantic retrieval criterion; fall back
        # to recent ID-valid behaviors. Otherwise exclude missing content.
        mask=torch.where(target_ok[:,None],eligible,valid)
        scores=torch.where(target_ok[:,None],cosine,positions.float())
    scores=scores.masked_fill(~mask,float('-inf'))
    # Reverse chronological positions before a stable descending sort so ties
    # favor recency. Invalid slots remain explicitly masked after gathering.
    order=torch.argsort(scores.flip(1),dim=1,descending=True,stable=True)[:,:k]
    indices=history.shape[1]-1-order
    selected_valid=mask.gather(1,indices)
    return indices,selected_valid


def gather_history(values,indices,valid):
    index=indices if values.ndim==2 else indices[...,None].expand(-1,-1,values.shape[-1])
    result=values.gather(1,index)
    return result.masked_fill(~(valid if values.ndim==2 else valid[...,None]),0)
