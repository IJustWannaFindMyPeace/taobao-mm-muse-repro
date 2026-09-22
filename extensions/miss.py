"""MISS-inspired dual-route CTR adaptation, without tree retrieval or MMoE."""
import torch
from torch import nn
from c0 import C0
from feature_store import FIELDS
from retrieval import select_history,gather_history
from twin import masked_topk


class RouteAttention(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.query=nn.Linear(width,width,bias=False)
        self.key=nn.Linear(width,width,bias=False)
        self.value=nn.Linear(width,width,bias=False)

    def scores(self,target,history):
        return (self.query(target)[:,None]*self.key(history)).sum(-1)/history.shape[-1]**.5

    def forward(self,target,history,valid):
        scores=self.scores(target,history)
        weights=scores.masked_fill(~valid,torch.finfo(scores.dtype).min).softmax(-1)*valid
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
        return (weights[...,None]*self.value(history)).sum(1)


class MissAdaptation(C0):
    def __init__(self,sizes,width=16,short_length=50,k=50,variant='hybrid_separate'):
        super().__init__(sizes,width,short_length)
        if variant not in ('hybrid_separate','content_separate','hybrid_shared'):raise ValueError('Unknown variant')
        if k%2:raise ValueError('Even total route budget required')
        self.k=k;self.variant=variant
        self.long_attention.requires_grad_(False)
        self.co_attention=RouteAttention(width*2)
        self.mm_attention=RouteAttention(width*2)
        self.merge=nn.Linear(width*4,width*2,bias=False)
        if variant=='hybrid_shared':self.mm_attention=self.co_attention

    def history_vectors(self,item,category):
        return torch.cat([self.embeddings['205'](item),self.embeddings['206'](category)],-1)

    @torch.no_grad()
    def select(self,target,item,category,valid,target_content,history_content,target_available,history_available):
        half=self.k//2
        if self.variant=='content_separate':
            index,mask=select_history(target_content,history_content,valid,target_available,history_available,self.k)
            # Preserve identical two-route capacity while selecting 50 distinct
            # positions by one content criterion. No implicit score fusion.
            return [(index[:,half:],mask[:,half:]),(index[:,:half],mask[:,:half])]
        scores=self.co_attention.scores(target,self.history_vectors(item,category))
        co=masked_topk(scores,valid,half)
        mm=select_history(target_content,history_content,valid,target_available,history_available,half)
        return [co,mm]

    def forward(self,features,item,category,valid,target_content,history_content,target_available,history_available):
        scalar=[self.embeddings[f](features[:,i]) for i,f in enumerate(FIELDS)]
        target=torch.cat([scalar[6],scalar[7]],-1)
        short=self.short_attention(target,self.history_vectors(item[:,-self.short_length:],category[:,-self.short_length:]),valid[:,-self.short_length:])
        routes=self.select(target,item,category,valid,target_content,history_content,target_available,history_available)
        representations=[]
        union_counts=torch.zeros_like(valid,dtype=torch.int32)
        for attention,(indices,mask) in zip((self.co_attention,self.mm_attention),routes):
            history=self.history_vectors(gather_history(item,indices,mask),gather_history(category,indices,mask))
            # Recompute using the Co-GSU parameters to preserve training
            # gradients; the multimodal route also aggregates ID values only.
            representations.append(attention(target,history,mask))
            union_counts.scatter_add_(1,indices,mask.int())
        self.last_budget={'route_positions':union_counts.sum().detach(),
                          'unique_positions':(union_counts>0).sum().detach(),
                          'overlap_positions':(union_counts>1).sum().detach(),
                          'examples':len(item)}
        long=self.merge(torch.cat(representations,-1))
        return self.tower(torch.cat(scalar+[short,long],-1)).squeeze(-1)
