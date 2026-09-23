"""G1 local adaptation: train a content scorer, then freeze it for retrieval."""
import torch
from torch import nn
from torch.nn import functional as F
from fused_attention_search import FusedAttentionSearch
from retrieval import select_history
from twin import masked_topk


class ContentProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Linear(64,32))

    def forward(self,x):return F.normalize(self.layers(F.normalize(x,dim=-1)),dim=-1)

    def scores(self,target,history):return (self(target)[:,None]*self(history)).sum(-1)


class ContentScorerPretraining(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection=ContentProjection()
        self.tower=nn.Sequential(nn.Linear(256,64),nn.ReLU(),nn.Linear(64,1))

    def forward(self,features,item,category,valid,tc,hc,ta,ha):
        mask=valid & ha & ta[:,None] & (tc.norm(dim=-1)>0)[:,None] & (hc.norm(dim=-1)>0)
        total=mask.sum().detach()
        self.last_budget={'route_positions':total,'unique_positions':total,'overlap_positions':total*0,'examples':len(tc)}
        scores=self.projection.scores(tc,hc)/.1
        weights=scores.masked_fill(~mask,torch.finfo(scores.dtype).min).softmax(-1)*mask
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
        interest=(F.normalize(hc,dim=-1)*weights[...,None]).sum(1)
        return self.tower(torch.cat([F.normalize(tc,dim=-1)*ta[:,None],interest],-1)).squeeze(-1)


class LearnedContentSearch(FusedAttentionSearch):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,retrieval='content',**kwargs)
        self.projection=ContentProjection().requires_grad_(False)

    @torch.no_grad()
    def select(self,features,item,category,valid,tc,hc,ta,ha):
        target_ok=ta & (tc.norm(dim=-1)>0)
        eligible=valid & ha & (hc.norm(dim=-1)>0) & target_ok[:,None]
        mask=torch.where(target_ok[:,None],eligible,valid)
        scores=self.projection.scores(tc,hc)
        positions=torch.arange(item.shape[1],device=item.device).expand_as(item).float()
        scores=torch.where(target_ok[:,None],scores,positions)
        return masked_topk(scores,mask,self.k)
