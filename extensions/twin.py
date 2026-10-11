"""Four-head consistent-search adaptation; no industrial cross-feature cache."""
import torch
from torch import nn
from c0 import C0
from feature_store import FIELDS
from retrieval import select_history, gather_history
from enhanced_esu import LearnedSimTier, SimTierProjection, semantic_scores


def masked_topk(scores,valid,k):
    if not 1<=k<=scores.shape[-1]:raise ValueError('Invalid K')
    scores=scores.masked_fill(~valid,float('-inf'))
    order=torch.argsort(scores.flip(-1),descending=True,stable=True,dim=-1)[...,:k]
    indices=scores.shape[-1]-1-order
    return indices,valid.expand_as(scores).gather(-1,indices)


class FourHeadAttention(nn.Module):
    def __init__(self,width,heads=4):
        super().__init__()
        if width%heads:raise ValueError('Width must divide into heads')
        self.heads=heads;self.dim=width//heads
        self.query=nn.Linear(width,width,bias=False)
        self.key=nn.Linear(width,width,bias=False)
        self.value=nn.Linear(width,width,bias=False)
        self.output=nn.Linear(width,width,bias=False)

    def scores(self,target,history):
        q=self.query(target).reshape(len(target),self.heads,self.dim)
        k=self.key(history).reshape(len(target),history.shape[1],self.heads,self.dim)
        return torch.einsum('bhd,blhd->bhl',q,k)/self.dim**.5

    def aggregate(self,target,histories,valid):
        # histories: B,H,K,W. Each head recomputes its own selected scores
        # with the same projection parameters used for full-history search.
        q=self.query(target).reshape(len(target),self.heads,self.dim)
        keys=[];values=[]
        for h in range(self.heads):
            keys.append(self.key(histories[:,h]).reshape(len(target),-1,self.heads,self.dim)[:,:,h])
            values.append(self.value(histories[:,h]).reshape(len(target),-1,self.heads,self.dim)[:,:,h])
        keys=torch.stack(keys,1);values=torch.stack(values,1)
        scores=(q[:,:,None]*keys).sum(-1)/self.dim**.5
        weights=scores.masked_fill(~valid,torch.finfo(scores.dtype).min).softmax(-1)*valid
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
        result=(weights[...,None]*values).sum(2).flatten(1)
        return self.output(result),scores


class TwinAdaptation(C0):
    def __init__(self,sizes,width=16,short_length=50,k=50,retrieval='consistent'):
        super().__init__(sizes,width,short_length)
        if retrieval not in ('consistent','content'):raise ValueError('Unknown retrieval')
        self.k=k;self.retrieval=retrieval
        self.long_attention=FourHeadAttention(width*2)
        self.histogram=LearnedSimTier()
        self.histogram_projection=SimTierProjection()
        nn.init.zeros_(self.histogram_projection.weight)

    def history_vectors(self,item,category):
        return torch.cat([self.embeddings['205'](item),self.embeddings['206'](category)],-1)

    @torch.no_grad()
    def select(self,target,item,category,valid,target_content,history_content,target_available,history_available):
        if self.retrieval=='content':
            index,mask=select_history(target_content,history_content,valid,target_available,history_available,self.k)
            return index[:,None].expand(-1,4,-1),mask[:,None].expand(-1,4,-1)
        # Transient embedding lookup does not retain an L-length training graph.
        scores=self.long_attention.scores(target,self.history_vectors(item,category))
        return masked_topk(scores,valid[:,None],self.k)

    def forward(self,features,item,category,valid,target_content,history_content,target_available,history_available):
        scalar=[self.embeddings[f](features[:,i]) for i,f in enumerate(FIELDS)]
        target=torch.cat([scalar[6],scalar[7]],-1)
        short=self.short_attention(target,self.history_vectors(item[:,-self.short_length:],category[:,-self.short_length:]),valid[:,-self.short_length:])
        indices,selected=self.select(target,item,category,valid,target_content,history_content,target_available,history_available)
        histories=torch.stack([self.history_vectors(gather_history(item,indices[:,h],selected[:,h]),
            gather_history(category,indices[:,h],selected[:,h])) for h in range(4)],1)
        long,_=self.long_attention.aggregate(target,histories,selected)
        # Count each selected behavior position once, even if several heads
        # retrieve it. Repeated item IDs at distinct positions remain behaviors.
        counts=torch.zeros_like(valid,dtype=torch.int32)
        counts.scatter_add_(1,indices.flatten(1),selected.flatten(1).int())
        union=counts>0
        similarity,semantic_valid=semantic_scores(target_content,history_content,union,target_available,history_available)
        self.last_budget={'head_positions':selected.sum().detach(),
                          'unique_positions':union.sum().detach(),'examples':len(item)}
        first=self.tower[0](torch.cat(scalar+[short,long],-1))
        first=first+self.histogram_projection(self.histogram(similarity,semantic_valid))
        return self.tower[1:](first).squeeze(-1)
