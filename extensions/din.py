"""DIN local-activation mechanism on the frozen study's shared CTR backbone.

This is a paper-informed adaptation, not a byte-equivalent industrial DIN model.
"""
import torch
from torch import nn
from torch.nn import functional as F
from c0 import C0
from feature_store import FIELDS


class Dice(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.norm=nn.BatchNorm1d(width,eps=1e-8,momentum=.1,affine=False)
        self.alpha=nn.Parameter(torch.zeros(width))

    def forward(self,x):
        # Only valid behavior rows enter this module. A singleton cannot supply
        # batch variance, so use the stored inference statistics in that case.
        normalized=(F.batch_norm(x,self.norm.running_mean,self.norm.running_var,
                                training=False,eps=self.norm.eps)
                    if self.training and len(x)<2 else self.norm(x))
        gate=normalized.sigmoid()
        return gate*x+(1-gate)*self.alpha*x


class LocalActivation(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.score=nn.Sequential(nn.Linear(4*width,80),Dice(80),nn.Linear(80,40),Dice(40),nn.Linear(40,1))

    def forward(self,target,history,valid):
        if not valid.any():return history.sum(1)*0
        query=target[:,None,:].expand_as(history)
        features=torch.cat([query,history,query-history,query*history],-1)
        weights=history.new_zeros(valid.shape)
        weights[valid]=self.score(features[valid]).squeeze(-1)
        # DIN preserves interest intensity: do not normalize weights with softmax.
        return (history*weights[...,None]).sum(1)


class ShortInterest(C0):
    def __init__(self,sizes,width=16,short_length=50,mechanism='din_lau'):
        if mechanism not in ('din_lau','dot_softmax'):raise ValueError('Unknown short-interest mechanism')
        super().__init__(sizes,width,short_length)
        self.long_attention.requires_grad_(False)
        if mechanism=='din_lau':self.short_attention=LocalActivation(width*2)

    def representation(self,features,item,category,valid):
        scalar=[self.embeddings[f](features[:,i]) for i,f in enumerate(FIELDS)]
        target=torch.cat([scalar[6],scalar[7]],-1)
        history=torch.cat([self.embeddings['205'](item[:,-self.short_length:]),
                           self.embeddings['206'](category[:,-self.short_length:])],-1)
        short=self.short_attention(target,history,valid[:,-self.short_length:])
        return torch.cat(scalar+[short,torch.zeros_like(short)],-1)
