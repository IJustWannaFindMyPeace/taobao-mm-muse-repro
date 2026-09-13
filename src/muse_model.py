"""Controlled R0-R4 backbone with common projected ID values."""
import torch
from torch import nn
from c0 import C0
from feature_store import FIELDS
from retrieval import select_history,gather_history
from enhanced_esu import LearnedSimTier,semantic_scores,semantic_attention


class MUSEModel(C0):
    def __init__(self,sizes,width=16,short_length=50,k=50,mode='cosine',semantic=True,histogram=True):
        super().__init__(sizes,width,short_length)
        self.k=k;self.mode=mode
        self.value=nn.Linear(width*2,width*2,bias=False)
        nn.init.eye_(self.value.weight)
        self.gamma=nn.Parameter(torch.tensor([1.,0.,0.])) if semantic else None
        self.histogram=LearnedSimTier() if histogram else None
        if histogram:
            self.histogram_projection=nn.Linear(22,128,bias=False)
            nn.init.zeros_(self.histogram_projection.weight)

    def forward(self,features,item,category,valid,target_content,history_content,target_available,history_available):
        indices,selected=select_history(target_content,history_content,valid,target_available,history_available,self.k,self.mode)
        scalar=[self.embeddings[f](features[:,i]) for i,f in enumerate(FIELDS)]
        target=torch.cat([scalar[6],scalar[7]],-1)
        short_hist=torch.cat([self.embeddings['205'](item[:,-self.short_length:]),
                              self.embeddings['206'](category[:,-self.short_length:])],-1)
        short=self.short_attention(target,short_hist,valid[:,-self.short_length:])
        long_hist=torch.cat([self.embeddings['205'](gather_history(item,indices,selected)),
                             self.embeddings['206'](gather_history(category,indices,selected))],-1)
        content=gather_history(history_content,indices,selected)
        available=gather_history(history_available,indices,selected)
        similarity,semantic_valid=semantic_scores(target_content,content,selected,target_available,available)
        long=semantic_attention(self.long_attention,self.value,target,long_hist,selected,similarity,self.gamma)
        first=self.tower[0](torch.cat(scalar+[short,long],-1))
        if self.histogram is not None:first=first+self.histogram_projection(self.histogram(similarity,semantic_valid))
        return self.tower[1:](first).squeeze(-1)
