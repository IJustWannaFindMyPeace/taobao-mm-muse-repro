"""R0/R1 search controls using the unchanged companion C0 parameters."""
import torch
from c0 import C0
from feature_store import FIELDS
from retrieval import select_history,gather_history


class SearchModel(C0):
    def __init__(self,sizes,width=16,short_length=50,k=50,mode='cosine'):
        super().__init__(sizes,width,short_length)
        self.k=k;self.mode=mode

    def forward(self,features,item,category,valid,target_content,history_content,target_available,history_available):
        indices,selected=select_history(target_content,history_content,valid,target_available,history_available,self.k,self.mode)
        scalar=[self.embeddings[f](features[:,i]) for i,f in enumerate(FIELDS)]
        target=torch.cat([scalar[6],scalar[7]],-1)
        # Short branch uses original recent history, independent of retrieval.
        short_item=item[:,-self.short_length:];short_category=category[:,-self.short_length:]
        short_hist=torch.cat([self.embeddings['205'](short_item),self.embeddings['206'](short_category)],-1)
        short=self.short_attention(target,short_hist,valid[:,-self.short_length:])
        chosen_item=gather_history(item,indices,selected)
        chosen_category=gather_history(category,indices,selected)
        long_hist=torch.cat([self.embeddings['205'](chosen_item),self.embeddings['206'](chosen_category)],-1)
        long=self.long_attention(target,long_hist,selected)
        return self.tower(torch.cat(scalar+[short,long],-1)).squeeze(-1)
