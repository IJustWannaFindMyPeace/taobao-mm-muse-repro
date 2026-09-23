"""G3 adaptation: share SA-TA scoring parameters between GSU and ESU."""
import torch
from muse_model import MUSEModel
from feature_store import FIELDS
from retrieval import select_history,gather_history
from enhanced_esu import semantic_scores,semantic_attention,fuse_scores
from twin import masked_topk


class FusedAttentionSearch(MUSEModel):
    def __init__(self,*args,retrieval='fused_attention',**kwargs):
        super().__init__(*args,**kwargs)
        if retrieval not in ('fused_attention','content'):raise ValueError('Unknown retrieval')
        if self.gamma is None or not self.long_id:raise ValueError('Shared SA-TA required')
        self.retrieval=retrieval

    @torch.no_grad()
    def select(self,features,item,category,valid,tc,hc,ta,ha):
        if self.retrieval=='content':return select_history(tc,hc,valid,ta,ha,self.k,self.mode)
        target=torch.cat([self.embeddings['205'](features[:,6]),self.embeddings['206'](features[:,7])],-1)
        history=torch.cat([self.embeddings['205'](item),self.embeddings['206'](category)],-1)
        scores=(self.long_attention.query(target)[:,None]*self.long_attention.key(history)).sum(-1)/history.shape[-1]**.5
        similarity,eligible=semantic_scores(tc,hc,valid,ta,ha)
        fused=fuse_scores(scores,similarity,self.gamma)
        target_ok=ta & (tc.norm(dim=-1)>0)
        mask=torch.where(target_ok[:,None],eligible,valid)
        positions=torch.arange(item.shape[1],device=item.device).expand_as(item).float()
        fused=torch.where(target_ok[:,None],fused,positions)
        return masked_topk(fused,mask,self.k)

    def forward(self,features,item,category,valid,target_content,history_content,target_available,history_available):
        indices,selected=self.select(features,item,category,valid,target_content,history_content,target_available,history_available)
        total=selected.sum().detach()
        self.last_budget={'route_positions':total,'unique_positions':total,
                          'overlap_positions':total*0,'examples':len(item)}
        scalar=[self.embeddings[f](features[:,i]) for i,f in enumerate(FIELDS)]
        target=torch.cat([scalar[6],scalar[7]],-1)
        short_hist=torch.cat([self.embeddings['205'](item[:,-self.short_length:]),self.embeddings['206'](category[:,-self.short_length:])],-1)
        short=self.short_attention(target,short_hist,valid[:,-self.short_length:])
        long_hist=torch.cat([self.embeddings['205'](gather_history(item,indices,selected)),self.embeddings['206'](gather_history(category,indices,selected))],-1)
        content=gather_history(history_content,indices,selected)
        available=gather_history(history_available,indices,selected)
        similarity,semantic_valid=semantic_scores(target_content,content,selected,target_available,available)
        long=semantic_attention(self.long_attention,self.value,target,long_hist,selected,similarity,self.gamma)
        first=self.tower[0](torch.cat(scalar+[short,long],-1))
        if self.histogram is not None:first=first+self.histogram_projection(self.histogram(similarity,semantic_valid))
        return self.tower[1:](first).squeeze(-1)
