import sys
import unittest
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'extensions'),str(root/'src')]
from twin import FourHeadAttention,TwinAdaptation,masked_topk
from feature_store import FIELDS


class TwinExtension(unittest.TestCase):
    def test_selection_scores_match_aggregation_and_receive_gradients(self):
        torch.manual_seed(9);a=FourHeadAttention(8)
        q=torch.randn(3,8);h=torch.randn(3,11,8)
        valid=torch.ones(3,11,dtype=torch.bool);valid[0]=False
        with torch.no_grad():
            scores=a.scores(q,h);index,mask=masked_topk(scores,valid[:,None],4)
        selected=torch.stack([h.gather(1,index[:,head,:,None].expand(-1,-1,8)) for head in range(4)],1)
        out,recomputed=a.aggregate(q,selected,mask)
        torch.testing.assert_close(recomputed,scores.gather(-1,index))
        self.assertEqual(out[0].abs().sum().item(),0)
        out.square().sum().backward()
        for layer in [a.query,a.key,a.value,a.output]:
            self.assertGreater(layer.weight.grad.abs().sum().item(),0)

    def test_mask_and_recency_ties(self):
        scores=torch.zeros(1,4,7);valid=torch.tensor([[[False,True,False,True,False,False,True]]])
        index,mask=masked_topk(scores,valid,5)
        self.assertEqual(index[0,0,:3].tolist(),[6,3,1])
        self.assertEqual(mask.sum().item(),12)

    def test_same_parameters_controls_and_valid_budget(self):
        sizes={f:20 for f in FIELDS}
        torch.manual_seed(4);a=TwinAdaptation(sizes,width=4,k=3)
        torch.manual_seed(4);b=TwinAdaptation(sizes,width=4,k=3,retrieval='content')
        for key,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[key])
        features=torch.full((2,len(FIELDS)),2,dtype=torch.long)
        item=torch.tensor([[0,2,3,4,5],[0,0,0,0,0]])
        category=item.clone();valid=item!=0
        tc=torch.randn(2,6);hc=torch.randn(2,5,6)
        args=(features,item,category,valid,tc,hc,torch.ones(2,dtype=torch.bool),valid)
        out=b(*args)
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(b.last_budget['head_positions'].item(),12)
        self.assertEqual(b.last_budget['unique_positions'].item(),3)
        a(*args).sum().backward()
        self.assertGreater(a.long_attention.query.weight.grad.abs().sum().item(),0)
        self.assertTrue(a.embeddings['205'].weight.grad.is_sparse)
        b.load_state_dict(a.state_dict());b.retrieval='consistent'
        torch.testing.assert_close(a(*args),b(*args),rtol=0,atol=0)

    def test_content_missing_fallback(self):
        sizes={f:10 for f in FIELDS};a=TwinAdaptation(sizes,width=4,k=2,retrieval='content')
        valid=torch.tensor([[False,True,True,True]])
        idx,mask=a.select(torch.zeros(1,8),torch.tensor([[0,2,3,4]]),torch.tensor([[0,2,3,4]]),
                          valid,torch.zeros(1,6),torch.randn(1,4,6),torch.tensor([False]),valid)
        self.assertEqual(idx[0,0].tolist(),[3,2]);self.assertTrue(mask.all())


if __name__=='__main__':unittest.main()
