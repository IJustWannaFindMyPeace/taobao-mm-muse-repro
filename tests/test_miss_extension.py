import sys
import unittest
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'extensions'),str(root/'src')]
from miss import MissAdaptation
from feature_store import FIELDS


class MissExtension(unittest.TestCase):
    def inputs(self):
        features=torch.full((2,len(FIELDS)),2,dtype=torch.long)
        item=torch.tensor([[0,2,3,4,5,6],[0,0,0,0,0,0]])
        valid=item!=0
        return features,item,item.clone(),valid,torch.randn(2,8),torch.randn(2,6,8),torch.ones(2,dtype=torch.bool),valid

    def test_shared_identity_and_common_initialization(self):
        sizes={f:12 for f in FIELDS}
        torch.manual_seed(5);a=MissAdaptation(sizes,width=4,k=4)
        torch.manual_seed(5);b=MissAdaptation(sizes,width=4,k=4,variant='content_separate')
        for key,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[key])
        shared=MissAdaptation(sizes,width=4,k=4,variant='hybrid_shared')
        self.assertIs(shared.co_attention,shared.mm_attention)
        self.assertIsNot(a.co_attention,a.mm_attention)

    def test_selection_consistency_and_gradients(self):
        torch.manual_seed(7);a=MissAdaptation({f:12 for f in FIELDS},width=4,k=4)
        args=self.inputs();features,item,category,valid,tc,hc,ta,ha=args
        target=a.history_vectors(features[:,6],features[:,7])
        routes=a.select(target,item,category,valid,tc,hc,ta,ha)
        idx,mask=routes[0]
        full=a.co_attention.scores(target,a.history_vectors(item,category))
        selected=a.history_vectors(item.gather(1,idx),category.gather(1,idx))
        torch.testing.assert_close(a.co_attention.scores(target,selected),full.gather(1,idx))
        out=a(*args);self.assertTrue(torch.isfinite(out).all());out.sum().backward()
        for attention in (a.co_attention,a.mm_attention):
            for layer in (attention.query,attention.key,attention.value):
                self.assertGreater(layer.weight.grad.abs().sum().item(),0)
        self.assertIsNone(a.long_attention.query.weight.grad)
        self.assertTrue(a.embeddings['205'].weight.grad.is_sparse)

    def test_content_partition_and_empty_history(self):
        a=MissAdaptation({f:12 for f in FIELDS},width=4,k=4,variant='content_separate')
        out=a(*self.inputs())
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(a.last_budget['route_positions'].item(),4)
        self.assertEqual(a.last_budget['unique_positions'].item(),4)
        self.assertEqual(a.last_budget['overlap_positions'].item(),0)

    def test_checkpoint_roundtrip(self):
        sizes={f:12 for f in FIELDS};a=MissAdaptation(sizes,width=4,k=4,variant='hybrid_shared')
        b=MissAdaptation(sizes,width=4,k=4,variant='hybrid_shared');b.load_state_dict(a.state_dict())
        args=self.inputs();torch.testing.assert_close(a(*args),b(*args),rtol=0,atol=0)


if __name__=='__main__':unittest.main()
