import sys
import unittest
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'extensions'),str(root/'src')]
from din import LocalActivation,ShortInterest
from feature_store import FIELDS


class DINExtension(unittest.TestCase):
    def test_duplicate_behavior_preserves_intensity(self):
        torch.manual_seed(1);a=LocalActivation(4).eval()
        q=torch.randn(2,4);h=torch.randn(2,1,4);mask=torch.ones(2,1,dtype=torch.bool)
        one=a(q,h,mask);two=a(q,h.repeat(1,2,1),mask.repeat(1,2))
        torch.testing.assert_close(two,2*one)
        self.assertGreater(one.abs().sum().item(),0)

    def test_padding_does_not_change_scores_or_batch_stats(self):
        torch.manual_seed(2);a=LocalActivation(4);b=LocalActivation(4);b.load_state_dict(a.state_dict())
        q=torch.randn(2,4);h=torch.randn(2,3,4);valid=torch.ones(2,3,dtype=torch.bool)
        expected=a(q,h,valid)
        padded=torch.cat([h,torch.randn(2,2,4)*1000],1)
        actual=b(q,padded,torch.cat([valid,torch.zeros(2,2,dtype=torch.bool)],1))
        torch.testing.assert_close(expected,actual)
        torch.testing.assert_close(a.score[1].norm.running_mean,b.score[1].norm.running_mean)

    def test_empty_and_singleton_are_finite(self):
        a=LocalActivation(4);q=torch.randn(1,4);h=torch.randn(1,2,4)
        torch.testing.assert_close(a(q,h,torch.zeros(1,2,dtype=torch.bool)),torch.zeros(1,4))
        self.assertTrue(torch.isfinite(a(q,h,torch.tensor([[True,False]]))).all())

    def test_common_initialization_and_short_only_gradients(self):
        sizes={f:20 for f in FIELDS}
        torch.manual_seed(3);a=ShortInterest(sizes,short_length=2)
        torch.manual_seed(3);b=ShortInterest(sizes,short_length=2,mechanism='dot_softmax')
        for f in FIELDS:torch.testing.assert_close(a.embeddings[f].weight,b.embeddings[f].weight,rtol=0,atol=0)
        for x,y in zip(a.tower.parameters(),b.tower.parameters()):torch.testing.assert_close(x,y,rtol=0,atol=0)
        features=torch.randint(2,20,(3,10));item=torch.randint(2,20,(3,5));cat=torch.randint(2,20,(3,5));valid=torch.ones(3,5,dtype=torch.bool)
        a.eval();pred=a(features,item,cat,valid)
        changed=item.clone();changed[:,:-2]=0
        torch.testing.assert_close(pred,a(features,changed,cat,valid),rtol=0,atol=0)
        pred.sum().backward()
        self.assertGreater(a.short_attention.score[0].weight.grad.abs().sum().item(),0)
        self.assertGreater(a.embeddings['205'].weight.grad.coalesce().values().abs().sum().item(),0)
        self.assertIsNone(a.long_attention.query.weight.grad)
        state=a.state_dict();c=ShortInterest(sizes,short_length=2).eval();c.load_state_dict(state)
        torch.testing.assert_close(pred,c(features,item,cat,valid),rtol=0,atol=0)


if __name__=='__main__':unittest.main()
