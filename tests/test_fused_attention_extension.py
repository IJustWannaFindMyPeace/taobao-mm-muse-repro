import sys,unittest
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'src'),str(root/'extensions')]
from fused_attention_search import FusedAttentionSearch
from muse_model import MUSEModel
from feature_store import FIELDS


class FusedSearch(unittest.TestCase):
    def args(self):
        f=torch.full((3,len(FIELDS)),2,dtype=torch.long)
        item=torch.tensor([[0,2,3,4,5,6],[0,0,2,3,4,5],[0,0,0,0,0,0]])
        return f,item,item.clone(),item!=0,torch.randn(3,7),torch.randn(3,6,7),torch.ones(3,dtype=torch.bool),item!=0

    def test_content_control_exact_original_forward_and_gradients(self):
        torch.manual_seed(1);base=MUSEModel({f:12 for f in FIELDS},width=4,k=3)
        control=FusedAttentionSearch({f:12 for f in FIELDS},width=4,k=3,retrieval='content');control.load_state_dict(base.state_dict())
        args=self.args();x=base(*args);y=control(*args)
        torch.testing.assert_close(x,y,rtol=0,atol=0)
        x.sum().backward();y.sum().backward()
        for (name,p),(other,q) in zip(base.named_parameters(),control.named_parameters()):
            self.assertEqual(name,other)
            if p.grad is not None:
                a=p.grad.to_dense() if p.grad.is_sparse else p.grad
                b=q.grad.to_dense() if q.grad.is_sparse else q.grad
                torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_shared_scores_train_and_masks(self):
        torch.manual_seed(4);m=FusedAttentionSearch({f:12 for f in FIELDS},width=4,k=3)
        args=self.args();index,valid=m.select(*args)
        self.assertFalse(valid[2].any());self.assertFalse(index.requires_grad)
        y=m(*args);self.assertTrue(torch.isfinite(y).all());y.sum().backward()
        for p in (m.long_attention.query.weight,m.long_attention.key.weight,m.gamma):
            self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(p.grad.abs().sum().item(),0)

    def test_zero_id_coefficient_recovers_content_ranking(self):
        torch.manual_seed(7);m=FusedAttentionSearch({f:12 for f in FIELDS},width=4,k=3)
        with torch.no_grad():m.gamma.copy_(torch.tensor([0.,1.,0.]))
        args=self.args();a=m.select(*args);m.retrieval='content';b=m.select(*args)
        for x,y in zip(a,b):torch.testing.assert_close(x,y)


if __name__=='__main__':unittest.main()
