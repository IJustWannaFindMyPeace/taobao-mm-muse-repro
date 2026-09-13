import sys,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from retrieval import select_history,gather_history
from search_model import SearchModel
from c0 import C0
from feature_store import FIELDS


class RetrievalContracts(unittest.TestCase):
    def test_random_selection_matches_scalar_oracle(self):
        torch.manual_seed(7)
        q=torch.randn(4,8);h=torch.randn(4,17,8)
        valid=torch.rand(4,17)>.2;available=torch.rand(4,17)>.3
        target_ok=torch.tensor([True,False,True,True])
        idx,mask=select_history(q,h,valid,target_ok,available,k=5)
        scores=torch.nn.functional.cosine_similarity(q[:,None],h,dim=-1)
        for row in range(4):
            candidates=[i for i in range(17) if valid[row,i] and (not target_ok[row] or available[row,i])]
            ranked=sorted(candidates,key=lambda i:(float(scores[row,i]) if target_ok[row] else float(i),i),reverse=True)[:5]
            self.assertEqual(idx[row][mask[row]].tolist(),ranked)

    def test_zero_negative_padding_and_ties(self):
        q=torch.tensor([[1.,0.]],requires_grad=True)
        h=torch.tensor([[[0.,1.],[-1.,0.],[0.,0.],[0.,1.]]],requires_grad=True)
        valid=torch.tensor([[True,True,False,True]])
        idx,mask=select_history(q,h,valid,torch.tensor([True]),valid,k=4)
        self.assertEqual(idx.tolist(),[[3,0,1,2]])
        self.assertEqual(mask.tolist(),[[True,True,True,False]])
        self.assertFalse(idx.requires_grad)
        self.assertEqual(gather_history(torch.tensor([[10,20,999,10]]),idx,mask).tolist(),[[10,10,20,0]])

    def test_missing_target_fallback_and_empty(self):
        q=torch.zeros(2,2);h=torch.ones(2,4,2)
        valid=torch.tensor([[True,False,True,True],[False]*4])
        idx,mask=select_history(q,h,valid,torch.zeros(2,dtype=torch.bool),valid,k=3)
        self.assertEqual(idx[0].tolist(),[3,2,0]);self.assertFalse(mask[1].any())

    def test_search_preserves_full_history_model_when_k_equals_l(self):
        sizes={f:12 for f in FIELDS}
        torch.manual_seed(42);a=C0(sizes)
        torch.manual_seed(42);b=SearchModel(sizes,k=4,mode='recent')
        for key,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[key],rtol=0,atol=0)
        f=torch.full((2,10),2,dtype=torch.long);it=torch.tensor([[2,3,4,0],[0,0,0,0]])
        valid=it!=0;q=torch.randn(2,128);h=torch.randn(2,4,128)
        expected=a(f,it,it,valid)
        actual=b(f,it,it,valid,q,h,torch.ones(2,dtype=torch.bool),valid)
        torch.testing.assert_close(expected,actual,rtol=1e-6,atol=1e-7)
        actual.sum().backward()
        self.assertIsNotNone(b.long_attention.query.weight.grad)


if __name__=='__main__':unittest.main()
