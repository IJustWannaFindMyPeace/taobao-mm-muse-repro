import sys,unittest,io
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from enhanced_esu import LearnedSimTier,fuse_scores,semantic_attention
from c0 import TargetAttention
from feature_store import FIELDS
from search_model import SearchModel
from muse_model import MUSEModel


class ESUContracts(unittest.TestCase):
    def test_statistics_only_has_no_long_id_path(self):
        sizes={f:12 for f in FIELDS}
        model=MUSEModel(sizes,k=3,short_length=2,semantic=False,histogram=True,long_id=False)
        f=torch.full((2,10),2,dtype=torch.long);it=torch.tensor([[2,3,4,5,6],[3,4,5,6,7]])
        valid=it!=0;q=torch.randn(2,128);h=torch.randn(2,5,128);target_valid=torch.ones(2,dtype=torch.bool)
        with torch.no_grad():model.histogram_projection.weight.normal_(std=.1)
        out=model(f,it,it,valid,q,h,target_valid,valid)
        changed=it.clone();changed[:,:3]=10
        torch.testing.assert_close(out,model(f,changed,changed,valid,q,h,target_valid,valid),rtol=0,atol=0)
        out.sum().backward()
        self.assertTrue(all(p.grad is None for p in model.long_attention.parameters()))
        self.assertIsNone(model.value.weight.grad)
        self.assertGreater(model.histogram_projection.weight.grad.abs().sum().item(),0)

    def test_enhanced_checkpoint_roundtrip(self):
        sizes={f:12 for f in FIELDS};model=MUSEModel(sizes,k=3)
        f=torch.full((2,10),2,dtype=torch.long);it=torch.tensor([[2,3,4],[3,4,5]])
        valid=it!=0
        args=(f,it,it,valid,torch.randn(2,128),torch.randn(2,3,128),torch.ones(2,dtype=torch.bool),valid)
        with torch.no_grad():
            model.gamma.copy_(torch.tensor([.7,1.2,-.3]))
            model.histogram.scale.fill_(.8);model.histogram_projection.weight.normal_(std=.1)
        expected=model(*args)
        stream=io.BytesIO();torch.save(model.state_dict(),stream);stream.seek(0)
        restored=MUSEModel(sizes,k=3);restored.load_state_dict(torch.load(stream,weights_only=True))
        torch.testing.assert_close(expected,restored(*args),rtol=0,atol=0)

    def test_fusion_and_mask_after_negative_coefficient(self):
        d=torch.tensor([[2.,3.]]);r=torch.tensor([[.5,-.5]])
        torch.testing.assert_close(fuse_scores(d,r,torch.tensor([1.,2.,3.])),torch.tensor([[6.,-2.5]]))
        a=TargetAttention(2);v=torch.nn.Linear(2,2,bias=False);torch.nn.init.eye_(v.weight)
        target=torch.ones(2,2);history=torch.randn(2,3,2);mask=torch.tensor([[True,False,False],[False]*3])
        out=semantic_attention(a,v,target,history,mask,torch.ones(2,3),torch.tensor([-1.,2.,-3.]))
        torch.testing.assert_close(out[0],history[0,0]);self.assertEqual(out[1].abs().sum().item(),0)

    def test_bins_endpoints_missing_and_gradients(self):
        model=LearnedSimTier();s=torch.tensor([[-1.,0.,1.,1.]])
        out=model(s,torch.tensor([[True,True,True,False]]))
        self.assertEqual(out.shape,(1,22));self.assertEqual(torch.count_nonzero(out).item(),3)
        torch.testing.assert_close(out[0,[0,10,20]],torch.log(torch.tensor([2.,2.,2.])))
        self.assertEqual(out[0,21].item(),0);out.sum().backward()
        self.assertGreater(model.scale.grad.abs().sum().item(),0)

    def test_common_initial_predictions_and_learning(self):
        sizes={f:12 for f in FIELDS};f=torch.full((2,10),2,dtype=torch.long)
        it=torch.tensor([[2,3,4,0],[0,0,0,0]]);valid=it!=0
        args=(f,it,it,valid,torch.randn(2,128),torch.randn(2,4,128),torch.ones(2,dtype=torch.bool),valid)
        torch.manual_seed(42);base=SearchModel(sizes,k=3)
        for semantic,histogram in [(False,False),(False,True),(True,False),(True,True)]:
            torch.manual_seed(42);model=MUSEModel(sizes,k=3,semantic=semantic,histogram=histogram)
            torch.testing.assert_close(base(*args),model(*args),rtol=1e-6,atol=1e-7)
            model(*args).sum().backward()
            self.assertGreater(model.value.weight.grad.abs().sum().item(),0)
            if semantic:self.assertGreater(model.gamma.grad.abs().sum().item(),0)
            if histogram:
                self.assertGreater(model.histogram_projection.weight.grad.abs().sum().item(),0)
                model.zero_grad(set_to_none=True)
                with torch.no_grad():model.histogram_projection.weight.fill_(.01)
                model(*args).sum().backward()
                self.assertGreater(model.histogram.scale.grad.abs().sum().item(),0)


if __name__=='__main__':unittest.main()
