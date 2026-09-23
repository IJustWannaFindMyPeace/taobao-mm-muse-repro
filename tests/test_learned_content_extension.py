import sys,unittest
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'src'),str(root/'extensions')]
from learned_content_search import ContentScorerPretraining,LearnedContentSearch
from feature_store import FIELDS


class LearnedContent(unittest.TestCase):
    def inputs(self):
        f=torch.full((3,len(FIELDS)),2,dtype=torch.long);i=torch.arange(2,9)[None].repeat(3,1)
        valid=torch.ones(3,7,dtype=torch.bool);valid[2]=False
        return f,i,i,valid,torch.randn(3,128),torch.randn(3,7,128),torch.ones(3,dtype=torch.bool),valid

    def test_encoder_is_trained_by_click_objective(self):
        torch.manual_seed(42);m=ContentScorerPretraining();args=self.inputs()
        loss=torch.nn.functional.binary_cross_entropy_with_logits(m(*args),torch.tensor([0.,1.,0.]))
        loss.backward()
        for layer in (m.projection.layers[0],m.projection.layers[2]):
            self.assertTrue(torch.isfinite(layer.weight.grad).all());self.assertGreater(layer.weight.grad.abs().sum().item(),0)

    def test_frozen_projection_and_unchanged_esu(self):
        torch.manual_seed(8);m=LearnedContentSearch({f:12 for f in FIELDS},width=4,k=3)
        args=self.inputs();y=m(*args);self.assertTrue(torch.isfinite(y).all());y.sum().backward()
        self.assertTrue(all(p.grad is None for p in m.projection.parameters()))
        self.assertGreater(m.long_attention.query.weight.grad.abs().sum().item(),0)
        idx,mask=m.select(*args);self.assertFalse(mask[2].any())


if __name__=='__main__':unittest.main()
