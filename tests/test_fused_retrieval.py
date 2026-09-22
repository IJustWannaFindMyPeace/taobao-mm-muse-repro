import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fused_retrieval import fused_cosine_history
from retrieval import select_history


class FusedRetrieval(unittest.TestCase):
    def setUp(self):
        self.q = torch.tensor([[1., 0.]])
        self.h = torch.tensor([[[1., 0.], [.6, .8], [1., 0.]]])
        self.valid = torch.tensor([[True, True, False]])
        self.qa = torch.tensor([True])
        self.ha = torch.ones(1, 3, dtype=torch.bool)
        self.iq = torch.tensor([[1., 0.]], requires_grad=True)
        self.ih = torch.tensor([[[-1., 0.], [1., 0.], [1., 0.]]], requires_grad=True)
        self.tid = torch.tensor([2])
        self.hid = torch.tensor([[3, 4, 0]])

    def select(self, k=1):
        return fused_cosine_history(self.q, self.h, self.valid, self.qa, self.ha,
                                    self.iq, self.ih, self.tid, self.hid, k)

    def test_id_changes_ranking_but_padding_cannot_win(self):
        self.assertEqual(select_history(self.q,self.h,self.valid,self.qa,self.ha,1)[0].item(),0)
        indices, mask = self.select()
        self.assertEqual(indices.item(), 1)
        self.assertTrue(mask.item())
        self.assertIsNone(self.iq.grad)

    def test_unknown_target_id_exactly_reduces_to_content(self):
        self.tid.fill_(1)
        for a,b in zip(self.select(3), select_history(self.q,self.h,self.valid,self.qa,self.ha,3)):
            torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_missing_content_preserves_reference_candidate_set(self):
        self.ha[0,1] = False
        indices,mask=self.select(3)
        self.assertEqual(indices[mask].tolist(),[0])
        self.qa.fill_(False)
        for a,b in zip(self.select(3), select_history(self.q,self.h,self.valid,self.qa,self.ha,3)):
            torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_tie_recency_and_empty_history(self):
        self.h[0,1]=self.h[0,0]
        self.hid.fill_(1)
        self.assertEqual(self.select()[0].item(),1)
        self.valid.zero_()
        self.assertFalse(self.select(3)[1].any())


if __name__ == '__main__':
    unittest.main()
