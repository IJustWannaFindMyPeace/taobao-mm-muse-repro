import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from feature_store import FIELDS
from muse_model import MUSEModel
from baseline_model import BaselineModel


class BaselineModels(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.sizes = {f: 20 for f in FIELDS}
        self.inputs = [torch.randint(2, 20, (3, 10)), torch.randint(2, 20, (3, 8)),
                       torch.randint(2, 20, (3, 8)), torch.ones(3, 8, dtype=torch.bool),
                       torch.randn(3, 5), torch.randn(3, 8, 5),
                       torch.ones(3, dtype=torch.bool), torch.ones(3, 8, dtype=torch.bool)]

    def test_reference_predictions_and_gradients_equal(self):
        original = MUSEModel(self.sizes, k=3, short_length=2)
        reference = BaselineModel(self.sizes, 'content_reference', k=3, short_length=2)
        reference.load_state_dict(original.state_dict())
        a = original(*self.inputs); b = reference(*self.inputs)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        a.sum().backward(); b.sum().backward()
        for (_, x), (_, y) in zip(original.named_parameters(), reference.named_parameters()):
            if x.grad is None:
                self.assertIsNone(y.grad)
            else:
                xg = x.grad.to_dense() if x.grad.is_sparse else x.grad
                yg = y.grad.to_dense() if y.grad.is_sparse else y.grad
                torch.testing.assert_close(xg, yg, rtol=0, atol=0)

    def test_short_ignores_old_history_and_content(self):
        model = BaselineModel(self.sizes, 'short', k=3, short_length=2)
        a = model(*self.inputs)
        changed = [x.clone() for x in self.inputs]
        changed[1][:, :-2] = 0; changed[2][:, :-2] = 0; changed[3][:, :-2] = False
        changed[4].zero_(); changed[5].zero_()
        torch.testing.assert_close(a, model(*changed), rtol=0, atol=0)
        a.sum().backward()
        self.assertIsNone(model.long_attention.query.weight.grad)

    def test_id_paths_update_shared_embeddings_and_handle_empty(self):
        for variant in ['id_item', 'id_item_category', 'category', 'content_id_cosine']:
            model = BaselineModel(self.sizes, variant, k=3, short_length=2)
            logits = model(*self.inputs)
            logits.sum().backward()
            self.assertTrue(torch.isfinite(logits).all())
            self.assertGreater(model.embeddings['205'].weight.grad.coalesce().values().abs().sum().item(), 0)
            empty = [x.clone() for x in self.inputs]
            empty[1].zero_(); empty[2].zero_(); empty[3].zero_(); empty[7].zero_()
            self.assertTrue(torch.isfinite(model(*empty)).all())


if __name__ == '__main__':
    unittest.main()
