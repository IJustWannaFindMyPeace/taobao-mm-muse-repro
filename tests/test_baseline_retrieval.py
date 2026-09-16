import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from baseline_retrieval import category_history, id_cosine_history
from retrieval import gather_history


class BaselineRetrievalContracts(unittest.TestCase):
    def test_category_recency_duplicates_and_no_backfill(self):
        ids = torch.tensor([[2, 3, 2, 0, 2]])
        idx, mask = category_history(torch.tensor([2]), ids, ids != 0, 5)
        self.assertEqual(idx[mask].tolist(), [4, 2, 0])
        self.assertEqual(int(mask.sum()), 3)

    def test_unknown_categories_never_match(self):
        ids = torch.tensor([[1, 1, 0], [2, 3, 0]])
        _, mask = category_history(torch.tensor([1, 4]), ids, ids != 0, 3)
        self.assertFalse(mask.any())

    def test_id_scores_zero_negative_ties_and_unknown(self):
        q = torch.tensor([[1., 0.]])
        h = torch.tensor([[[0., 1.], [-1., 0.], [0., 1.], [1., 0.], [0., 0.]]])
        ids = torch.tensor([[2, 3, 4, 1, 0]])
        idx, mask = id_cosine_history(q, h, torch.tensor([2]), ids, ids != 0, 5)
        self.assertEqual(idx[mask].tolist(), [2, 0, 1])

    def test_missing_target_fallback_and_empty(self):
        ids = torch.tensor([[2, 1, 3], [0, 0, 0]])
        idx, mask = id_cosine_history(torch.zeros(2, 2), torch.ones(2, 3, 2),
                                     torch.tensor([1, 2]), ids, ids != 0, 2)
        self.assertEqual(idx[0].tolist(), [2, 1])
        self.assertTrue(mask[0].all())
        self.assertFalse(mask[1].any())

    def test_selection_detached_but_shared_embeddings_train(self):
        embedding = torch.nn.Embedding(6, 2, sparse=True)
        with torch.no_grad():
            embedding.weight.copy_(torch.tensor([[0.,0.],[0.,0.],[1.,0.],[0.,1.],[-1.,0.],[1.,1.]]))
        target = torch.tensor([2]); ids = torch.tensor([[3, 4, 5]])
        idx, mask = id_cosine_history(embedding(target), embedding(ids), target, ids, ids != 0, 1)
        self.assertEqual(idx.item(), 2)
        self.assertFalse(idx.requires_grad)
        selected = gather_history(ids, idx, mask)
        loss = ((embedding(selected).sum(1) - embedding(target)) ** 2).sum()
        loss.backward()
        changed = embedding.weight.grad.coalesce().indices()[0].tolist()
        self.assertEqual(set(changed), {2, 5})

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            category_history(torch.tensor([2]), torch.tensor([[2]]), torch.tensor([[True]]), 2)


if __name__ == '__main__':
    unittest.main()
