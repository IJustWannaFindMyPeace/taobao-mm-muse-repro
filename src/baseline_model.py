"""Controlled baseline variants on the existing MUSE backbone, not original DIN/SIM."""
import torch
from muse_model import MUSEModel
from feature_store import FIELDS
from retrieval import select_history, gather_history
from baseline_retrieval import category_history, id_cosine_history
from enhanced_esu import semantic_scores, semantic_attention
from fused_retrieval import fused_cosine_history


class BaselineModel(MUSEModel):
    def __init__(self, sizes, variant, **kwargs):
        if variant not in ['short', 'category', 'id_item', 'id_item_category', 'content_reference', 'content_id_cosine']:
            raise ValueError('Unknown baseline variant')
        super().__init__(sizes, **kwargs)
        self.variant = variant
        if variant == 'short':
            for layer in [self.long_attention, self.value, self.histogram, getattr(self, 'histogram_projection', None)]:
                if layer is not None:
                    layer.requires_grad_(False)
            if self.gamma is not None:
                self.gamma.requires_grad_(False)

    @torch.no_grad()
    def select(self, features, item, category, valid, target_content, history_content,
               target_available, history_available):
        if self.variant == 'category':
            return category_history(features[:, 7], category, valid, self.k)
        if self.variant == 'content_id_cosine':
            return fused_cosine_history(target_content, history_content, valid,
                target_available, history_available, self.embeddings['205'](features[:, 6]),
                self.embeddings['205'](item), features[:, 6], item, self.k)
        if self.variant in ['id_item', 'id_item_category']:
            tid = features[:, 6]
            q = self.embeddings['205'](tid) * (tid > 1)[:, None]
            h = self.embeddings['205'](item) * (item > 1)[..., None]
            if self.variant == 'id_item_category':
                tc = features[:, 7]
                q = torch.cat([q, self.embeddings['206'](tc) * (tc > 1)[:, None]], -1)
                h = torch.cat([h, self.embeddings['206'](category) * (category > 1)[..., None]], -1)
                # A known category can provide retrieval signal for an OOV item.
                tid = torch.where((tid > 1) | (tc > 1), 2, 1)
                item = torch.where((item > 1) | (category > 1), 2, 1)
            return id_cosine_history(q, h, tid, item, valid, self.k)
        return select_history(target_content, history_content, valid, target_available,
                              history_available, self.k, self.mode)

    def forward(self, features, item, category, valid, target_content, history_content,
                target_available, history_available):
        scalar = [self.embeddings[f](features[:, i]) for i, f in enumerate(FIELDS)]
        target = torch.cat([scalar[6], scalar[7]], -1)
        short_hist = torch.cat([self.embeddings['205'](item[:, -self.short_length:]),
                                self.embeddings['206'](category[:, -self.short_length:])], -1)
        short = self.short_attention(target, short_hist, valid[:, -self.short_length:])
        if self.variant == 'short':
            return self.tower(torch.cat(scalar + [short, torch.zeros_like(target)], -1)).squeeze(-1)
        indices, selected = self.select(features, item, category, valid, target_content,
                                        history_content, target_available, history_available)
        # Re-lookup outside selection's no-grad so shared embeddings remain trainable.
        long_hist = torch.cat([self.embeddings['205'](gather_history(item, indices, selected)),
                               self.embeddings['206'](gather_history(category, indices, selected))], -1)
        content = gather_history(history_content, indices, selected)
        available = gather_history(history_available, indices, selected)
        similarity, semantic_valid = semantic_scores(target_content, content, selected,
                                                     target_available, available)
        long = semantic_attention(self.long_attention, self.value, target, long_hist,
                                  selected, similarity, self.gamma)
        first = self.tower[0](torch.cat(scalar + [short, long], -1))
        if self.histogram is not None:
            first = first + self.histogram_projection(self.histogram(similarity, semantic_valid))
        return self.tower[1:](first).squeeze(-1)
