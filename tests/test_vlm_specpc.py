from __future__ import annotations

import unittest

import torch

from draft_approx_llm.vlm.attention import (
    SpecPCScoringConfig,
    aggregate_specpc_attention,
)
from draft_approx_llm.vlm.generation import PreparedTargetInput, select_target_input
from draft_approx_llm.vlm.layout import MultimodalTokenLayout


class AttentionAggregationTest(unittest.TestCase):
    def test_weighting_layer_skip_and_future_query(self) -> None:
        skipped = torch.zeros(1, 1, 6, 6)
        used = torch.zeros(1, 2, 6, 6)
        used[0, 0, 4, 0] = 0.8
        used[0, 0, 5, 0] = 0.1
        used[0, 0, 5, 1] = 0.7
        future_used = torch.zeros(1, 2, 1, 7)
        future_used[0, 1, 0, 2] = 0.9

        current, future, future_only, keep_start = aggregate_specpc_attention(
            ((skipped, used), (skipped[:, :, :1, :], future_used)),
            input_length=6,
            config=SpecPCScoringConfig(
                query_window=2, starting_layer=1, weighted_query=True
            ),
        )

        self.assertEqual(keep_start, 4)
        self.assertTrue(torch.allclose(current[:4], torch.tensor([0.4, 0.7, 0.0, 0.0])))
        self.assertAlmostEqual(float(future_only[2]), 0.9, places=6)
        self.assertTrue(torch.equal(future, torch.maximum(current, future_only)))
        self.assertTrue(torch.equal(current[4:], torch.zeros(2)))


class LayoutSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = MultimodalTokenLayout(
            input_ids=torch.arange(9).view(1, 9),
            visual_indices=torch.tensor([1, 2, 3, 6, 7]),
            non_visual_indices=torch.tensor([0, 4, 5, 8]),
            image_visual_indices=[torch.tensor([1, 2, 3]), torch.tensor([6, 7])],
            image_grid_thw=torch.tensor([[1, 3, 1], [1, 2, 1]]),
            spatial_merge_size=1,
            vision_start_indices=torch.tensor([0, 5]),
            vision_end_indices=torch.tensor([4, 8]),
        )

    def test_avg_then_neighbor_pool_stays_within_image(self) -> None:
        scores = torch.zeros(9)
        scores[3] = 1.0
        pooled = self.layout.pool_visual_scores(scores, average_kernel=1, neighbor_kernel=3)
        self.assertGreater(float(pooled[2]), 0.0)
        self.assertEqual(float(pooled[6]), 0.0)

    def test_mandatory_visual_window_token_is_kept(self) -> None:
        scores = torch.tensor([0.0, 9.0, 8.0, 7.0, 0.0, 0.0, 6.0, 0.0, 0.0])
        selected = self.layout.select_visual_tokens_by_count(
            scores, keep_count=2, always_keep_indices=torch.tensor([7, 8])
        )
        self.assertIn(7, selected.tolist())
        self.assertEqual(sum(i in {1, 2, 3, 6, 7} for i in selected.tolist()), 2)


class PositionStrategyTest(unittest.TestCase):
    def test_mrope_and_contiguous_strategies(self) -> None:
        positions = torch.tensor(
            [
                [[0, 1, 2, 3, 4]],
                [[0, 4, 5, 6, 7]],
                [[0, 8, 9, 10, 11]],
            ]
        )
        prepared = PreparedTargetInput(
            inputs_embeds=torch.zeros(1, 5, 4),
            attention_mask=torch.ones(1, 5, dtype=torch.long),
            position_ids=positions,
            input_ids=torch.arange(5).view(1, 5),
            vision_encoder_ms=0.0,
        )
        keep = torch.tensor([0, 2, 4])

        mrope = select_target_input(prepared, keep, "mrope")
        contiguous = select_target_input(prepared, keep, "contiguous")

        self.assertTrue(torch.equal(mrope.position_ids, positions.index_select(2, keep)))
        expected = torch.arange(3).view(1, 1, 3).expand(3, 1, 3)
        self.assertTrue(torch.equal(contiguous.position_ids, expected))


if __name__ == "__main__":
    unittest.main()
