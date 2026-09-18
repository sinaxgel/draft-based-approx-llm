from __future__ import annotations

import unittest

import torch

from draft_approx_llm.vlm.diagnostics import (
    pearson_correlation,
    percentile_rank_scores,
    set_overlap,
)


class DiagnosticUtilitiesTest(unittest.TestCase):
    def test_percentile_rank_only_changes_selected_indices(self) -> None:
        scores = torch.tensor([9.0, 3.0, 0.0, 6.0])
        ranked = percentile_rank_scores(scores, torch.tensor([1, 2, 3]))
        self.assertTrue(torch.equal(ranked, torch.tensor([0.0, 0.5, 0.0, 1.0])))

    def test_correlations(self) -> None:
        values = torch.tensor([1.0, 2.0, 3.0])
        self.assertAlmostEqual(pearson_correlation(values, values), 1.0, places=6)
        self.assertAlmostEqual(
            pearson_correlation(values, values.flip(0)), -1.0, places=6
        )

    def test_set_overlap(self) -> None:
        overlap = set_overlap([1, 2, 3], [2, 3, 4])
        self.assertEqual(overlap["intersection"], 2)
        self.assertEqual(overlap["symmetric_difference"], 2)
        self.assertAlmostEqual(overlap["jaccard"], 0.5)


if __name__ == "__main__":
    unittest.main()
