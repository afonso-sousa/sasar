import unittest

import torch

import beam_search


class BeamSearchTest(unittest.TestCase):
    def test_beam_search_single_tagging(self):
        test_cases = [
            {
                "predicted_points_logits": [
                    [0, 10, 0, 0, 0, 0],
                    [0, 0, 10, 0, 0, 0],
                    [0, 0, 0, 10, 0, 0],
                    [0, 0, 0, 0, 10, 0],
                    [10, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
                "good_indexes": [0, 1, 2, 3, 4, 5],
                "sep_indexes": {5},
                "end_index": 5,
                "best_sequence": [0, 1, 2, 3, 4, 5],
            },
            {
                "predicted_points_logits": [
                    [0, 0, 0, 10, 0, 0],
                    [0, 0, 0, 0, 10, 0],
                    [0, 10, 0, 0, 0, 0],
                    [0, 0, 10, 0, 0, 0],
                    [10, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
                "good_indexes": [0, 1, 2, 3, 4, 5],
                "sep_indexes": {5},
                "end_index": 5,
                "best_sequence": [0, 3, 2, 1, 4, 5],
            },
        ]

        for test_case in test_cases:
            predicted_points_logits = test_case["predicted_points_logits"]
            good_indexes = test_case["good_indexes"]
            sep_indexes = test_case["sep_indexes"]
            end_index = test_case["end_index"]
            best_sequence = test_case["best_sequence"]

            with self.subTest(
                predicted_points_logits=predicted_points_logits,
                good_indexes=good_indexes,
                sep_indexes=sep_indexes,
                end_index=end_index,
                best_sequence=best_sequence,
            ):
                prediction = beam_search.beam_search_single_tagging(
                    torch.tensor(predicted_points_logits),
                    torch.tensor(good_indexes),
                    sep_indexes,
                    end_index,
                )
                assert torch.eq(prediction, torch.tensor(best_sequence)).all()


if __name__ == "__main__":
    unittest.main()
