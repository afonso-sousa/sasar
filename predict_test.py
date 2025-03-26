import json
import random
import tempfile
import unittest

import torch
from transformers import BertConfig

import predict


def _convert_to_one_hot(labels, vocab_size):
    one_hots = []
    for label in labels:
        one_hot = torch.zeros(vocab_size)
        one_hot[label] = 1
        one_hots.append(one_hot)
    return torch.stack(one_hots)


class DummyPredictorTagging:
    def __init__(self, pred, raw_points=None):
        """Initializer for a dummy predictor.

        Args:
          pred: The predicted tag.
          raw_points: Logits for the pointer network (X,X) matrix. If None, then this value won't be returned when calling the predictor.
        """
        self._pred = pred
        self._raw_points = raw_points

    def __call__(self, input_ids, attention_mask, token_type_ids, edit_tags=None):
        del input_ids, attention_mask, token_type_ids
        if edit_tags:
            del edit_tags
        if self._raw_points is not None:
            return self._pred.unsqueeze(0), self._raw_points.unsqueeze(0)
        else:
            return self._pred.unsqueeze(0)


class DummyPredictorInsertion:
    def __init__(self, prediction):
        """Initializer for a dummy predictor.

        Args:
          prediction: Predicted tokens
        """
        self._prediction = prediction

    def __call__(self, input_ids, attention_mask, token_type_ids):
        del input_ids, attention_mask, token_type_ids
        return (self._prediction,)


class TestPredictUtils(unittest.TestCase):
    def setUp(self):
        self._vocab_tokens = [
            "NOTHING",
            "[CLS]",
            "[SEP]",
            "[PAD]",
            "[UNK]",
            "[MASK]",
            "[unused1]",
            "[unused2]",
            "a",
            "b",
            "c",
            "d",
        ]

        with tempfile.NamedTemporaryFile(delete=False) as vocab_file:
            vocab_file.write("\n".join(self._vocab_tokens).encode())
        self._vocab_file = vocab_file.name
        self._vocab_to_id = {
            vocab_token: i for i, vocab_token in enumerate(self._vocab_tokens)
        }

        self._label_map = {"PAD": 0, "KEEP": 1, "DELETE": 2, "KEEP|1": 3}
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as label_map_file:
            json.dump(self._label_map, label_map_file)
        self._label_map_path = label_map_file.name

        self._max_sequence_length = 30
        self._bert_test_tagging_config = BertConfig(
            attention_probs_dropout_prob=0.0,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            hidden_size=16,
            initializer_range=0.02,
            intermediate_size=32,
            max_position_embeddings=40,
            num_attention_heads=1,
            num_hidden_layers=1,
            type_vocab_size=3,
            vocab_size=len(self._vocab_tokens),
        )
        self._bert_test_tagging_config.num_classes = len(self._label_map)
        self._bert_test_tagging_config.query_size = 23
        self._bert_test_tagging_config.pointing = False
        self._bert_test_tagging_config.query_transformer = False

    def test_predict_end_to_end_batch_random(self):
        """Test the model predictions end-2-end with randomly initialized models."""
        batch_size = 11
        felix_predictor = predict.Predictor(
            bert_config_insertion=self._bert_test_tagging_config,
            bert_config_tagging=self._bert_test_tagging_config,
            vocab_file=self._vocab_file,
            model_tagging_filepath=None,
            model_insertion_filepath=None,
            label_map_file=self._label_map_path,
            sequence_length=self._max_sequence_length,
            is_pointing=True,
            do_lowercase=True,
            use_open_vocab=True,
        )
        source_batch = []

        for i in range(batch_size):
            source_batch.append(
                " ".join(random.choices(self._vocab_tokens[8:], k=i + 1))
            )
        # Uses a randomly initialized tagging model.
        (
            predictions_tagging,
            predictions_insertion,
        ) = felix_predictor.predict_end_to_end_batch(source_batch)
        self.assertEqual(len(predictions_tagging), batch_size)
        self.assertEqual(len(predictions_insertion), batch_size)

    def test_predict_end_to_end_batch_fake(self):
        """Test end-to-end with fake PyTorch models."""
        test_cases = [
            # Straightforward.
            {
                "pred": [1, 1, 1, 1, 1],
                "sources": ["a b c"],
                "gold": "[CLS] a b c [SEP]",
                "insertions": ["NOTHING"],
                "gold_with_deletions": "[CLS] a b c [SEP]",
                "raw_points": torch.tensor(
                    [
                        [0, 10, 0, 0, 0, 0],
                        [0, 0, 10, 0, 0, 0],
                        [0, 0, 0, 10, 0, 0],
                        [0, 0, 0, 0, 10, 0],
                        [10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
            },
            # With deletion.
            {
                "pred": [1, 2, 1, 1, 1],
                "sources": ["a b c"],
                "gold": "[CLS] b c [SEP]",
                "insertions": ["NOTHING"],
                "gold_with_deletions": "[CLS] [unused1] a [unused2] b c [SEP]",
                "raw_points": torch.tensor(
                    [
                        [0, 0, 10, 0, 0, 0],
                        [10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 10, 0, 0],
                        [0, 0, 0, 0, 10, 0],
                        [10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
            },
            # With deletion and insertion.
            {
                "pred": [3, 2, 1, 1, 1],
                "sources": ["a b c"],
                "gold": "[CLS] d b c [SEP]",
                "insertions": ["d"],
                "gold_with_deletions": "[CLS] [MASK] [unused1] a [unused2] b c [SEP]",
                "raw_points": torch.tensor(
                    [
                        [0, 0, 10, 0, 0, 0],
                        [10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 10, 0, 0],
                        [0, 0, 0, 0, 10, 0],
                        [10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
            },
        ]

        for idx, test_case in enumerate(test_cases):
            with self.subTest(f"Test case {idx+1}"):
                pred = test_case["pred"]
                raw_points = test_case["raw_points"]
                sources = test_case["sources"]
                gold = test_case["gold"]
                gold_with_deletions = test_case["gold_with_deletions"]
                insertions = test_case["insertions"]

                felix_predictor = predict.Predictor(
                    bert_config_insertion=self._bert_test_tagging_config,
                    bert_config_tagging=self._bert_test_tagging_config,
                    vocab_file=self._vocab_file,
                    model_tagging_filepath=None,
                    model_insertion_filepath=None,
                    label_map_file=self._label_map_path,
                    sequence_length=self._max_sequence_length,
                    is_pointing=True,
                    do_lowercase=True,
                    use_open_vocab=True,
                )
                tagging_model = DummyPredictorTagging(
                    _convert_to_one_hot(pred, len(self._label_map)), raw_points
                )
                insertions = torch.stack(
                    [
                        _convert_to_one_hot(
                            [self._vocab_to_id[token] for token in insertions],
                            len(self._vocab_tokens),
                        )
                    ]
                )
                insertion_model = DummyPredictorInsertion(insertions)
                felix_predictor._tagging_model = tagging_model
                felix_predictor._insertion_model = insertion_model
                (
                    taggings_outputs,
                    insertion_outputs,
                ) = felix_predictor.predict_end_to_end_batch(sources)
                self.assertEqual(taggings_outputs[0], gold_with_deletions)
                self.assertEqual(insertion_outputs[0], gold)

    def test_convert_source_sentences_into_tagging_batch(self):
        batch_size = 11
        felix_predictor = predict.Predictor(
            bert_config_insertion=self._bert_test_tagging_config,
            bert_config_tagging=self._bert_test_tagging_config,
            model_tagging_filepath=None,
            model_insertion_filepath=None,
            label_map_file=self._label_map_path,
            sequence_length=self._max_sequence_length,
            is_pointing=True,
            do_lowercase=True,
            vocab_file=self._vocab_file,
            use_open_vocab=True,
        )
        source_batch = []
        for i in range(batch_size):
            # Produce random sentences from the vocab (excluding special tokens).
            source_batch.append(
                " ".join(random.choices(self._vocab_tokens[7:], k=i + 1))
            )
        (
            batch_dictionaries,
            _,
        ) = felix_predictor._convert_source_sentences_into_batch(
            source_batch, is_insertion=False
        )

        self.assertEqual(len(batch_dictionaries), batch_size)

    def test_convert_source_sentences_into_insertion_batch(self):
        batch_size = 11
        felix_predictor = predict.Predictor(
            bert_config_insertion=self._bert_test_tagging_config,
            bert_config_tagging=self._bert_test_tagging_config,
            vocab_file=self._vocab_file,
            model_tagging_filepath=None,
            model_insertion_filepath=None,
            label_map_file=self._label_map_path,
            sequence_length=self._max_sequence_length,
            is_pointing=True,
            do_lowercase=True,
            use_open_vocab=True,
        )
        source_batch = []
        for i in range(batch_size):
            # Produce random sentences from the vocab (excluding special tokens).
            source_batch.append(
                " ".join(random.choices(self._vocab_tokens[7:], k=i + 1))
            )

        (
            batch_dictionaries,
            batch_list,
        ) = felix_predictor._convert_source_sentences_into_batch(
            source_batch, is_insertion=True
        )
        # Each input should be of the size (batch_size, max_sequence_length).
        self.assertEqual(
            batch_list["input_ids"].shape,
            batch_list["attention_mask"].shape,
        )
        self.assertEqual(
            batch_list["attention_mask"].shape,
            batch_list["token_type_ids"].shape,
        )

        self.assertEqual(len(batch_dictionaries), batch_size)

    def test_predict_tagging_batch(self):
        batch_size = 11
        felix_predictor = predict.Predictor(
            bert_config_insertion=self._bert_test_tagging_config,
            bert_config_tagging=self._bert_test_tagging_config,
            model_tagging_filepath=None,
            model_insertion_filepath=None,
            label_map_file=self._label_map_path,
            sequence_length=self._max_sequence_length,
            is_pointing=True,
            vocab_file=self._vocab_file,
            do_lowercase=True,
            use_open_vocab=True,
        )
        source_batch = []
        for i in range(batch_size):
            source_batch.append(
                " ".join(random.choices(self._vocab_tokens[7:], k=i + 1))
            )
        # Uses a randomly initialized tagging model.
        predictions = felix_predictor._predict_batch(
            felix_predictor._convert_source_sentences_into_batch(
                source_batch, is_insertion=False
            )[1],
            is_insertion=False,
        )
        self.assertEqual(len(predictions), batch_size)

        for tag_logits, pointing_logits in predictions:
            self.assertEqual(len(tag_logits), len(predictions[0][0]))
            self.assertEqual(
                pointing_logits.shape,
                (len(predictions[0][1]), len(predictions[0][1])),
            )

    def test_predict_insertion_batch(self):
        batch_size = 11
        felix_predictor = predict.Predictor(
            bert_config_insertion=self._bert_test_tagging_config,
            bert_config_tagging=self._bert_test_tagging_config,
            vocab_file=self._vocab_file,
            model_tagging_filepath=None,
            model_insertion_filepath=None,
            label_map_file=self._label_map_path,
            sequence_length=self._max_sequence_length,
            is_pointing=True,
            do_lowercase=True,
            use_open_vocab=True,
        )
        source_batch = []
        for i in range(batch_size):
            source_batch.append(
                " ".join(random.choices(self._vocab_tokens[7:], k=i + 1))
            )
        # Uses a randomly initialized tagging model.
        predictions = felix_predictor._predict_batch(
            felix_predictor._convert_source_sentences_into_batch(
                source_batch, is_insertion=True
            )[1],
            is_insertion=True,
        )
        self.assertEqual(len(predictions), batch_size)

        for prediction in predictions:
            self.assertEqual(len(prediction), len(predictions[0]))

    def test_predict_and_realize_tagging_batch(self):
        test_cases = [
            # Straightforward.
            {
                "pred": [1, 1, 1, 1, 1],
                "sources": ["a b c"],
                "gold": "[CLS] a b c [SEP]",
                "gold_with_deletions": "[CLS] a b c [SEP]",
                "raw_points": torch.tensor(
                    [
                        [0, 10, 0, 0, 0, 0],
                        [0, 0, 10, 0, 0, 0],
                        [0, 0, 0, 10, 0, 0],
                        [0, 0, 0, 0, 10, 0],
                        [10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
            },
            # Go backwards.
            {
                "pred": [1, 1, 1, 1, 1],
                "sources": ["a b c"],
                "gold": "[CLS] c b a [SEP]",
                "gold_with_deletions": "[CLS] c b a [SEP]",
                "raw_points": torch.tensor(
                    [
                        [0, 0, 0, 10, 0, 0],
                        [0, 0, 0, 0, 10, 0],
                        [0, 10, 0, 0, 0, 0],
                        [0, 0, 10, 0, 0, 0],
                        [10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
            },
            # Make everything noisier.
            {
                "pred": [1, 1, 1, 1, 1],
                "sources": ["a b c"],
                "gold": "[CLS] c b a [SEP]",
                "gold_with_deletions": "[CLS] c b a [SEP]",
                "raw_points": torch.tensor(
                    [
                        [2, 3, 4, 10, 5, 6],
                        [2, 3, 4, 5, 10, 6],
                        [2, 10, 3, 4, 5, 6],
                        [2, 3, 10, 4, 5, 7],
                        [10, 2, 4, 5, 6, 7],
                        [1, 1, 1, 1, 1, 1],
                    ]
                ),
            },
            # A tempting start.
            {
                "pred": [1, 1, 1, 1, 1],
                "sources": ["a b c"],
                "gold": "[CLS] a b c [SEP]",
                "gold_with_deletions": "[CLS] a b c [SEP]",
                "raw_points": torch.tensor(
                    [
                        [0, 10, 30, 0, 0, 0],
                        [0, 0, 100, 0, 0, 0],
                        [0, 0, 0, 10, 0, 0],
                        [0, 0, 0, 0, 10, 0],
                        [10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
            },
            # A temptation in the middle.
            {
                "pred": [1, 1, 1, 1, 1],
                "sources": ["a b c"],
                "gold": "[CLS] a b c [SEP]",
                "gold_with_deletions": "[CLS] a b c [SEP]",
                "raw_points": torch.tensor(
                    [
                        [0, 10, 0, 0, 0, 0],
                        [0, 0, 10, 30, 0, 0],
                        [0, 0, 0, 10, 0, 0],
                        [0, 0, 0, 0, 10, 0],
                        [10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
            },
            # Don't revisit the past.
            {
                "pred": [1, 1, 1, 1, 1],
                "sources": ["a b c"],
                "gold": "[CLS] a b c [SEP]",
                "gold_with_deletions": "[CLS] a b c [SEP]",
                "raw_points": torch.tensor(
                    [
                        [0, 10, 0, 0, 0, 0],
                        [0, 6, 5, 0, 0, 0],
                        [0, 0, 0, 10, 0, 0],
                        [0, 0, 0, 0, 10, 0],
                        [10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
            },
            #  No starting place.
            {
                "pred": [1, 1, 1, 1, 1],
                "sources": ["a b c"],
                "gold": "[CLS] a b c [SEP]",
                "gold_with_deletions": "[CLS] a b c [SEP]",
                "raw_points": torch.tensor(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 10, 0, 0, 0],
                        [0, 0, 0, 10, 0, 0],
                        [0, 0, 0, 0, 10, 0],
                        [10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
            },
            # Lost in the middle.
            {
                "pred": [1, 1, 1, 1, 1],
                "sources": ["a b c"],
                "gold": "[CLS] a b c [SEP]",
                "gold_with_deletions": "[CLS] a b c [SEP]",
                "raw_points": torch.tensor(
                    [
                        [0, 10, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 10, 0, 0],
                        [0, 0, 0, 0, 10, 0],
                        [10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
            },
            # Skip to the end.
            {
                "pred": [1, 1, 1, 1, 1],
                "sources": ["a b c"],
                "gold": "[CLS] a b c [SEP]",
                "gold_with_deletions": "[CLS] a b c [SEP]",
                "raw_points": torch.tensor(
                    [
                        [0, 1, 0, 0, 100, 0],
                        [0, 0, 10, 0, 0, 0],
                        [0, 0, 0, 10, 0, 0],
                        [0, 0, 0, 0, 10, 0],
                        [10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
            },
            # Skip past the end.
            {
                "pred": [1, 1, 1, 1, 1],
                "sources": ["a b c"],
                "gold_with_deletions": "[CLS] a b c [SEP]",
                "gold": "[CLS] a b c [SEP]",
                "raw_points": torch.tensor(
                    [
                        [0, 1, 0, 0, 100, 0],
                        [0, 0, 10, 0, 0, 0],
                        [0, 0, 0, 10, 0, 0],
                        [0, 0, 0, 0, 0, 100],
                        [10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 10, 0],
                    ]
                ),
            },
            {
                # Don't visit that!
                "pred": [1, 0, 1, 1, 1],
                "sources": ["a b c"],
                "gold": "[CLS] b c [SEP]",
                "gold_with_deletions": "[CLS] [unused1] a [unused2] b c [SEP]",
                "raw_points": torch.tensor(
                    [
                        [0, 10, 1, 0, 0, 0],
                        [0, 0, 10, 0, 0, 0],
                        [0, 0, 0, 10, 0, 0],
                        [0, 0, 0, 0, 10, 0],
                        [10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
            },
            # Straightforward with multiple SEPs.
            {
                "pred": [1, 1, 1, 1, 1],
                "sources": ["a [SEP] c"],
                "gold": "[CLS] a [SEP] c [SEP]",
                "gold_with_deletions": "[CLS] a [SEP] c [SEP]",
                "raw_points": torch.tensor(
                    [
                        [0, 10, 0, 0, 0, 0],
                        [0, 0, 10, 0, 0, 0],
                        [0, 0, 0, 10, 0, 0],
                        [0, 0, 0, 0, 10, 0],
                        [10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
            },
            # Last SEP becomes middle SEP.
            {
                "pred": [1, 1, 1, 1, 1],
                "sources": ["a [SEP] c"],
                "gold": "[CLS] a [SEP] c [SEP]",
                "gold_with_deletions": "[CLS] a [SEP] c [SEP]",
                "raw_points": torch.tensor(
                    [
                        [0, 10, 0, 0, 0, 0],
                        [0, 0, 0, 0, 10, 0],
                        [10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 10, 0],
                        [0, 0, 0, 10, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
            },
            # Delete middle SEP.
            {
                "pred": [1, 1, 2, 1, 1],
                "sources": ["a [SEP] c"],
                "gold": "[CLS] a c [SEP]",
                "gold_with_deletions": "[CLS] a [unused1] [SEP] [unused2] c [SEP]",
                "raw_points": torch.tensor(
                    [
                        [0, 10, 0, 0, 0, 0],
                        [0, 0, 0, 10, 0, 0],
                        [10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 10, 0],
                        [10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
            },
            # Delete last SEP.
            {
                "pred": [1, 1, 1, 1, 2],
                "sources": ["a [SEP] c"],
                "gold": "[CLS] a c [SEP]",
                "gold_with_deletions": "[CLS] a c [unused1] [SEP] [unused2] [SEP]",
                "raw_points": torch.tensor(
                    [
                        [0, 10, 0, 0, 0, 0],
                        [0, 0, 10, 0, 0, 0],
                        [10, 0, 0, 0, 0, 0],
                        [0, 0, 10, 0, 0, 0],
                        [10, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
            },
        ]
        for idx, test_case in enumerate(test_cases):
            with self.subTest(f"Test case {idx+1}"):
                pred = test_case["pred"]
                raw_points = test_case["raw_points"]
                sources = test_case["sources"]
                gold_with_deletions = test_case["gold_with_deletions"]

                felix_predictor = predict.Predictor(
                    bert_config_insertion=self._bert_test_tagging_config,
                    bert_config_tagging=self._bert_test_tagging_config,
                    model_tagging_filepath=None,
                    model_insertion_filepath=None,
                    label_map_file=self._label_map_path,
                    sequence_length=self._max_sequence_length,
                    is_pointing=True,
                    do_lowercase=True,
                    vocab_file=self._vocab_file,
                    use_open_vocab=True,
                )
                tagging_model = DummyPredictorTagging(
                    _convert_to_one_hot(pred, len(self._label_map)), raw_points
                )
                felix_predictor._tagging_model = tagging_model
                realized_predictions = felix_predictor._predict_and_realize_batch(
                    sources, is_insertion=False
                )
                self.assertEqual(realized_predictions[0], gold_with_deletions)

    def test_predict_and_realize_insertion_batch(self):
        """Test predicting and realizing insertion with fake PyTorch models."""
        test_cases = [
            # No insertions.
            {
                "prediction": ["NOTHING"],
                "sources": ["[CLS] a b c [SEP]"],
                "gold": "[CLS] a b c [SEP]",
            },
            {
                "prediction": ["b"],
                "sources": ["[CLS] a [MASK] c [SEP]"],
                "gold": "[CLS] a b c [SEP]",
            },
            {
                "prediction": ["b", "c"],
                "sources": ["[CLS] a [MASK] [MASK] [SEP]"],
                "gold": "[CLS] a b c [SEP]",
            },
            {
                "prediction": ["a", "c"],
                "sources": ["[CLS] [MASK] b [MASK] [SEP]"],
                "gold": "[CLS] a b c [SEP]",
            },
            {
                "prediction": ["c"],
                "sources": ["[CLS] [unused1] a [unused2] b [MASK] [SEP]"],
                "gold": "[CLS] b c [SEP]",
            },
        ]

        for idx, test_case in enumerate(test_cases):
            with self.subTest(f"Test case {idx+1}"):
                prediction = test_case["prediction"]
                sources = test_case["sources"]
                gold = test_case["gold"]

                prediction = torch.stack(
                    [
                        _convert_to_one_hot(
                            [self._vocab_to_id[token] for token in prediction],
                            len(self._vocab_tokens),
                        )
                    ]
                )

                felix_predictor = predict.Predictor(
                    bert_config_insertion=self._bert_test_tagging_config,
                    bert_config_tagging=self._bert_test_tagging_config,
                    vocab_file=self._vocab_file,
                    model_tagging_filepath=None,
                    model_insertion_filepath=None,
                    label_map_file=self._label_map_path,
                    sequence_length=self._max_sequence_length,
                    is_pointing=True,
                    do_lowercase=True,
                    use_open_vocab=True,
                )

                insertion_model = DummyPredictorInsertion(prediction)
                felix_predictor._insertion_model = insertion_model
                realized_predictions = felix_predictor._predict_and_realize_batch(
                    sources, is_insertion=True
                )
                self.assertEqual(realized_predictions[0], gold)


if __name__ == "__main__":
    unittest.main()
