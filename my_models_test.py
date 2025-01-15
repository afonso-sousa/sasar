import unittest

import torch
from transformers import BertConfig

import my_models


class MyModelsTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self._bert_test_config = BertConfig(
            attention_probs_dropout_prob=0.0,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            hidden_size=16,
            initializer_range=0.02,
            intermediate_size=32,
            max_position_embeddings=128,
            num_attention_heads=2,
            num_hidden_layers=2,
            type_vocab_size=2,
            vocab_size=30522,
        )
        self._bert_test_config.num_classes = 20
        self._bert_test_config.query_size = 23
        self._bert_test_config.query_transformer = True

    def test_pretrain_model_insertion(self):
        model = my_models.get_insertion_model(self._bert_test_config)
        self.assertIsInstance(model, torch.nn.Module)

    def test_pretrain_model_tagging(self):
        model = my_models.get_tagging_model(
            self._bert_test_config, seq_length=5, use_pointing=True
        )
        self.assertIsInstance(model, torch.nn.Module)


if __name__ == "__main__":
    unittest.main()
