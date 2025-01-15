import unittest

import torch
from transformers import BertConfig, BertModel

import my_tagger


class FelixTaggerTest(unittest.TestCase):
    def test_forward_pass(self):
        """Randomly generate and run different configurations for Felix Tagger."""
        # Setup.
        test_cases = [
            {
                "use_pointing": False,
                "query_transformer": False,
                "is_training": True,
            },
            {
                "use_pointing": True,
                "query_transformer": False,
                "is_training": True,
            },
            {
                "use_pointing": True,
                "query_transformer": True,
                "is_training": True,
            },
            {
                "use_pointing": False,
                "query_transformer": False,
                "is_training": False,
            },
            {
                "use_pointing": True,
                "query_transformer": False,
                "is_training": False,
            },
            {
                "use_pointing": True,
                "query_transformer": True,
                "is_training": False,
            },
        ]

        for case in test_cases:
            with self.subTest(case=case):
                use_pointing = case["use_pointing"]
                query_transformer = case["query_transformer"]
                is_training = case["is_training"]

                sequence_length = 7
                vocab_size = 11
                bert_hidden_size = 13
                bert_num_hidden_layers = 1
                bert_num_attention_heads = 1
                bert_intermediate_size = 4
                bert_type_vocab_size = 2
                bert_max_position_embeddings = sequence_length
                bert_config = BertConfig(
                    vocab_size=vocab_size,
                    hidden_size=bert_hidden_size,
                    num_hidden_layers=bert_num_hidden_layers,
                    num_attention_heads=bert_num_attention_heads,
                    intermediate_size=bert_intermediate_size,
                    type_vocab_size=bert_type_vocab_size,
                    max_position_embeddings=bert_max_position_embeddings,
                )

                batch_size = 17
                edit_tags_size = 19
                bert_config.num_classes = edit_tags_size
                bert_config.query_size = 23
                bert_config.query_transformer = query_transformer

                tagger = my_tagger.MyTagger(
                    config=bert_config,
                    seq_length=sequence_length,
                    use_pointing=use_pointing,
                )

                # Create inputs.
                # Set random seed.
                torch.manual_seed(42)

                # Create inputs.
                input_ids = torch.randint(
                    low=0,
                    high=vocab_size - 1,
                    size=(batch_size, sequence_length),
                )
                attention_mask = torch.randint(
                    low=1, high=2, size=(batch_size, sequence_length)
                )
                token_type_ids = torch.ones(
                    (batch_size, sequence_length)
                ).long()
                edit_tags = torch.randint(
                    low=0,
                    high=edit_tags_size - 2,
                    size=(batch_size, sequence_length),
                )

                # Run the model.
                if is_training:
                    tagger.train()
                    output = tagger(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        edit_tags=edit_tags,
                    )
                else:
                    tagger.eval()
                    output = tagger(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                    )

                # Check output shapes.
                if use_pointing:
                    tag_logits, pointing_logits = output
                    self.assertEqual(
                        pointing_logits.shape,
                        (batch_size, sequence_length, sequence_length),
                    )
                else:
                    tag_logits = output[0]
                self.assertEqual(
                    tag_logits.shape,
                    (batch_size, sequence_length, edit_tags_size),
                )

    def test_tag_loss(self):
        felix_tag_loss = my_tagger.TagLoss(
            use_pointing=True, pointing_weight=0.5
        )

        # Generate some dummy data for testing
        batch_size = 32
        seq_length = 128
        vocab_size = 13

        tag_logits = torch.rand(batch_size, seq_length, vocab_size)
        tag_labels = torch.randint(
            0, vocab_size, size=(batch_size, seq_length)
        )
        input_mask = torch.randint(0, 2, size=(batch_size, seq_length))
        labels_mask = torch.rand(batch_size, seq_length)
        point_logits = torch.rand(batch_size, seq_length, seq_length)
        point_labels = torch.randint(
            0, seq_length, size=(batch_size, seq_length)
        )

        # Compute the loss and metrics using the FelixTagLoss layer
        total_loss = felix_tag_loss(
            tag_logits,
            tag_labels,
            input_mask,
            labels_mask,
            point_logits,
            point_labels,
        )

        assert isinstance(total_loss, torch.Tensor)
        assert isinstance(total_loss.item(), float)


if __name__ == "__main__":
    unittest.main()
