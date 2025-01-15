import collections
import tempfile
import unittest

import bert_example
import insertion_converter
import pointing_converter


class BertExampleTest(unittest.TestCase):
    def setUp(self):
        super(BertExampleTest, self).setUp()

        vocab_tokens = [
            "[CLS]",
            "[SEP]",
            "[PAD]",
            "a",
            "b",
            "c",
            "##d",
            "##e",
            "[MASK]",
            "[unused1]",
            "[unused2]",
        ]
        with tempfile.NamedTemporaryFile(delete=False) as vocab_file:
            vocab_file.write(
                "".join([x + "\n" for x in vocab_tokens]).encode()
            )

        label_map = {"KEEP": 1, "DELETE": 2, "KEEP|1": 3, "KEEP|2:": 4}
        max_seq_length = 8
        do_lower_case = False
        converter = pointing_converter.PointingConverter([])
        self._builder = bert_example.BertExampleBuilder(
            label_map=label_map,
            vocab_file=vocab_file.name,
            max_seq_length=max_seq_length,
            do_lower_case=do_lower_case,
            converter=converter,
            use_open_vocab=False,
        )
        converter_insertion = insertion_converter.InsertionConverter(
            max_seq_length=max_seq_length,
            vocab_file=vocab_file.name,
            label_map=label_map,
        )
        self._builder_mask = bert_example.BertExampleBuilder(
            label_map=label_map,
            vocab_file=vocab_file.name,
            max_seq_length=max_seq_length,
            do_lower_case=do_lower_case,
            converter=converter,
            use_open_vocab=True,
            converter_insertion=converter_insertion,
        )

        self.pad_token_id = self._builder.tokenizer.pad_token_id
        self.label_pad_token_id = -100

    def _check_label_weights(self, labels_mask, labels, input_mask):
        self.assertAlmostEqual(sum(labels_mask), sum(input_mask))
        label_weights = collections.defaultdict(float)
        # Labels should have the same weight.
        for label, label_mask in zip(labels, labels_mask):
            # Ignore pad labels.
            if label == 0:
                continue
            label_weights[label] += label_mask
        label_weights_values = list(label_weights.values())
        for i in range(1, len(label_weights_values)):
            self.assertAlmostEqual(
                label_weights_values[i], label_weights_values[i - 1]
            )

    def test_building_with_target(self):
        sources = ["a b ade"]  # Tokenized: [CLS] a b a ##d ##e [SEP]
        target = "a ade"  # Tokenized: [CLS] a a ##d ##e [SEP]
        example, _ = self._builder.build_bert_example(sources, target)
        # input_ids should contain the IDs for the following tokens:
        #   [CLS] a b a ##d ##e [SEP] [PAD]
        self.assertEqual(example.input_ids, [0, 3, 4, 3, 6, 7, 1])
        self.assertEqual(example.input_mask, [1, 1, 1, 1, 1, 1, 1])
        self.assertEqual(example.token_type_ids, [0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(example.labels, [1, 1, 2, 1, 1, 1, 1])
        self.assertEqual(example.point_indexes, [1, 3, 0, 4, 5, 6, 0])
        self.assertEqual(
            [1 if x > 0 else 0 for x in example.labels_mask],
            [1, 1, 1, 1, 1, 1, 1],
        )
        self._check_label_weights(
            example.labels_mask, example.labels, example.input_mask
        )

    def test_building_no_target_truncated(self):
        sources = ["ade bed cde"]
        example, _ = self._builder.build_bert_example(sources)
        # input_ids should contain the IDs for the following tokens:
        #   [CLS] a ##d ##e b ##e ##d [SEP]
        # where the last token 'cde' has been truncated.
        self.assertEqual(example.input_ids, [0, 3, 6, 7, 4, 7, 6, 1])
        self.assertEqual(example.input_mask, [1, 1, 1, 1, 1, 1, 1, 1])
        self.assertEqual(example.token_type_ids, [0, 0, 0, 0, 0, 0, 0, 0])

    def test_building_with_target_mask(self):
        sources = ["a b ade"]  # Tokenized: [CLS] a b a ##d ##e [SEP]
        target = "a ade"  # Tokenized: [CLS] a a ##d ##e [SEP]

        example, _ = self._builder_mask.build_bert_example(sources, target)
        # input_ids should contain the IDs for the following tokens:
        #   [CLS] a b a ##d ##e [SEP]
        self.assertEqual(example.input_ids, [0, 3, 4, 3, 6, 7, 1])
        self.assertEqual(example.input_mask, [1, 1, 1, 1, 1, 1, 1])
        self.assertEqual(example.token_type_ids, [0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(example.labels, [1, 1, 2, 1, 1, 1, 1])
        self.assertEqual(example.point_indexes, [1, 3, 0, 4, 5, 6, 0])
        self.assertEqual(
            [1 if x > 0 else 0 for x in example.labels_mask],
            [1, 1, 1, 1, 1, 1, 1],
        )
        self._check_label_weights(
            example.labels_mask, example.labels, example.input_mask
        )

    def test_building_with_insertion(self):
        sources = ["a b"]  # Tokenized: [CLS] a b [SEP]
        target = "a b c"  # Tokenized: [CLS] a b c [SEP]
        example, insertion_example = self._builder_mask.build_bert_example(
            sources, target
        )
        # input_ids should contain the IDs for the following tokens:
        #   [CLS] a b [SEP]
        self.assertEqual(example.input_ids, [0, 3, 4, 1])
        self.assertEqual(example.input_mask, [1, 1, 1, 1])
        self.assertEqual(example.token_type_ids, [0, 0, 0, 0])
        self.assertEqual(example.labels, [1, 1, 3, 1])
        self.assertEqual(example.point_indexes, [1, 2, 3, 0])
        self.assertEqual(
            [1 if x > 0 else 0 for x in example.labels_mask],
            [1, 1, 1, 1],
        )
        self._check_label_weights(
            example.labels_mask, example.labels, example.input_mask
        )
        self.assertEqual(insertion_example["input_ids"], [0, 3, 4, 8, 1])
        self.assertEqual(insertion_example["input_mask"], [1, 1, 1, 1, 1])
        self.assertEqual(
            insertion_example["masked_lm_ids"],
            [self.label_pad_token_id] * 3 + [5] + [self.label_pad_token_id],
        )

    def test_building_with_custom_source_separator(self):
        vocab_tokens = [
            "[CLS]",
            "[SEP]",
            "[PAD]",
            "a",
            "b",
            "c",
            "##d",
            "##e",
            "[MASK]",
            "[unused1]",
            "[unused2]",
        ]
        vocab_file = tempfile.NamedTemporaryFile(delete=False).name
        with open(vocab_file, "w") as vocab_writer:
            vocab_writer.write("".join([x + "\n" for x in vocab_tokens]))

        builder = bert_example.BertExampleBuilder(
            vocab_file=vocab_file,
            label_map={"KEEP": 1, "DELETE": 2, "KEEP|1": 3, "KEEP|2:": 4},
            max_seq_length=9,
            do_lower_case=False,
            converter=pointing_converter.PointingConverter([]),
            use_open_vocab=False,
            special_glue_string_for_sources=" [SEP] ",
        )

        sources = ["a b", "ade"]  # Tokenized: [CLS] a b [SEP] a ##d ##e [SEP]
        target = "a ade"  # Tokenized: [CLS] a a ##d ##e [SEP]
        example, _ = builder.build_bert_example(sources, target)
        # input_ids should contain the IDs for the following tokens:
        #   [CLS] a b [SEP] a ##d ##e [SEP]
        self.assertEqual(example.input_ids, [0, 3, 4, 1, 3, 6, 7, 1])
        self.assertEqual(example.input_mask, [1, 1, 1, 1, 1, 1, 1, 1])
        self.assertEqual(example.token_type_ids, [0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(example.labels, [1, 1, 2, 2, 1, 1, 1, 1])
        self.assertEqual(example.point_indexes, [1, 4, 0, 0, 5, 6, 7, 0])
        self._check_label_weights(
            example.labels_mask, example.labels, example.input_mask
        )
        self.assertEqual(
            [1 if x > 0 else 0 for x in example.labels_mask],
            [1, 1, 1, 1, 1, 1, 1, 1],
        )


if __name__ == "__main__":
    unittest.main()
