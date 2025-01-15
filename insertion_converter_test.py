import tempfile
import unittest

import insertion_converter


class IntegrationInsertionConverterTest(unittest.TestCase):
    def setUp(self):
        super().setUp()

        vocab_tokens = [
            "[CLS]",
            "[SEP]",
            "[PAD]",
            "[unused1]",
            "[MASK]",
            "[unused2]",
            "a",
            "b",
            "c",
            "d",
            "e",
        ]
        self.label_map = {
            "KEEP": 1,
            "DELETE": 0,
            "MASK|1": 2,
            "MASK|2": 3,
        }
        with tempfile.NamedTemporaryFile(delete=False) as vocab_file:
            vocab_file.write(
                "".join([x + "\n" for x in vocab_tokens]).encode()
            )
        self.vocab_file = vocab_file

        self.converter = insertion_converter.InsertionConverter(
            max_seq_length=20,
            vocab_file=self.vocab_file.name,
            label_map=self.label_map,
            fall_back_mode="force",
        )

        self.pad_token = self.converter._tokenizer.pad_token

    def test_create_insertion_example(self):
        test_cases = [
            # A simple test.
            {
                "input_texts": "a b c [SEP]".split(),
                "target": "a b c [SEP]",
                "target_points": [1, 2, 3, 0],
                "target_masked": "a b c [SEP]",
                "labels": [1, 1, 1, 1],
                "gold_unused_tokens": set([]),
                "with_delete_source": "a b c [SEP]".split(),
                "with_delete_target": "a b c [SEP]".split(),
                "feed_dict": {
                    "token_type_ids": [0, 0, 0, 0],
                    "masked_lm_ids": [self.pad_token] * 4,
                },
            },
            # A multiple SEPs (think post editing) test.
            {
                "input_texts": "a b [SEP] c [SEP]".split(),
                "target": "a b [SEP] c [SEP]",
                "target_points": [1, 2, 3, 4, 0],
                "target_masked": "a b [SEP] c [SEP]",
                "labels": [1, 1, 1, 1, 1],
                "gold_unused_tokens": set([]),
                "with_delete_source": "a b [SEP] c [SEP]".split(),
                "with_delete_target": "a b [SEP] c [SEP]".split(),
                "feed_dict": {
                    "token_type_ids": [0, 0, 0, 0, 0],
                    "masked_lm_ids": [self.pad_token] * 5,
                },
            },
            # A multiple SEPs (sep order changed) test.
            {
                "input_texts": "a b [SEP] c [SEP]".split(),
                "target": "a b [SEP] c [SEP]",
                "target_points": [1, 4, 0, 2, 3],
                "target_masked": "a b [SEP] c [SEP]",
                "labels": [1, 1, 1, 1, 1],
                "gold_unused_tokens": set([]),
                "with_delete_source": "a b [SEP] c [SEP]".split(),
                "with_delete_target": "a b [SEP] c [SEP]".split(),
                "feed_dict": {
                    "token_type_ids": [0, 0, 0, 0, 0],
                    "masked_lm_ids": [self.pad_token] * 5,
                },
            },
            # A multiple input SEPs with first SEP deleted test.
            {
                "input_texts": "a b [SEP] c [SEP]".split(),
                "target": "a b c [SEP]",
                "target_points": [1, 3, 0, 4, 0],
                "target_masked": "a b c [SEP]",
                "labels": [1, 1, 0, 1, 1],
                "gold_unused_tokens": set([]),
                "with_delete_source": "a b [unused1] [SEP] [unused2] c [SEP]".split(),
                "with_delete_target": "a b [unused1] [SEP] [unused2] c [SEP]".split(),
                "feed_dict": {
                    "token_type_ids": [0, 0, 1, 1, 1, 0, 0],
                    "masked_lm_ids": [self.pad_token] * 7,
                },
            },
            # A multiple input SEPs with second SEP deleted test.
            {
                "input_texts": "a b [SEP] c [SEP]".split(),
                "target": "a b c [SEP]",
                "target_points": [1, 3, 0, 2, 0],
                "target_masked": "a b c [SEP]",
                "labels": [1, 1, 1, 1, 0],
                "gold_unused_tokens": set([]),
                "with_delete_source": "a b c [unused1] [SEP] [unused2] [SEP]".split(),
                "with_delete_target": "a b c [unused1] [SEP] [unused2] [SEP]".split(),
                "feed_dict": {
                    "token_type_ids": [0, 0, 0, 1, 1, 1, 0],
                    "masked_lm_ids": [self.pad_token] * 7,
                },
            },
            # A multiple SEPs (sep order changed) with delete test.
            {
                "input_texts": "a b [SEP] c [SEP]".split(),
                "target": "a  [SEP] c [SEP]",
                "target_points": [4, 0, 0, 2, 3],
                "target_masked": "a [SEP] c [SEP]",
                "labels": [1, 0, 1, 1, 1],
                "gold_unused_tokens": set([]),
                "with_delete_source": "a [unused1] b [unused2] [SEP] c [SEP]".split(),
                "with_delete_target": "a [unused1] b [unused2] [SEP] c [SEP]".split(),
                "feed_dict": {
                    "token_type_ids": [0, 1, 1, 1, 0, 0, 0],
                    "masked_lm_ids": [self.pad_token] * 7,
                },
            },
            # A multiple SEPs (sep order changed) with delete and insert test.
            {
                "input_texts": "a b [SEP] c [SEP]".split(),
                "target": "a  [SEP] c d [SEP]",
                "target_points": [4, 0, 0, 2, 3],
                "target_masked": "a [SEP] c [MASK] [SEP]",
                "labels": [1, 0, 1, 2, 1],
                "gold_unused_tokens": set([]),
                "with_delete_source": "a [unused1] b [unused2] [SEP] c [MASK] [SEP]".split(),
                "with_delete_target": "a [unused1] b [unused2] [SEP] c d [SEP]".split(),
                "feed_dict": {
                    "token_type_ids": [0, 1, 1, 1, 0, 0, 0, 0],
                    "masked_lm_ids": [self.pad_token] * 6
                    + ["d"]
                    + [self.pad_token],
                },
            },
            # Missing a middle token.
            {
                "input_texts": "a b [SEP]".split(),
                "target": "a b c [SEP]",
                "target_points": [1, 2, 0],
                "target_masked": "a b [MASK] [SEP]",
                "gold_unused_tokens": set([]),
                "with_delete_source": "a b [MASK] [SEP]".split(),
                "with_delete_target": "a b c [SEP]".split(),
                "labels": [1, 2, 1],
                "feed_dict": {
                    "token_type_ids": [0, 0, 0, 0],
                    "masked_lm_ids": [
                        self.pad_token,
                        self.pad_token,
                        "c",
                        self.pad_token,
                    ],
                },
            },
            # Missing a start token.
            {
                "input_texts": "a c [SEP]".split(),
                "target": "a b c [SEP]",
                "target_points": [1, 2, 0],
                "target_masked": "a [MASK] c [SEP]",
                "gold_unused_tokens": set([]),
                "with_delete_source": "a [MASK] c [SEP]".split(),
                "with_delete_target": "a b c [SEP] ".split(),
                "labels": [2, 1, 1],
                "feed_dict": {
                    "token_type_ids": [0, 0, 0, 0],
                    "masked_lm_ids": [
                        self.pad_token,
                        "b",
                        self.pad_token,
                        self.pad_token,
                    ],
                },
            },
            # Missing multiple tokens.
            {
                "input_texts": "a c [SEP]".split(),
                "target": "a b e c [SEP]",
                "target_points": [1, 2, 0],
                "target_masked": "a [MASK] [MASK] c [SEP]",
                "gold_unused_tokens": set([]),
                "with_delete_source": "a [MASK] [MASK] c [SEP]".split(),
                "with_delete_target": "a b e c [SEP] ".split(),
                "labels": [3, 1, 1],
                "feed_dict": {
                    "token_type_ids": [0, 0, 0, 0, 0],
                    "masked_lm_ids": [
                        self.pad_token,
                        "b",
                        "e",
                        self.pad_token,
                        self.pad_token,
                    ],
                },
            },
            # An additional source token.
            {
                "input_texts": "a b e [SEP]".split(),
                "target": "a b [SEP]",
                "target_points": [1, 3, 0, 0],
                "target_masked": "a b [SEP]",
                "gold_unused_tokens": set([("e", "[SEP]")]),
                "labels": [1, 1, 0, 1],
                "with_delete_source": "a b [unused1] e [unused2] [SEP]".split(),
                "with_delete_target": "a b [unused1] e [unused2] [SEP]".split(),
                "feed_dict": {
                    "token_type_ids": [0, 0, 1, 1, 1, 0],
                    "masked_lm_ids": [self.pad_token] * 6,
                },
            },
            # Missing a middle token + an additional source token.
            {
                "input_texts": "a b e [SEP]".split(),
                "target": "a b c [SEP]",
                "target_points": [1, 3, 0, 0],
                "target_masked": "a b [MASK] [SEP]",
                "gold_unused_tokens": set([("e", "[SEP]")]),
                "with_delete_source": "a b [MASK] [unused1] e [unused2] [SEP]".split(),
                "with_delete_target": "a b c [unused1] e [unused2] [SEP]".split(),
                "labels": [1, 2, 0, 1],
                "feed_dict": {
                    "token_type_ids": [0, 0, 0, 1, 1, 1, 0],
                    "masked_lm_ids": [
                        self.pad_token,
                        self.pad_token,
                        "c",
                        self.pad_token,
                        self.pad_token,
                        self.pad_token,
                        self.pad_token,
                    ],
                },
            },
            # duplicate target and source token.
            {
                "input_texts": "a d b e [SEP]".split(),
                "target": "a b d [SEP]",
                "target_points": [2, 4, 1, 0, 0],
                "target_masked": "a b d [SEP]",
                "gold_unused_tokens": set([("e", "[SEP]")]),
                "with_delete_source": "a b [unused1] e [unused2] d [SEP]".split(),
                "with_delete_target": "a b [unused1] e [unused2] d [SEP]".split(),
                "labels": [1, 1, 1, 0, 1],
                "feed_dict": {
                    "token_type_ids": [0, 0, 1, 1, 1, 0, 0],
                    "masked_lm_ids": [self.pad_token] * 7,
                },
            },
            # Multiple sequential deleted tokens.
            {
                "input_texts": "a d b e [SEP]".split(),
                "target": "a [SEP]",
                "target_points": [4, 0, 0, 0, 0],
                "target_masked": "a [SEP]",
                "gold_unused_tokens": set([("d b e", "[SEP]")]),
                "with_delete_source": "a [unused1] d b e [unused2] [SEP]".split(),
                "with_delete_target": "a [unused1] d b e [unused2] [SEP]".split(),
                "labels": [1, 0, 0, 0, 1],
                "feed_dict": {
                    "token_type_ids": [0, 1, 1, 1, 1, 1, 0],
                    "masked_lm_ids": [self.pad_token] * 7,
                },
            },
            # Multiple non sequential deleted tokens.
            {
                "input_texts": "a b c d e [SEP]".split(),
                "target": "a d b [SEP]",
                "target_points": [3, 5, 0, 1, 0, 0],
                "target_masked": "a d b [SEP]",
                "gold_unused_tokens": set([("c", "d"), ("e", "[SEP]")]),
                "with_delete_source": "a d [unused1] e [unused2] b [unused1] c [unused2] [SEP]".split(),
                "with_delete_target": "a d [unused1] e [unused2] b [unused1] c [unused2] [SEP]".split(),
                "labels": [1, 1, 0, 1, 0, 1],
                "feed_dict": {
                    "token_type_ids": [0, 0, 1, 1, 1, 0, 1, 1, 1, 0],
                    "masked_lm_ids": [self.pad_token] * 10,
                },
            },
        ]

        for case in test_cases:
            with self.subTest(case=case):
                input_texts = case["input_texts"]
                target = case["target"]
                target_points = case["target_points"]
                labels = case["labels"]
                feed_dict = case["feed_dict"]

                output_feed_dict = self.converter.create_insertion_example(
                    input_texts, labels, target_points, target.split()
                )

                # Perform assertions for output_feed_dict and feed_dict here.
                no_pad_input_ids = []
                seen_padding = False

                for input_id in output_feed_dict["input_ids"]:
                    if input_id != self.converter._tokenizer.pad_token_id:
                        no_pad_input_ids.append(input_id)
                        self.assertFalse(seen_padding)
                    else:
                        seen_padding = True

                self.assertEqual(
                    self.converter._tokenizer.convert_ids_to_tokens(
                        no_pad_input_ids
                    ),
                    case["with_delete_source"],
                )

                self.assertEqual(
                    feed_dict["token_type_ids"],
                    output_feed_dict["token_type_ids"][
                        : len(feed_dict["token_type_ids"])
                    ],
                )
                # Replace -100 with '[PAD]'
                modified_list = [
                    self.converter._tokenizer.pad_token_id if x == -100 else x
                    for x in output_feed_dict["masked_lm_ids"]
                ]

                self.assertEqual(
                    feed_dict["masked_lm_ids"],
                    self.converter._tokenizer.convert_ids_to_tokens(
                        modified_list
                    ),
                )


if __name__ == "__main__":
    unittest.main()
