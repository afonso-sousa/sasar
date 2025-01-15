import json
import tempfile
import unittest

from transformers import BertTokenizer

import utils


class UtilsTest(unittest.TestCase):
    def setUp(self):
        self._vocab_tokens = [
            "NOTHING",
            "[CLS]",
            "[SEP]",
            "[MASK]",
            "[PAD]",
            "a",
            "b",
            "c",
            "##d",
            "d",
            "##e",
        ]
        with tempfile.NamedTemporaryFile(delete=False) as vocab_file:
            vocab_file.write("\n".join(self._vocab_tokens).encode())
        self._vocab_file = vocab_file.name
        self._tokenizer = BertTokenizer.from_pretrained(
            self._vocab_file, do_lower_case=True
        )

    def test_build_feed_dict(self):
        test_cases = [
            {
                "source": "a [MASK] b".split(),
                "target": "a c b".split(),
                "masks": ["", "c", ""],
            },
            {
                "source": "a [MASK] [MASK] b".split(),
                "target": "a c d b".split(),
                "masks": ["", "c", "d", ""],
            },
            {
                "source": "[MASK] b [MASK] d".split(),
                "target": "a b c d".split(),
                "masks": ["a", "", "c", ""],
            },
        ]

        for test_case in test_cases:
            source = test_case["source"]
            target = test_case["target"]
            masks = test_case["masks"]
            feed_dict = utils.build_feed_dict(source, self._tokenizer, target)

            for i, mask_id in enumerate(feed_dict["masked_lm_ids"]):
                # Ignore padding.
                if mask_id == -100:
                    continue

                self.assertEqual(
                    mask_id, self._tokenizer.convert_tokens_to_ids(masks[i])
                )

    def test_read_label_map_with_tuple_keys(self):
        orig_label_map = {"KEEP": 0, "DELETE|2": 1, "DELETE|1": 2}
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(json.dumps(orig_label_map).encode())
            temp_file.seek(0)
            label_map = utils.read_label_map(
                temp_file.name, use_str_keys=False
            )
            self.assertEqual(
                label_map,
                {
                    ("KEEP", 0): 0,
                    ("DELETE", 2): 1,
                    ("DELETE", 1): 2,
                },
            )
