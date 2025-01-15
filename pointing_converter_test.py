import random
import string
import unittest

import pointing_converter


class PointingConverterTest(unittest.TestCase):
    def test_matching_conversion(self):
        test_cases = [
            # A simple test.
            {
                "input_texts": "A B C D".split(),
                "target": "A B C D",
                "phrase_vocabulary": ["and"],
                "target_points": [1, 2, 3, 0],
                "target_phrase": ["", "", "", ""],
            },
            # Missing a middle token.
            {
                "input_texts": "A B D".split(),
                "target": "A b C D",
                "phrase_vocabulary": ["c"],
                "target_points": [1, 2, 0],
                "target_phrase": ["", "c", ""],
            },
            # An additional source token.
            {
                "input_texts": "A B E D".split(),
                "target": "A B D",
                "phrase_vocabulary": ["and"],
                "target_points": [1, 3, 0, 0],
                "target_phrase": ["", "", "", ""],
            },
            # Missing a middle token + an additional source token.
            {
                "input_texts": "A B E D".split(),
                "target": "A B C D",
                "phrase_vocabulary": ["c"],
                "target_points": [1, 3, 0, 0],
                "target_phrase": ["", "c", "", ""],
            },
            # Duplicate target token.
            {
                "input_texts": "A B E D".split(),
                "target": "A B C D D",
                "phrase_vocabulary": ["c"],
                "target_points": [],
                "target_phrase": [],
            },
            # Duplicate target and source token.
            {
                "input_texts": "A D B E D".split(),
                "target": "A B D D",
                "phrase_vocabulary": ["c"],
                "target_points": [2, 4, 1, 0, 0],
                "target_phrase": ["", "", "", "", ""],
            },
        ]

        for test_case in test_cases:
            with self.subTest(test_case=test_case):
                input_texts = test_case["input_texts"]
                target = test_case["target"]
                phrase_vocabulary = test_case["phrase_vocabulary"]
                target_points = test_case["target_points"]
                target_phrase = test_case["target_phrase"]

                converter = pointing_converter.PointingConverter(phrase_vocabulary)
                points = converter.compute_points(input_texts, target)

                if not points:
                    self.assertEqual(points, target_phrase)
                    self.assertEqual(points, target_points)
                else:
                    self.assertEqual([x.added_phrase for x in points], target_phrase)
                    self.assertEqual([x.point_index for x in points], target_points)

    def test_no_match(self):
        input_texts = "Turing was born in 1912 . Turing died in 1954 .".split()
        target = "Turing was born in 1912 and died in 1954 ."
        phrase_vocabulary = ["but"]

        converter = pointing_converter.PointingConverter(phrase_vocabulary)
        points = converter.compute_points(input_texts, target)

        # Vocabulary doesn't contain "and" so the inputs can't be converted to the target.
        self.assertEqual(points, [])

    def test_match(self):
        input_texts = "Turing was born in 1912 . Turing died in 1954 .".split()
        target = "Turing was born in 1912 and died in 1954 ."
        phrase_vocabulary = ["but", "KEEP|and"]

        converter = pointing_converter.PointingConverter(phrase_vocabulary)
        points = converter.compute_points(input_texts, target)

        target_points = [1, 2, 3, 4, 7, 0, 0, 8, 9, 10, 0]
        target_phrases = ["", "", "", "", "and", "", "", "", "", "", ""]
        self.assertEqual([x.point_index for x in points], target_points)
        self.assertEqual([x.added_phrase for x in points], target_phrases)

    def test_match_all(self):
        random.seed(1337)
        phrase_vocabulary = set()
        converter = pointing_converter.PointingConverter(phrase_vocabulary)

        for _ in range(10):
            input_texts = [
                random.choice(string.ascii_uppercase + string.digits) for _ in range(10)
            ]
            input_texts.append("eos")  # One token needs to match.
            target = " ".join(
                [
                    random.choice(string.ascii_uppercase + string.digits)
                    for _ in range(11)
                ]
            )
            target += " eos"
            points = converter.compute_points(input_texts, target)
            self.assertTrue(points)


if __name__ == "__main__":
    unittest.main()
