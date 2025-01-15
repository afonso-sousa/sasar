import unittest

import constants
import converter_for_insert as converter


class InsertConverterTest(unittest.TestCase):
    """Tests when `insert_after_token == True`."""

    def test_compute_edits_and_insertions(self):
        source = ["A", "B", "c", "D"]  # noqa
        target = ["A", "Z", "B", "D", "W"]  # noqa
        #          K    I  | K |   D |   K    I
        edits, insertions = converter.compute_edits_and_insertions(
            source, target, max_insertions_per_token=1
        )
        self.assertEqual(
            edits,
            [constants.KEEP, constants.KEEP, constants.DELETE, constants.KEEP],
        )
        self.assertEqual(insertions, [["Z"], [], [], ["W"]])

    def test_compute_edits_and_insertions_for_replacement(self):
        source = ["A", "b", "C"]
        target = ["A", "B", "C"]
        edits, insertions = converter.compute_edits_and_insertions(
            source, target, max_insertions_per_token=1
        )
        self.assertEqual(
            edits, [constants.KEEP, constants.DELETE, constants.KEEP]
        )
        self.assertEqual(insertions, [[], ["B"], []])

    def test_compute_edits_and_insertions_for_long_insertion(self):
        source = ["A", "B"]
        target = ["A", "X", "Y", "B"]
        edits_and_insertions = converter.compute_edits_and_insertions(
            source, target, max_insertions_per_token=1
        )
        self.assertIsNone(edits_and_insertions)
        edits, insertions = converter.compute_edits_and_insertions(
            source, target, max_insertions_per_token=2
        )
        self.assertEqual(edits, [constants.KEEP, constants.KEEP])
        self.assertEqual(insertions, [["X", "Y"], []])

    def test_compute_edits_and_insertions_for_long_insertion_and_deletions(
        self,
    ):
        source = ["A", "b", "c", "D"]
        target = ["A", "X", "Y", "Z", "U", "V", "D"]
        edits_and_insertions = converter.compute_edits_and_insertions(
            source, target, max_insertions_per_token=2
        )
        self.assertIsNone(edits_and_insertions)
        edits, insertions = converter.compute_edits_and_insertions(
            source, target, max_insertions_per_token=3
        )
        self.assertEqual(
            edits,
            [
                constants.KEEP,
                constants.DELETE,
                constants.DELETE,
                constants.KEEP,
            ],
        )
        self.assertEqual(insertions, [[], ["X", "Y", "Z"], ["U", "V"], []])

    def test_compute_edits_and_insertions_no_overlap(self):
        source = ["a", "b"]
        target = ["C", "D"]
        edits, insertions = converter.compute_edits_and_insertions(
            source, target, max_insertions_per_token=2
        )
        self.assertEqual(edits, [constants.DELETE, constants.DELETE])
        self.assertEqual(insertions, [["C", "D"], []])


class InsertConverterTestInsertBefore(unittest.TestCase):
    """Tests when `insert_after_token == False`."""

    def test_compute_edits_and_insertions(self):
        source = ["A", "B", "c", "D"]
        target = ["X", "A", "Z", "B", "D"]
        #          K    I  | K |   D |   K    I
        edits, insertions = converter.compute_edits_and_insertions(
            source,
            target,
            max_insertions_per_token=1,
            insert_after_token=False,
        )
        self.assertEqual(
            edits,
            [constants.KEEP, constants.KEEP, constants.DELETE, constants.KEEP],
        )
        self.assertEqual(insertions, [["X"], ["Z"], [], []])

    def test_compute_edits_and_insertions_for_replacement(self):
        source = ["A", "b", "C", "D"]
        target = ["A", "B", "C", "D"]

        # We should insert 'B' before 'b' (not before 'C' although the result is the
        # same).
        edits, insertions = converter.compute_edits_and_insertions(
            source,
            target,
            max_insertions_per_token=1,
            insert_after_token=False,
        )
        self.assertEqual(
            edits,
            [constants.KEEP, constants.DELETE, constants.KEEP, constants.KEEP],
        )
        self.assertEqual(insertions, [[], ["B"], [], []])

    def test_compute_edits_and_insertions_for_long_insertion(self):
        source = ["A", "B"]
        target = ["A", "X", "Y", "B"]
        edits_and_insertions = converter.compute_edits_and_insertions(
            source,
            target,
            max_insertions_per_token=1,
            insert_after_token=False,
        )
        self.assertIsNone(edits_and_insertions)
        edits, insertions = converter.compute_edits_and_insertions(
            source,
            target,
            max_insertions_per_token=2,
            insert_after_token=False,
        )
        self.assertEqual(edits, [constants.KEEP, constants.KEEP])
        self.assertEqual(insertions, [[], ["X", "Y"]])

    def test_compute_edits_and_insertions_for_long_insertion_and_deletions(
        self,
    ):
        source = ["a", "b", "C"]
        target = ["X", "Y", "Z", "U", "V", "C"]
        edits_and_insertions = converter.compute_edits_and_insertions(
            source,
            target,
            max_insertions_per_token=2,
            insert_after_token=False,
        )
        self.assertIsNone(edits_and_insertions)
        edits, insertions = converter.compute_edits_and_insertions(
            source,
            target,
            max_insertions_per_token=3,
            insert_after_token=False,
        )
        self.assertEqual(
            edits, [constants.DELETE, constants.DELETE, constants.KEEP]
        )
        self.assertEqual(insertions, [["X", "Y"], ["Z", "U", "V"], []])

    def test_compute_edits_and_insertions_no_overlap(self):
        source = ["a", "b"]
        target = ["C", "D"]
        edits, insertions = converter.compute_edits_and_insertions(
            source, target, max_insertions_per_token=2
        )
        self.assertEqual(edits, [constants.DELETE, constants.DELETE])
        self.assertEqual(insertions, [["C", "D"], []])


if __name__ == "__main__":
    unittest.main()
