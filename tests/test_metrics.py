"""Dice tests use only tiny synthetic masks."""
import unittest

import numpy as np

from nucleo_viewer.errors import AppError
from nucleo_viewer.imaging import Mask
from nucleo_viewer.metrics import dice_score


def mask(values, affine=None):
    return Mask(np.asarray(values, dtype=np.uint8).reshape(2, 2, 1),
                np.eye(4) if affine is None else affine)


class DiceTests(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(dice_score(mask([0, 1, 1, 0]), mask([0, 1, 1, 0])), 1.0)

    def test_disjoint(self):
        self.assertEqual(dice_score(mask([1, 1, 0, 0]), mask([0, 0, 1, 1])), 0.0)

    def test_partial_overlap(self):
        score = dice_score(mask([1, 1, 0, 0]), mask([0, 1, 1, 1]))
        self.assertIsInstance(score, float)
        self.assertAlmostEqual(score, 2 / 5)

    def test_both_empty_is_one(self):
        self.assertEqual(dice_score(mask([0, 0, 0, 0]), mask([0, 0, 0, 0])), 1.0)

    def test_one_empty_is_zero(self):
        self.assertEqual(dice_score(mask([0, 0, 0, 0]), mask([0, 0, 1, 0])), 0.0)

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(AppError):
            dice_score(mask([0, 0, 0, 0]), Mask(np.zeros((2, 1, 1), dtype=np.uint8), np.eye(4)))

    def test_affine_mismatch_rejected(self):
        affine = np.eye(4)
        affine[0, 3] = 0.01
        with self.assertRaises(AppError):
            dice_score(mask([1, 0, 0, 0]), mask([1, 0, 0, 0], affine))

    def test_absolute_tolerance_not_relative(self):
        affine = np.eye(4)
        affine[0, 3] = 1000
        changed = affine.copy()
        changed[0, 3] += 0.001
        with self.assertRaises(AppError):
            dice_score(mask([0, 0, 0, 0], affine), mask([0, 0, 0, 0], changed))
        changed[0, 3] = affine[0, 3] + 0.00005
        self.assertEqual(dice_score(mask([0, 0, 0, 0], affine), mask([0, 0, 0, 0], changed)), 1.0)

    def test_invalid_affine_and_nonbinary_values(self):
        for affine in (np.zeros((4, 4)), np.full((4, 4), np.nan)):
            with self.subTest(affine=affine), self.assertRaises(AppError):
                dice_score(mask([0, 0, 0, 0], affine), mask([0, 0, 0, 0]))
        with self.assertRaises(AppError):
            dice_score(mask([2, 0, 0, 0]), mask([1, 0, 0, 0]))


if __name__ == "__main__":
    unittest.main()
