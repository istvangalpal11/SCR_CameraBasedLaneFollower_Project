"""Focused, camera-free checks for Phase 1.1 global row-scan observations."""
import unittest
import numpy as np
from src.algorithms.LaneDetection.candidate_extractor import extract_row_markings


class RowScanTests(unittest.TestCase):
    def test_two_runs_and_unchanged_input(self):
        image = np.zeros((1, 24), np.uint8)
        image[0, 4:8] = 1
        image[0, 15:18] = 1
        original = image.copy()
        runs = extract_row_markings(image, row_step=1)
        self.assertEqual([(r['x_start'], r['x_end'], r['x'], r['y'], r['width_px'])
                          for r in runs], [(4, 7, 5.5, 0, 4), (15, 17, 16.0, 0, 3)])
        np.testing.assert_array_equal(image, original)

    def test_continuous_and_dashed_rectangular(self):
        image = np.zeros((17, 31), np.uint8)
        image[:, 3:5] = 1
        image[0:4, 12:15] = 1
        image[9:13, 12:15] = 1
        runs = extract_row_markings(image, row_step=3)
        self.assertEqual([r['y'] for r in runs if r['x_start'] == 3], [0, 3, 6, 9, 12, 15])
        self.assertEqual([r['y'] for r in runs if r['x_start'] == 12], [0, 3, 9, 12])
        self.assertTrue(all(r['width_px'] == 3 for r in runs if r['x_start'] == 12))

    def test_black_and_edge_runs(self):
        self.assertEqual(extract_row_markings(np.zeros((13, 29), np.uint8)), [])
        image = np.array([[1, 1, 0, 1, 0, 1, 1]], np.uint8)
        runs = extract_row_markings(image)
        self.assertEqual([(r['x_start'], r['x_end'], r['width_px']) for r in runs],
                         [(0, 1, 2), (3, 3, 1), (5, 6, 2)])
        self.assertTrue(runs[0]['touches_left_edge'])
        self.assertTrue(runs[-1]['touches_right_edge'])
        full = extract_row_markings(np.ones((2, 9), np.uint8), row_step=1)
        self.assertEqual(len(full), 2)
        self.assertTrue(all(r['width_px'] == 9 and r['touches_left_edge']
                            and r['touches_right_edge'] for r in full))

    def test_invalid_inputs(self):
        image = np.zeros((3, 7), np.uint8)
        for step in (0, -1, 1.5, True):
            with self.subTest(step=step), self.assertRaises(ValueError):
                extract_row_markings(image, step)
        for invalid in (image.astype(float), np.zeros((3, 7, 3), np.uint8),
                        np.full((3, 7), 255, np.uint8)):
            with self.assertRaises(ValueError):
                extract_row_markings(invalid)


if __name__ == '__main__':
    unittest.main(verbosity=2)
