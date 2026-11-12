"""Focused camera-free tests for independent Phase 2.2 RANSAC candidates."""
import copy
import unittest
import numpy as np
from src.algorithms.LaneDetection.candidate_extractor import extract_row_markings
from src.algorithms.LaneDetection.ransac_boundary_builder import build_ransac_candidates


def extract(image):
    return extract_row_markings(image, row_step=5)


class RansacBuilderTests(unittest.TestCase):
    def test_continuous_straight_and_preserved_evidence(self):
        image = np.zeros((160, 200), np.uint8)
        image[:, :3] = 1
        obs = extract(image)
        original = copy.deepcopy(obs)
        candidates = build_ransac_candidates(obs)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]['point_count'], 32)
        self.assertEqual(candidates[0]['gaps'], [])
        self.assertTrue(all(p['touches_left_edge'] for p in candidates[0]['observations']))
        self.assertIs(candidates[0]['observations'][0], obs[0])
        self.assertEqual(obs, original)

    def test_nearby_parallel_markings_stay_separate(self):
        image = np.zeros((180, 200), np.uint8)
        for y in range(image.shape[0]):
            x = 40 + round(0.02 * y)
            image[y, x:x+3] = image[y, x+6:x+9] = 1
        candidates = build_ransac_candidates(extract(image))
        self.assertEqual(len(candidates), 2)
        self.assertEqual(sorted(c['point_count'] for c in candidates), [36, 36])
        for c in candidates:
            offsets = {p['x_start'] - round(0.02 * p['y']) for p in c['observations']}
            self.assertIn(offsets, ({40}, {46}))

    def test_multiple_markings_on_same_image_side(self):
        image = np.zeros((180, 240), np.uint8)
        for x in (12, 32, 52):
            image[:, x:x+3] = 1
        candidates = build_ransac_candidates(extract(image))
        self.assertEqual(len(candidates), 3)
        self.assertEqual({c['observations'][0]['x'] for c in candidates}, {13, 33, 53})
        self.assertTrue(all(c['point_count'] == 36 for c in candidates))

    def test_long_dashed_gaps_do_not_invent_evidence(self):
        image = np.zeros((400, 180), np.uint8)
        for y in (0, 180, 350):
            image[y:y+25, 40:45] = 1
        obs = extract(image)
        candidates = build_ransac_candidates(obs)
        self.assertEqual(len(candidates), 1)
        c = candidates[0]
        self.assertEqual(c['observations'], obs)
        self.assertEqual(c['point_count'], 15)
        self.assertEqual(c['max_gap_px'], 160)
        self.assertEqual(c['gaps'], [
            {'y_from': 20, 'y_to': 180, 'gap_px': 160, 'missing_scan_rows': 31},
            {'y_from': 200, 'y_to': 350, 'gap_px': 150, 'missing_scan_rows': 29}])
        self.assertEqual({id(p) for p in c['observations']}, {id(p) for p in obs})

    def test_smooth_quadratic_and_normalized_model(self):
        image = np.zeros((200, 300), np.uint8)
        for y in range(image.shape[0]):
            x = round(20 + 0.0015 * y * y)
            image[y, x:x+3] = 1
        obs = extract(image)
        candidates = build_ransac_candidates(obs)
        self.assertEqual(len(candidates), 1)
        c = candidates[0]
        self.assertEqual(c['point_count'], len(obs))
        normalized_y = (np.array([p['y'] for p in obs]) - c['y_offset']) / c['y_scale']
        errors = np.array([p['x'] for p in obs]) - np.polyval(c['model_coefficients'], normalized_y)
        self.assertLess(np.max(np.abs(errors)), 1)
        self.assertAlmostEqual(c['rms_residual_px'], float(np.sqrt(np.mean(errors**2))))

    def test_isolated_random_outliers_do_not_corrupt_main_marking(self):
        image = np.zeros((200, 320), np.uint8)
        image[:, 30:33] = 1
        rng = np.random.default_rng(7)
        for _ in range(24):
            y, x = int(rng.integers(0, 40)) * 5, int(rng.integers(120, 310))
            image[y, x] = 1
        candidates = build_ransac_candidates(extract(image))
        main = max(candidates, key=lambda c: c['point_count'])
        self.assertEqual(main['point_count'], 40)
        self.assertTrue(all(p['x'] == 31 for p in main['observations']))

    def test_distracting_horizontal_stripe(self):
        image = np.zeros((200, 300), np.uint8)
        image[:, 30:33] = 1
        image[60:65, 80:241] = 1
        obs = extract(image)
        candidates = build_ransac_candidates(obs)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]['point_count'], 40)
        self.assertTrue(all(p['width_px'] == 3 for p in candidates[0]['observations']))
        self.assertEqual(len(obs), 41)

    def test_aligned_crossing_is_a_documented_semantic_ambiguity(self):
        image = np.zeros((200, 100), np.uint8)
        image[:, 48:53] = 1
        image[60:65, 10:91] = 1  # Same center as the line: XY alone cannot identify paint type.
        candidates = build_ransac_candidates(extract(image))
        self.assertEqual(len(candidates), 1)
        self.assertTrue(any(p['width_px'] == 81 for p in candidates[0]['observations']))

    def test_empty_insufficient_and_short_coverage(self):
        self.assertEqual(build_ransac_candidates([]), [])
        image = np.zeros((30, 100), np.uint8)
        image[:, 30:33] = 1
        self.assertEqual(build_ransac_candidates(extract(image)), [])
        self.assertEqual(build_ransac_candidates(extract(image)[:2]), [])
        image = np.zeros((5, 100), np.uint8)
        image[:, ::5] = 1
        self.assertEqual(build_ransac_candidates(extract(image)), [])

    def test_deterministic_disjoint_candidates_and_limits(self):
        image = np.zeros((180, 200), np.uint8)
        image[:, 12:15] = image[:, 42:45] = image[:, 72:75] = 1
        obs = extract(image)
        first = build_ransac_candidates(obs, random_seed=42)
        self.assertEqual(first, build_ransac_candidates(obs, random_seed=42))
        self.assertEqual(first, build_ransac_candidates(list(reversed(obs)), random_seed=42))
        assigned = [id(p) for c in first for p in c['observations']]
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual(set(assigned), {id(p) for p in obs})
        self.assertEqual([c['candidate_id'] for c in first], [0, 1, 2])
        self.assertEqual(len(build_ransac_candidates(obs, max_candidates=1)), 1)
        for c in first:
            self.assertEqual(len({p['y'] for p in c['observations']}), c['point_count'])
        for kwargs in ({'max_iterations': 0}, {'random_seed': -1}, {'min_inliers': 2},
                       {'max_candidates': 0}, {'inlier_threshold_px': 0}, {'min_y_span_px': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_ransac_candidates(obs, **kwargs)


if __name__ == '__main__':
    unittest.main(verbosity=2)
