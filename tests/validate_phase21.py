"""Focused camera-free checks for the Phase 2.1 continuity baseline."""
import copy
import unittest
import numpy as np
from src.algorithms.LaneDetection.candidate_extractor import extract_row_markings
from src.algorithms.LaneDetection.boundary_builder import build_boundary_candidates


def observations(image):
    return extract_row_markings(image, row_step=5)


class BoundaryBuilderTests(unittest.TestCase):
    def test_continuous_straight_and_edge_flags(self):
        image = np.zeros((120, 180), np.uint8)
        image[:, :5] = 1
        obs = observations(image)
        original = copy.deepcopy(obs)
        candidates = build_boundary_candidates(obs)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]['point_count'], 24)
        self.assertEqual((candidates[0]['y_min'], candidates[0]['y_max']), (0, 115))
        self.assertEqual(candidates[0]['gaps'], [])
        self.assertTrue(all(p['touches_left_edge'] for p in candidates[0]['observations']))
        self.assertIs(candidates[0]['observations'][0], obs[0])
        self.assertEqual(obs, original)

    def test_two_parallel_and_multiple_in_same_half(self):
        for xs in ((12, 36), (12, 36, 60)):
            with self.subTest(xs=xs):
                image = np.zeros((120, 200), np.uint8)
                for x in xs:
                    image[:, x:x+3] = 1
                candidates = build_boundary_candidates(observations(image))
                self.assertEqual(len(candidates), len(xs))
                self.assertTrue(all(c['point_count'] == 24 for c in candidates))
                self.assertTrue(all(len({p['x'] for p in c['observations']}) == 1 for c in candidates))

    def test_smooth_curve(self):
        image = np.zeros((120, 180), np.uint8)
        for y in range(image.shape[0]):
            x = round(20 + 0.002 * y * y)
            image[y, x:x+3] = 1
        obs = observations(image)
        candidates = build_boundary_candidates(obs)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]['observations'], obs)
        self.assertGreater(obs[-1]['x'] - obs[0]['x'], 20)

    def test_dashes_preserve_only_real_evidence(self):
        image = np.zeros((125, 180), np.uint8)
        for y in (0, 50, 100):
            image[y:y+25, 40:45] = 1
        obs = observations(image)
        candidates = build_boundary_candidates(obs)
        self.assertEqual(len(candidates), 1)
        c = candidates[0]
        self.assertEqual(c['observations'], obs)
        self.assertEqual(c['point_count'], 15)
        self.assertEqual(c['max_gap_px'], 30)
        self.assertEqual(c['gaps'], [
            {'y_from': 20, 'y_to': 50, 'gap_px': 30, 'missing_scan_rows': 5},
            {'y_from': 70, 'y_to': 100, 'gap_px': 30, 'missing_scan_rows': 5}])
        self.assertEqual({id(p) for p in c['observations']}, {id(p) for p in obs})

    def test_off_direction_noise_stays_unassigned(self):
        image = np.zeros((100, 100), np.uint8)
        image[:, 20:25] = 1
        image[30, 20:25] = 0
        image[30, 29:34] = 1
        obs = observations(image)
        candidates = build_boundary_candidates(obs)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]['point_count'], 19)
        self.assertTrue(all(p['x_start'] == 20 for p in candidates[0]['observations']))
        self.assertEqual(len(obs), 20)

    def test_crossing_stripe_is_not_joined_by_proximity(self):
        image = np.zeros((100, 100), np.uint8)
        image[:, 48:53] = 1
        image[40:45, 10:91] = 1  # Stripe center exactly matches the vertical line.
        obs = observations(image)
        candidates = build_boundary_candidates(obs)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]['point_count'], 19)
        self.assertTrue(all(p['width_px'] == 5 for p in candidates[0]['observations']))
        self.assertIn({'y_from': 35, 'y_to': 45, 'gap_px': 10, 'missing_scan_rows': 1},
                      candidates[0]['gaps'])
        self.assertTrue(any(p['width_px'] == 81 for p in obs))

    def test_large_gaps_and_singletons_need_evidence(self):
        image = np.zeros((180, 100), np.uint8)
        image[0:20, 20:25] = 1
        image[140:180, 20:25] = 1
        candidates = build_boundary_candidates(observations(image))
        self.assertEqual(len(candidates), 2)  # Never bridge a gap above the configured limit.
        image[:] = 0
        image[0, 20:25] = 1
        image[30:80, 20:25] = 1
        candidates = build_boundary_candidates(observations(image))
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]['y_min'], 30)  # Singleton cannot justify the long link.

    def test_empty_deterministic_and_configuration(self):
        self.assertEqual(build_boundary_candidates([]), [])
        image = np.zeros((50, 80), np.uint8)
        image[:, 10:13] = image[:, 40:43] = 1
        obs = observations(image)
        expected = build_boundary_candidates(obs)
        self.assertEqual(build_boundary_candidates(obs), expected)
        self.assertEqual(build_boundary_candidates(list(reversed(obs))), expected)
        for setting in ({'row_step': 0}, {'max_link_gap_rows': 0},
                        {'min_candidate_points': 0}, {'max_horizontal_jump_px': -1},
                        {'max_direction_change_deg': 0}, {'max_width_ratio': 0.5}):
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                build_boundary_candidates(obs, **setting)


if __name__ == '__main__':
    unittest.main(verbosity=2)
