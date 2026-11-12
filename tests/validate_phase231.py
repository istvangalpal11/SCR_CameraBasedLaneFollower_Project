"""Focused camera-free tests for Phase 2.3.1 visible fragment extraction."""
import copy
import unittest
import numpy as np
from src.algorithms.LaneDetection.candidate_extractor import extract_row_markings
from src.algorithms.LaneDetection.fragment_extractor import extract_marking_fragments


def observe(image):
    return extract_row_markings(image, row_step=5)


class FragmentExtractorTests(unittest.TestCase):
    def test_continuous_marking_and_endpoint_metadata(self):
        image = np.zeros((120, 180), np.uint8)
        image[:, 30:35] = 1
        obs = observe(image)
        original = copy.deepcopy(obs)
        fragments = extract_marking_fragments(obs)
        self.assertEqual(len(fragments), 1)
        f = fragments[0]
        self.assertEqual(f['point_count'], 24)
        self.assertIs(f['start_observation'], obs[0])
        self.assertIs(f['end_observation'], obs[-1])
        self.assertEqual((f['y_min'], f['y_max']), (0, 115))
        self.assertEqual(f['typical_width_px'], 5.0)
        self.assertTrue(f['start_tangent']['valid'] and f['end_tangent']['valid'])
        self.assertEqual(f['start_tangent']['dx_dy'], 0)
        self.assertEqual(obs, original)

    def test_three_separated_dashes_remain_three_fragments(self):
        image = np.zeros((145, 150), np.uint8)
        for y in (0, 60, 120):
            image[y:y+25, 40:45] = 1
        obs = observe(image)
        fragments = extract_marking_fragments(obs)
        self.assertEqual(len(fragments), 3)
        self.assertEqual([f['point_count'] for f in fragments], [5, 5, 5])
        self.assertEqual([f['y_min'] for f in fragments], [0, 60, 120])
        self.assertTrue(all(f['gaps'] == [] for f in fragments))
        self.assertEqual({id(p) for f in fragments for p in f['observations']}, {id(p) for p in obs})

    def test_smooth_continuous_curve(self):
        image = np.zeros((120, 180), np.uint8)
        for y in range(image.shape[0]):
            x = round(20 + 0.002 * y * y)
            image[y, x:x+3] = 1
        obs = observe(image)
        fragments = extract_marking_fragments(obs)
        self.assertEqual(len(fragments), 1)
        self.assertEqual(fragments[0]['observations'], obs)
        self.assertGreater(fragments[0]['end_tangent']['dx_dy'],
                           fragments[0]['start_tangent']['dx_dy'])

    def test_nearby_parallel_markings_in_one_half(self):
        image = np.zeros((120, 200), np.uint8)
        for x in (20, 26, 32):
            image[:, x:x+3] = 1
        fragments = extract_marking_fragments(observe(image))
        self.assertEqual(len(fragments), 3)
        self.assertEqual({f['start_observation']['x'] for f in fragments}, {21, 27, 33})
        self.assertTrue(all(f['point_count'] == 24 for f in fragments))

    def test_bottom_edge_singleton_and_two_point_fragment_are_preserved(self):
        image = np.zeros((100, 120), np.uint8)
        image[90:100, 118:120] = 1
        image[95:100, :2] = 1
        obs = observe(image)
        fragments = extract_marking_fragments(obs)
        self.assertEqual(sorted(f['point_count'] for f in fragments), [1, 2])
        self.assertTrue(any(f['touches_right_edge'] for f in fragments))
        self.assertTrue(any(f['touches_left_edge'] for f in fragments))
        self.assertTrue(all(f['y_max'] == 95 for f in fragments))
        for f in fragments:
            for name in ('start_tangent', 'end_tangent'):
                self.assertFalse(f[name]['valid'])
                self.assertIsNone(f[name]['dx_dy'])
        self.assertEqual(sum(f['point_count'] for f in fragments), len(obs))

    def test_small_missing_row_is_configurable_and_not_filled(self):
        image = np.zeros((50, 100), np.uint8)
        image[:, 20:25] = 1
        image[20, 20:25] = 0
        obs = observe(image)
        tolerant = extract_marking_fragments(obs, max_gap_rows=2)
        strict = extract_marking_fragments(obs, max_gap_rows=1)
        self.assertEqual(len(tolerant), 1)
        self.assertEqual(len(strict), 2)
        self.assertEqual(tolerant[0]['gaps'], [
            {'y_from': 15, 'y_to': 25, 'gap_px': 10, 'missing_scan_rows': 1}])
        self.assertEqual(tolerant[0]['observations'], obs)

    def test_crossing_stripe_is_preserved_but_not_automatic_continuation(self):
        image = np.zeros((100, 100), np.uint8)
        image[:, 48:53] = 1
        image[40:45, 10:91] = 1
        obs = observe(image)
        fragments = extract_marking_fragments(obs)
        self.assertEqual(len(fragments), 2)
        stripe = next(f for f in fragments if f['typical_width_px'] == 81)
        self.assertEqual(stripe['point_count'], 1)
        line = next(f for f in fragments if f is not stripe)
        self.assertEqual(line['point_count'], 19)
        self.assertTrue(all(p['width_px'] == 5 for p in line['observations']))
        assigned = [id(p) for f in fragments for p in f['observations']]
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual(set(assigned), {id(p) for p in obs})

    def test_empty_and_deterministic(self):
        self.assertEqual(extract_marking_fragments([]), [])
        image = np.zeros((60, 100), np.uint8)
        image[:, 20:25] = 1
        image[55, 40:43] = 1
        obs = observe(image)
        expected = extract_marking_fragments(obs)
        self.assertEqual(expected, extract_marking_fragments(obs))
        self.assertEqual(expected, extract_marking_fragments(list(reversed(obs))))
        self.assertEqual([f['fragment_id'] for f in expected], list(range(len(expected))))


if __name__ == '__main__':
    unittest.main(verbosity=2)
