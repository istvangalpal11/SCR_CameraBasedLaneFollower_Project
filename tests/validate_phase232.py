"""Focused camera-free Phase 2.3.2 association and evidence-preservation tests."""
import copy
import unittest
import numpy as np
from src.algorithms.LaneDetection.candidate_extractor import extract_row_markings
from src.algorithms.LaneDetection.fragment_extractor import extract_marking_fragments
from src.algorithms.LaneDetection.fragment_linker import link_marking_fragments


def fragments(image):
    return extract_marking_fragments(extract_row_markings(image))


def paint(image, ranges, x_function, width=3):
    for start, stop in ranges:
        for y in range(start, stop):
            x = round(x_function(y))
            image[y, x:x+width] = 1


class FragmentLinkerTests(unittest.TestCase):
    def assert_preserved(self, original, candidates):
        self.assertEqual([c['candidate_id'] for c in candidates], list(range(len(candidates))))
        expected_points = [id(p) for f in original for p in f['observations']]
        actual_points = [id(p) for c in candidates for p in c['observations']]
        self.assertEqual(len(actual_points), len(set(actual_points)))
        self.assertCountEqual(actual_points, expected_points)
        self.assertCountEqual([i for c in candidates for i in c['fragment_ids']],
                              [f['fragment_id'] for f in original])
        for c in candidates:
            self.assertEqual(c['point_count'], len(c['observations']))
            self.assertEqual(len(c['links']), len(c['fragment_ids']) - 1)
            self.assertTrue(all(b['y'] > a['y'] for a, b in zip(c['observations'], c['observations'][1:])))

    def test_continuous_marking_stays_one_fragment(self):
        image = np.zeros((120, 200), np.uint8)
        image[:, 30:35] = 1
        fs = fragments(image)
        candidates = link_marking_fragments(fs)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]['fragment_ids'], [fs[0]['fragment_id']])
        self.assertEqual(candidates[0]['links'], [])
        self.assert_preserved(fs, candidates)

    def test_straight_dashes_link_without_invented_points(self):
        image = np.zeros((250, 200), np.uint8)
        paint(image, ((0,25),(100,125),(220,245)), lambda y:40, width=5)
        fs = fragments(image)
        candidates = link_marking_fragments(fs)
        self.assertEqual(len(candidates), 1)
        c = candidates[0]
        self.assertEqual(c['point_count'], 15)
        self.assertEqual(len(c['links']), 2)
        self.assertEqual(c['max_gap_px'], 100)
        self.assertEqual(c['gaps'], [
            {'y_from':20,'y_to':100,'gap_px':80,'missing_scan_rows':15},
            {'y_from':120,'y_to':220,'gap_px':100,'missing_scan_rows':19}])
        self.assert_preserved(fs, candidates)

    def test_curved_dashes_form_one_smooth_chain(self):
        image = np.zeros((250, 250), np.uint8)
        paint(image, ((0,25),(100,125),(220,245)), lambda y:30+0.001*y*y)
        fs = fragments(image)
        candidates = link_marking_fragments(fs)
        self.assertEqual(len(fs), 3)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(len(candidates[0]['links']), 2)
        self.assert_preserved(fs, candidates)

    def test_parallel_and_multiple_boundaries_in_one_half(self):
        for xs in ((20,42),(12,34,56)):
            with self.subTest(xs=xs):
                image = np.zeros((250, 240), np.uint8)
                for x in xs:
                    paint(image, ((0,25),(100,125),(220,245)), lambda y,x=x:x)
                fs = fragments(image)
                candidates = link_marking_fragments(fs)
                self.assertEqual(len(candidates), len(xs))
                self.assertTrue(all(len(c['fragment_ids']) == 3 for c in candidates))
                self.assertTrue(all(len({p['x'] for p in c['observations']}) == 1 for c in candidates))
                self.assert_preserved(fs, candidates)

    def test_short_invalid_tangents_are_conservative(self):
        image = np.zeros((80, 120), np.uint8)
        paint(image, ((0,20),(30,40)), lambda y:30)
        fs = fragments(image)
        self.assertEqual([f['point_count'] for f in fs], [4,2])
        self.assertFalse(fs[-1]['start_tangent']['valid'])
        candidates = link_marking_fragments(fs)
        self.assertEqual(len(candidates), 1)  # Near a real, supported tangent.
        self.assertIsNone(fs[-1]['start_tangent']['dx_dy'])
        image[:] = 0
        paint(image, ((0,10),(20,30)), lambda y:30)
        fs = fragments(image)
        self.assertEqual(len(link_marking_fragments(fs)), 2)  # No valid tangent at either end.
        image[:] = 0
        paint(image, ((0,20),(60,70)), lambda y:30)
        fs = fragments(image)
        candidates = link_marking_fragments(fs)
        self.assertEqual(len(candidates), 2)  # Long gap cannot use an invalid tangent.
        self.assert_preserved(fs, candidates)

    def test_noise_singletons_and_edge_points_remain_present(self):
        image = np.zeros((250, 200), np.uint8)
        paint(image, ((0,25),(100,125),(220,245)), lambda y:40)
        image[60, 40] = 1  # Even an aligned singleton is not long-link evidence.
        image[245, 199] = 1
        fs = fragments(image)
        candidates = link_marking_fragments(fs)
        self.assertEqual(len(candidates), 3)
        self.assertEqual(sorted(c['point_count'] for c in candidates), [1,1,15])
        self.assert_preserved(fs, candidates)

    def test_crossing_stripe_does_not_become_longitudinal_boundary(self):
        image = np.zeros((200, 100), np.uint8)
        image[:, 48:53] = 1
        image[100:120, 10:91] = 1
        fs = fragments(image)
        candidates = link_marking_fragments(fs)
        self.assertTrue(all(len({p['width_px'] for p in c['observations']}) == 1 for c in candidates))
        self.assert_preserved(fs, candidates)

    def test_accepted_associations_do_not_cross(self):
        image = np.zeros((130, 140), np.uint8)
        for fn in (lambda y:30+0.2*y, lambda y:50-0.2*y):
            paint(image, ((0,25),(100,125)), fn)
        fs = fragments(image)
        self.assertEqual(len(fs), 4)
        candidates = link_marking_fragments(fs)
        self.assertEqual(sum(len(c['links']) for c in candidates), 1)
        self.assertEqual(len(candidates), 3)
        self.assert_preserved(fs, candidates)

    def test_ambiguous_outgoing_and_incoming_links_remain_separate(self):
        for incoming in (False,True):
            with self.subTest(incoming=incoming):
                image = np.zeros((130, 150), np.uint8)
                paint(image, ((0,25),) if not incoming else ((100,125),), lambda y:50)
                for x in (45,55):
                    paint(image, ((100,125),) if not incoming else ((0,25),), lambda y,x=x:x)
                fs = fragments(image)
                candidates = link_marking_fragments(fs)
                self.assertEqual(len(candidates), 3)
                self.assertTrue(all(c['links'] == [] for c in candidates))
                self.assert_preserved(fs, candidates)

    def test_completed_chain_validation_can_reject_connections(self):
        image = np.zeros((370, 150), np.uint8)
        for y,x in ((0,40),(100,52),(220,40),(340,52)):
            paint(image, ((y,y+25),), lambda yy,x=x:x)
        fs = fragments(image)
        candidates = link_marking_fragments(fs)
        self.assertGreater(len(candidates), 1)  # No forced oscillating boundary through all fragments.
        self.assert_preserved(fs, candidates)

    def test_empty_deterministic_input_order_and_unchanged_descriptors(self):
        self.assertEqual(link_marking_fragments([]), [])
        image = np.zeros((250, 200), np.uint8)
        for x in (20,60):
            paint(image, ((0,25),(100,125),(220,245)), lambda y,x=x:x)
        fs = fragments(image)
        original = copy.deepcopy(fs)
        result = link_marking_fragments(fs)
        self.assertEqual(result, link_marking_fragments(list(reversed(fs))))
        self.assertEqual(result, link_marking_fragments(fs))
        self.assertEqual(fs, original)
        self.assert_preserved(fs, result)
        for kwargs in ({'max_gap_px':0},{'max_width_ratio':1},{'ambiguity_margin':-1},
                       {'min_strong_points':2},{'row_step':0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                link_marking_fragments(fs, **kwargs)


if __name__ == '__main__':
    unittest.main(verbosity=2)
