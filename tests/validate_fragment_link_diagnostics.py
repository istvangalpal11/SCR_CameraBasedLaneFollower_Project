"""Diagnostic-only regressions: linker outputs and original observations stay unchanged."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
from src.algorithms.LaneDetection.fragment_linker import link_marking_fragments
from tests.validate_phase232 import fragments, paint


class FragmentLinkDiagnosticsTests(unittest.TestCase):
    def test_disabled_mode_skips_collection_and_enabled_preserves_outputs(self):
        image = np.zeros((250, 150), np.uint8)
        paint(image, ((0,25),(100,125),(220,245)), lambda y:40)
        fs = fragments(image)
        before = copy.deepcopy(fs)
        with patch('src.algorithms.LaneDetection.fragment_linker._collect_pair_diagnostics',
                   side_effect=AssertionError('Disabled mode collected diagnostics')):
            expected = link_marking_fragments(fs)
        records = []
        actual = link_marking_fragments(fs, diagnostics=records)
        self.assertEqual(expected, actual)
        self.assertEqual(fs, before)
        accepted = {(r['source_fragment_id'], r['target_fragment_id']) for r in records if r['status']=='accepted'}
        links = {(l['source_fragment_id'], l['target_fragment_id']) for c in actual for l in c['links']}
        self.assertEqual(accepted, links)
        self.assertTrue(records)
        self.assertTrue(all(r['status'] in ('accepted','rejected') for r in records))
        self.assertFalse(any('checks' in link for c in actual for link in c['links']))

    def test_width_rejection_has_metrics_and_unreached_checks(self):
        image = np.zeros((130, 120), np.uint8)
        image[0:25,48:53] = 1
        image[100:125,10:91] = 1
        records=[]
        link_marking_fragments(fragments(image),diagnostics=records)
        row=records[0]
        self.assertIn('Width ratio',row['reason'])
        self.assertFalse(row['checks']['width']['passed'])
        self.assertTrue(row['checks']['width']['reached'])
        self.assertFalse(row['checks']['direction']['reached'])
        self.assertFalse(row['checks']['lateral']['reached'])
        self.assertEqual(row['lateral_error_px'],0)
        self.assertEqual(row['direction_difference_deg'],0)
        self.assertGreater(row['width_ratio'],2)

    def test_gap_support_and_invalid_tangent_reasons(self):
        cases=[(((0,25),(220,245)),'Longitudinal gap'),
               (((0,20),(60,70)),'Weak fragment support'),
               (((0,10),(20,30)),'Neither endpoint')]
        for ranges,reason in cases:
            with self.subTest(reason=reason):
                image=np.zeros((250,120),np.uint8)
                paint(image,ranges,lambda y:40)
                records=[]
                link_marking_fragments(fragments(image),diagnostics=records)
                self.assertTrue(any(reason in r['reason'] for r in records))

    def test_ambiguity_and_crossing_reasons_match_decisions(self):
        scenes=[]
        image=np.zeros((130,150),np.uint8)
        paint(image,((0,25),),lambda y:50)
        for x in (45,55):paint(image,((100,125),),lambda y,x=x:x)
        scenes.append((image,'Ambiguity'))
        image=np.zeros((130,140),np.uint8)
        for fn in (lambda y:30+0.2*y,lambda y:50-0.2*y):
            paint(image,((0,25),(100,125)),fn)
        scenes.append((image,'intersects an accepted link'))
        image=np.zeros((200,100),np.uint8)
        image[:,48:53]=1;image[100:120,10:91]=1
        scenes.append((image,'intersects an observed fragment'))
        for image,reason in scenes:
            with self.subTest(reason=reason):
                fs=fragments(image);records=[]
                expected=link_marking_fragments(fs)
                self.assertEqual(expected,link_marking_fragments(fs,diagnostics=records))
                self.assertTrue(any(reason in r['reason'] for r in records))
                self.assertFalse(any(r['status']=='pending' for r in records))

    def test_chain_fit_reports_failure_detail(self):
        image=np.zeros((370,150),np.uint8)
        for y,x in ((0,40),(100,52),(220,40),(340,52)):
            paint(image,((y,y+25),),lambda yy,x=x:x)
        fs=fragments(image);records=[]
        expected=link_marking_fragments(fs)
        self.assertEqual(expected,link_marking_fragments(fs,diagnostics=records))
        failed=[r for r in records if r['reason'].startswith('Chain-fit validation')]
        self.assertTrue(failed)
        self.assertTrue(all(r['chain_fit_reason'] for r in failed))
        self.assertTrue(all(r['checks']['chain_fit']['passed'] is False for r in failed))


if __name__=='__main__':
    unittest.main(verbosity=2)
