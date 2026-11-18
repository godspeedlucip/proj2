import importlib.util
import itertools
import unittest
import numpy as np


class CardiacExperimentsTests(unittest.TestCase):
    def test_matrix_only_changes_three_presets(self):
        self.assertIsNotNone(importlib.util.find_spec('experiments.cardiac_ablation'))
        from experiments.cardiac_ablation import matrix
        from rstar4d.config import validate_train
        base = validate_train({'stage2_crop_mode':'core_with_context'})
        variants = matrix(base)
        self.assertEqual(set(variants),{'baseline','M','P','B','MP','MB','PB','MPB'})
        factors = set()
        for row in variants.values():
            ext = row['extensions']
            factors.add((ext['motion']['mode'],ext['prior']['mode'],ext['boundary']['enabled']))
            ext['motion']['mode'] = ext['prior']['mode'] = 'off'
            ext['boundary']['enabled'] = False
            self.assertEqual(row,base)
        self.assertEqual(factors,set(itertools.product(('off','learned'),('off','geometry'),(False,True))))

    def test_comparison_intersects_keys_before_averaging_animals(self):
        self.assertIsNotNone(importlib.util.find_spec('experiments.cardiac_ablation'))
        from experiments.cardiac_ablation import paired_differences
        def row(case,animal,phase,value):
            return {'case_id':case,'animal_id':animal,'phase_label':phase,'structure':'lv',
                    'surface_dice':value,'status':'ok' if value is not None else 'missing_prediction'}
        baseline = [row('a1','a',5,.2),row('a1','a',6,.3),row('b1','b',1,.5)]
        other = [row('a1','a',5,.4),row('a1','a',6,None),row('b1','b',1,.8)]
        result = paired_differences(baseline,other,'surface_dice','lv')
        self.assertEqual(result['valid_states'],2)
        self.assertEqual(result['valid_animals'],2)
        self.assertAlmostEqual(result['mean_delta'],.25)

    def test_layout_control_freezes_every_other_phase(self):
        from rstar4d.simulation.cardiac_layouts import layout_summaries
        angles = np.deg2rad([0,20,40,80,120,175])
        a,b = layout_summaries(angles,[3,7,3,7,3,7],[7,3],3,[0,2],[2,4])
        np.testing.assert_array_equal(a[0],b[0])
        self.assertEqual(a[1,12],b[1,12])
        self.assertFalse(np.array_equal(a[1,:12],b[1,:12]))

    def test_paired_report_rejects_different_reference_annotations(self):
        from experiments.cardiac_ablation import check_common_references
        a = {'case_id':'x','paired_sha256':'original','annotations_sha256':'a'}
        with self.assertRaisesRegex(ValueError,'reference image and annotation'):
            check_common_references([a],[{**a,'annotations_sha256':'b'}])
        control = {'reference_paired_sha256':'original','reference_annotations_sha256':'a',
                   'source_projections_sha256':'views','phase':5}
        check_common_references([{**a,'paired_sha256':'A','cardiac_layout':control}],
                                [{**a,'paired_sha256':'B','cardiac_layout':control}],True)
