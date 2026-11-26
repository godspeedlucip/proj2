import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


class LvTests(unittest.TestCase):
    def test_lv_phase_mapping_boundary_volume_and_animal_weights(self):
        from rstar4d.evaluation.lv_metrics import boundary_band, evaluate_lv, summarize_lv
        from rstar4d.config import metric_defaults

        coords = np.indices((7,7,7)) - 3
        sphere = (coords ** 2).sum(axis=0) <= 4
        center = (coords ** 2).sum(axis=0) == 0
        blank = np.zeros_like(sphere)
        valid = np.ones_like(sphere)
        valid[1,3,3] = False
        band = boundary_band(center, valid)
        np.testing.assert_array_equal(band, sphere & valid)

        cube = np.ones((9,9,9), dtype=bool)
        cube[4,4,4] = False
        _, y, x = 4, 4, 4
        self.assertTrue(boundary_band(cube, np.ones_like(cube))[6,y,x])
        self.assertFalse(boundary_band(cube, np.ones_like(cube))[7,y,x])

        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'masks.h5'
            with h5py.File(p,'w') as f:
                f['ed'] = sphere
                f['es'] = center
                f['empty'] = blank
            masks = [center, blank, sphere, blank]
            prediction = np.stack(masks).astype(np.float32)*.02+.01
            prediction[2,3,3,5] = .01
            source = {'target': prediction - .002, 'valid_mask': np.ones_like(prediction, dtype=bool)}
            meta = {'phase_labels':[90, 10, 70, 40], 'output_spacing_zyx_mm':[2.,3.,4.], 'temporal_topology':'ordered_states', 'ed_phase_label':70, 'es_phase_label':90}
            declarations = {
                70: {'path':p,'dataset':'ed','role':'ED'},
                90: {'path':p,'dataset':'es','role':'ES'},
                10: {'path':p,'dataset':'empty'},
            }
            evaluated = evaluate_lv(prediction, source, meta, declarations, metric_defaults(), lambda a: a>.025)
            self.assertAlmostEqual(evaluated['boundary_mae'], .002, places=7)
            self.assertAlmostEqual(evaluated['motion']['reference_delta_ml'], (sphere.sum()-1)*24/1000)
            self.assertAlmostEqual(evaluated['motion']['absolute_error_ml'], .024)
            self.assertAlmostEqual(evaluated['motion']['signed_bias_ml'], -.024)
            self.assertAlmostEqual(evaluated['ordered_endpoint_mae'], .002, places=7)
            self.assertEqual([r['boundary_status'] for r in evaluated['phases']],
                             ['ok','empty_boundary_roi','ok','missing_annotation'])
            empty = evaluate_lv(prediction, source, meta, declarations, metric_defaults(), lambda a: a>.1)
            self.assertEqual(empty['motion']['status'], 'prediction_unavailable')
            self.assertEqual(empty['phases'][0]['extraction_status'], 'empty_prediction')
            summary = summarize_lv([{'animal_id':'high','lv':dict(evaluated,boundary_mae=.01)},
                                    {'animal_id':'high','lv':dict(evaluated,boundary_mae=.03)},
                                    {'animal_id':'low','lv':evaluated}])
            self.assertAlmostEqual(summary['animal_equal']['boundary_mae']['mean'], .011, places=7)
            self.assertEqual(summary['coverage']['high']['evaluated_phases'], 4)
