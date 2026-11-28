import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
import torch
from rstar4d.inference.sliding_window import predict_volume
from rstar4d.evaluation.image_metrics import image_metrics
from rstar4d.evaluation.projection_metrics import projection_metrics


class Identity(torch.nn.Module):
    stage=2
    def forward(self,x): return x[:,:1]


class InferenceTests(unittest.TestCase):
    def test_tiled_identity_odd_shapes_and_temporal_axis(self):
        for t in [10,11]:
            a=np.random.default_rng(2).random((t,7,19,21),dtype=np.float32)*.02
            p=a.mean(0)
            out=predict_volume(Identity(),a,p,{'scale':.02,'offset':0},tile=(4,16,16),overlap=(2,8,8),device='cpu',amp=False)
            np.testing.assert_allclose(out,a,rtol=2e-6,atol=1e-8)
            self.assertEqual(out.shape,a.shape)

    def test_local3_manifest_preserves_topology_and_checkpoint_outputs(self):
        from tests.test_data import fixture
        from rstar4d.data.contracts import write_manifest
        from rstar4d.inference.predict import predict_manifest
        from rstar4d.models import create_model, forward_volume
        from rstar4d.models.stam import architecture as stam_architecture
        from rstar4d.models.ttl import architecture as ttl_architecture
        from rstar4d.training.checkpoint import atomic_save
        from validation.run_stam_smoke import seam_report

        torch.set_num_threads(1)
        torch.manual_seed(21)
        with tempfile.TemporaryDirectory() as d, torch.no_grad():
            root = Path(d)
            topologies = ('ordered_states', 'cyclic')
            rows = [fixture(root, 5, label, label, (3, 17, 19), label) for label in topologies]
            manifest = root/'manifest.jsonl'
            write_manifest(manifest, rows)
            model = create_model(2, stam={'mode': 'temporal_only'}, ttl={'mode': 'local3'}).eval()
            torch.nn.init.normal_(model.ttl.temporal.weight, std=0.02)
            norm = {'scale': 0.02, 'offset': 0.0}
            with h5py.File(rows[0]['path']) as f:
                cbct, prior = f['cbct'][:], f['prior'][:]
            x = torch.from_numpy(np.stack([cbct / .02, np.broadcast_to(prior / .02, cbct.shape)]))[None]
            direct = {label: forward_volume(model, x, label)[0, 0].numpy() * .02 for label in topologies}
            self.assertGreater(np.max(np.abs(direct['cyclic'][0] - direct['ordered_states'][0])), 1e-7)
            checkpoint = root/'child.pt'
            atomic_save(checkpoint, dict(schema_version=1, checkpoint_kind='weights', stage=2,
                model_profile='official_v1', normalization=norm, intensity_domain='mu',
                stam={'architecture': stam_architecture('temporal_only')},
                ttl={'architecture': ttl_architecture('local3')}, state_dict=model.state_dict()))
            tile, overlap = (4, 16, 16), (2, 8, 8)
            for mode in ('full', 'tiled'):
                predict_manifest(checkpoint, manifest, root/mode, 'all', mode, tile, overlap, 'cpu', False)
                predictions = {}
                for label in topologies:
                    expected = predict_volume(model, cbct, prior, norm, mode, tile, overlap,
                                              'cpu', False, temporal_topology=label)
                    with h5py.File(root/mode/label/'prediction.h5') as f:
                        predictions[label] = f['prediction'][:]
                    np.testing.assert_array_equal(predictions[label], expected)
                    if mode == 'full':
                        np.testing.assert_array_equal(expected, direct[label])
                self.assertGreater(np.max(np.abs(predictions['cyclic'][0] - predictions['ordered_states'][0])), 1e-7)
            seams = seam_report(root/'full', root/'tiled', tile, overlap)
            self.assertEqual(set(seams), set(topologies))
            self.assertTrue(all(np.isfinite(row['seam_mae']) for row in seams.values()))

    def test_metric_units_empty_roi_and_window_mask(self):
        a=np.full((10,2,16,16),.02,np.float32); mask=np.ones_like(a,bool)
        same=image_metrics(a,a,mask)
        self.assertEqual(same['rmse'],0); self.assertAlmostEqual(same['ssim'],1,6)
        shifted=image_metrics(a+.002,a,mask)
        self.assertAlmostEqual(shifted['rmse'],.002,7)
        empty=image_metrics(a,a,mask,roi=np.zeros_like(mask))
        self.assertIsNone(empty['rmse']); self.assertIsNone(empty['ssim'])
        mask[:]=False; self.assertIsNone(image_metrics(a,a,mask)['ssim'])

    def test_projection_inapplicable(self):
        for meta in [{'intensity_domain':'calibrated_mu','full_object_fov':True},
                     {'intensity_domain':'mu','full_object_fov':False}]:
            result=projection_metrics(None,meta,None)
            self.assertEqual(result['status'],'not_applicable')

    def test_stam_and_legacy_infer_without_external_sam(self):
        import json
        from tests.test_data import fixture
        from rstar4d.cli import main
        from rstar4d.data.contracts import write_manifest
        from rstar4d.models import create_model
        from rstar4d.models.stam import architecture
        from rstar4d.training.checkpoint import atomic_save

        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rows = [fixture(root, t, f'p{t}', f'case{t}', shape=(3, 17, 19)) for t in (10, 11)]
            manifest = root/'manifest.jsonl'
            write_manifest(manifest, rows)
            standalone_sam = root/'sam.pth'
            standalone_sam.touch()

            for mode in ('off', 'joint'):
                model = create_model(2, stam={'mode': mode})
                payload = dict(schema_version=1, checkpoint_kind='weights', stage=2, model_profile='official_v1',
                               normalization={'scale': .02, 'offset': 0}, intensity_domain='mu')
                if mode == 'joint':
                    torch.nn.init.normal_(model.stam.output_projection.weight, std=.003)
                    payload.update(stam={'architecture': architecture(mode)}, config={
                        'stam': {'sam_checkpoint': str(standalone_sam)}})
                    standalone_sam.rename(root/'sam.hidden')
                expected = {}
                for row in rows:
                    with h5py.File(row['path']) as f:
                        expected[row['case_id']] = predict_volume(
                            model, f['cbct'], f['prior'], payload['normalization'],
                            mode='tiled', tile=(4,16,16), overlap=(2,8,8), device='cpu', amp=False)
                checkpoint = root/f'{mode}.pt'
                atomic_save(checkpoint, dict(payload, state_dict=model.state_dict()))
                out = root/mode
                self.assertEqual(main(['infer','--checkpoint',str(checkpoint),'--manifest',str(manifest),'--out',str(out),
                                       '--split-name','all','--mode','tiled','--tile','4','16','16',
                                       '--overlap','2','8','8','--device','cpu','--fp32']), 0)
                for row in rows:
                    with h5py.File(out/row['case_id']/'prediction.h5') as f:
                        np.testing.assert_array_equal(f['prediction'][:], expected[row['case_id']])
                        meta = json.loads(f.attrs['metadata_json'])
                        self.assertEqual(meta['inference']['stam']['mode'], mode)
                        self.assertEqual(len(meta['phase_labels']), len(expected[row['case_id']]))

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
            meta = {'phase_labels':[90, 10, 70, 40], 'output_spacing_zyx_mm':[2.,3.,4.]}
            declarations = {
                70: {'path':p,'dataset':'ed','role':'ED'},
                90: {'path':p,'dataset':'es','role':'ES'},
                10: {'path':p,'dataset':'empty'},
            }
            evaluated = evaluate_lv(prediction, source, meta, declarations, metric_defaults(), lambda a: a>.025)
            self.assertAlmostEqual(evaluated['boundary_mae'], .002, places=7)
            self.assertAlmostEqual(evaluated['motion']['reference_delta_ml'], (sphere.sum()-1)*24/1000)
            self.assertAlmostEqual(evaluated['motion']['absolute_error_ml'], .024)
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

if __name__=='__main__': unittest.main()
