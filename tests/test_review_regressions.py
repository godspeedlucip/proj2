import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import h5py
import numpy as np
from tests.test_data import fixture
from tests.test_training import inputs
from rstar4d.config import train_defaults, simulation_defaults
from rstar4d.data.contracts import read_manifest, write_manifest, basic_metadata
from rstar4d.data.paired import import_paired
from rstar4d.data.sampling import PatchDataset
from rstar4d.simulation.pipeline import simulate_manifest
from rstar4d.training.trainer import train
from rstar4d.evaluation.report import projection_summary
from rstar4d.utils.io import atomic_jsonl, read_json, sha256


class ReviewRegressions(unittest.TestCase):
    def test_projection_case_animal_aggregation_and_nulls(self):
        cases=[{'animal_id':animal,'projection':{'status':'ok','mae':v,'rmse':v,'relative_l2':v,
            'ncc':None,'constant_views':2}} for animal,v in [('a',1.),('a',3.),('b',8.)]]
        cases.append({'animal_id':'c','projection':{'status':'not_applicable','reason':'calibrated'}})
        result=projection_summary(cases)
        self.assertEqual(result['metrics']['mae']['case_equal']['mean'],4.)
        self.assertEqual(result['metrics']['mae']['animal_equal']['mean'],5.)
        self.assertIsNone(result['metrics']['ncc']['case_equal']['mean'])
        self.assertEqual(result['constant_views'],6); self.assertEqual(result['skipped_cases'],1)

    def test_reordered_manifest_preserves_draws(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); rows=[fixture(root,case='a'),fixture(root,case='b')]
            cfg=train_defaults(); cfg['stage1_patch']=[1,16,16]
            a=PatchDataset(rows,cfg,1,2,12); b=PatchDataset(rows[::-1],cfg,1,2,12)
            self.assertEqual([a[i]['case_id'] for i in range(12)],[b[i]['case_id'] for i in range(12)])

    def test_impossible_selected_animal_cannot_be_silently_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); bad=fixture(root,animal='bad',case='bad'); good=fixture(root,animal='good',case='good')
            with h5py.File(bad['path'],'r+') as f:
                f['valid_mask'][:]=0; f['valid_mask'][:,:,0,0]=1
            cfg=train_defaults(); cfg.update(stage1_patch=[1,16,16],max_attempts=32)
            data=PatchDataset([bad,good],cfg,1,0,100)
            errors=0
            for i in range(100):
                try: data[i]
                except ValueError as exc:
                    errors+=1; self.assertIn('bad',str(exc)); self.assertIn('16',str(exc))
            self.assertGreater(errors,25)

    def test_image_import_rejects_contradictory_contracts(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); r=fixture(root,t=2)
            with h5py.File(r['path'],'r') as f:
                import json
                meta=json.loads(f.attrs['metadata_json'])
            base={'case_id':'pair','animal_id':'a','metadata':meta,
                'source_metadata':{k:copy.deepcopy(meta) for k in ['cbct','prior','target','valid_mask']}}
            for k in ['cbct','prior','target','valid_mask']: base[k]={'path':r['path'],'dataset':k}
            for change in ['phase','calibration','geometry']:
                item=copy.deepcopy(base)
                if change=='phase': item['source_metadata']['target']['phase_labels']=[2,1]
                if change=='calibration':
                    for m in [item['metadata']]+list(item['source_metadata'].values()):
                        m['calibration_application']='post_fdk'; m['intensity_domain']='calibrated_mu'
                        m['calibration']={'mode':'post_fdk','coefficients':[1,0],'provenance':'regression fixture'}
                    item['source_metadata']['target']['calibration']['coefficients']=[2,0]
                if change=='geometry':
                    item['metadata']['geometry']={}
                    for m in item['source_metadata'].values(): m['geometry']={}
                atomic_jsonl(root/'sources.jsonl',[item])
                with self.subTest(change=change),self.assertRaises(ValueError):
                    import_paired(root/'sources.jsonl',root/change)

    def test_resume_cannot_write_foreign_or_empty_run(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); cfg,manifest,split=inputs(root); cfg['epochs']={'1':1,'2':1}
            train(cfg,manifest,split,root/'a',1)
            other=root/'b'; other.mkdir(); marker=other/'keep.txt'; marker.write_text('unrelated')
            for dest in [other,root/'empty']:
                with self.subTest(destination=str(dest)),self.assertRaises(ValueError):
                    train(cfg,manifest,split,dest,1,resume=root/'a/checkpoints/last.pt')
            self.assertEqual(marker.read_text(),'unrelated')
            self.assertFalse((other/'resolved_config.json').exists())
            # A complete directory migration retains ownership and historical best.
            import shutil
            migrated=root/'migrated'; shutil.copytree(root/'a',migrated)
            result=train(cfg,manifest,split,migrated,1,resume=migrated/'checkpoints/last.pt')
            self.assertTrue(result['complete']); self.assertTrue(Path(result['checkpoint']).is_file())

    def test_excluded_ambiguous_identity_does_not_block_selected_case(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); raw=root/'raw.jsonl'
            atomic_jsonl(raw,[{'case_id':'chosen','animal_id':'a','phase_paths':{'1':'1.nii','2':'2.nii'}},
                {'case_id':'excluded','animal_id':None,'phase_paths':{}}])
            cfg=simulation_defaults(); cfg.update(cycle_csv='cycle.csv',case_ids=['chosen'])
            prepared=fixture(root,t=2,case='chosen')
            with patch('rstar4d.simulation.pipeline.backend_fingerprint',return_value={}),patch('rstar4d.simulation.pipeline.simulate_case',return_value=prepared):
                result=simulate_manifest(raw,cfg,root/'prepared')
            self.assertEqual(len(read_manifest(result)),1)

    def test_failed_expansion_marks_current_request_incomplete(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); raw=root/'raw.jsonl'; out=root/'prepared'
            atomic_jsonl(raw,[{'case_id':k,'animal_id':k,'phase_paths':{'1':'1.nii','2':'2.nii'}} for k in ['a','b']])
            record=fixture(root,t=2,case='a'); cfg=simulation_defaults(); cfg.update(cycle_csv='cycle.csv',limit=1)
            with patch('rstar4d.simulation.pipeline.backend_fingerprint',return_value={}),patch('rstar4d.simulation.pipeline.simulate_case',return_value=record):
                simulate_manifest(raw,cfg,out)
            cfg['limit']=None
            with patch('rstar4d.simulation.pipeline.backend_fingerprint',return_value={}),patch('rstar4d.simulation.pipeline.simulate_case',side_effect=[record,RuntimeError('interrupted')]):
                with self.assertRaises(RuntimeError): simulate_manifest(raw,cfg,out)
            status=read_json(out/'generation_summary.json')
            self.assertFalse(status['complete']); self.assertEqual(status['requested_case_ids'],['a','b'])
            self.assertEqual(status['completed_case_ids'],['a'])


if __name__=='__main__': unittest.main()
