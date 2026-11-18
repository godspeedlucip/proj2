"""Regression checks for repeated acquisitions of a single synthetic subject."""
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch

from experiments.cardiac_ablation import compare, paired_differences
from rstar4d.config import metric_defaults, validate_train
from rstar4d.data.acquisition import load_geometry
from rstar4d.data.cardiac_annotations import prepare_annotations, STRUCTURES
from rstar4d.data.contracts import basic_metadata, read_manifest, validate_case, validate_manifest, write_case, write_manifest
from rstar4d.data.sampling import PatchDataset, validation_samples
from rstar4d.evaluation.image_metrics import image_metrics
from rstar4d.evaluation.report import evaluate_predictions
from rstar4d.inference.predict import predict_manifest
from rstar4d.simulation.backend import geometry, output_grid, project
from rstar4d.simulation.cardiac_layouts import angle_layouts
from rstar4d.simulation.reference import SeriesSettings
from rstar4d.training.trainer import train
from rstar4d.utils.io import atomic_json, atomic_jsonl, canonical_hash, read_json, read_jsonl, sha256
from tests.test_data import fixture
from validation.cardiac_smoke import contours


def run_joint_phantom(root):
    """Two labeled layouts of one simulated case, with an independent validation animal."""
    shape = (8,24,24)
    labels = [2,9]
    settings = SeriesSettings(voxel_spacing_multiplier=1.,n_detector_rows=48,n_detector_cols=48,
        d_detector_row_mm=1.,d_detector_col_mm=1.,padding_row_pixels=4,padding_col_pixels=4,
        fov_output_pixels=24,fov_source_pixels=24,fov_mask_shrink_pixels=2)
    geo = geometry(shape,[1.]*3,settings)
    z,y,x = np.indices(shape)
    masks = []
    for shift in (0,1):
        heart = ((z-3.5)/3)**2+((y-12+shift)/7)**2+((x-12)/8)**2 < 1
        lv = ((z-3.5)/2)**2+((y-12+shift)/4)**2+((x-13)/3)**2 < 1
        masks.append(heart.astype(int)+lv)
    roi = np.stack(masks)
    target = np.asarray(.003 + .016*(roi>0) + .019*(roi==2),np.float32)
    meta = basic_metadata(2,[1.]*3,phase_labels=labels,geometry=output_grid(geo,settings),
        source_shape_zyx=list(shape),source_spacing_zyx_mm=[1.]*3,settings=asdict(settings))
    path = root/'source.h5'
    write_case(path,target*.95,target.mean(0),target,np.ones_like(target),meta,roi_labels=roi)
    projections = root/'views.h5'
    angles = np.deg2rad(np.arange(0,180,30,dtype=np.float32))
    measured = np.empty((6,*geo.nDetector),dtype=np.float32)
    for index in (0,1):
        measured[index::2] = project(target[index],geo,angles[index::2])
    with h5py.File(projections,'w') as out:
        out['angles_rad'] = angles
        out['state_label'] = labels*3
        out['mixed_line_integral'] = measured
    row = {'case_id':'scan','animal_id':'synthetic-1','path':str(path),'sha256':sha256(path),
           'projections':str(projections),'projections_sha256':sha256(projections),**validate_case(path)}
    declared = {(label,name):{'available':True,'whole_fov_confirmed':True,'closed':True}
                for label in labels for name in STRUCTURES}
    row.update(prepare_annotations(row,{},declared,root/'reference-contours.h5'))
    source = root/'source.jsonl'
    write_manifest(source,[row])
    layouts = angle_layouts(source,'scan',2,[0,2],[2,4],root/'layouts')
    joint = Path(layouts['joint-correct'])
    records = validate_manifest(joint)
    validation = fixture(root,t=2,animal='synthetic-2',case='val',shape=shape)
    with h5py.File(root/'val-views.h5','w') as val:
        val['angles_rad'] = angles
        val['state_label'] = [1,2]*3
    validation.update(projections=str(root/'val-views.h5'),projections_sha256=sha256(root/'val-views.h5'))
    manifest = root/'train-val.jsonl'
    write_manifest(manifest,[*records,validation])
    split = {'train':['synthetic-1'],'val':['synthetic-2'],'test':[]}
    cfg = validate_train({'device':'cpu','threads':2,'amp':False,'log_every':1,'stage1_patch':[1,16,16],
        'stage2_patches':[[6,16,16]],'epochs':{'1':1,'2':1},'steps_per_epoch':{'1':2,'2':6},
        'loss':{'kind':'l1'},'stage2_crop_mode':'core_with_context',
        'extensions':{'prior':{'mode':'geometry'},'boundary':{'enabled':True}}})
    stage1 = train(cfg,manifest,split,root/'s1',1)
    stage2 = train(cfg,manifest,split,root/'s2',2,init={'path':stage1['checkpoint'],'source_stage':1})
    pred = predict_manifest(stage2['checkpoint'],joint,root/'predictions',split_name='train',
                            device='cpu',amp=False,mode='full')
    metrics = metric_defaults()
    metrics['contours']['surface_dice_tolerance_vox'] = {s:1. for s in STRUCTURES}
    evaluate_predictions(pred,joint,metrics,root/'evaluation',contours=contours(pred))
    return records,layouts,cfg


class LayoutMetricTests(unittest.TestCase):
    def test_contour_states_and_animal_mean_with_partial_layout_coverage(self):
        from rstar4d.evaluation.contours import summarize_contours
        def row(phase,layout,dice,hd,status):
            return {'animal_id':'pig','case_id':'scan','sample_id':layout,'phase_label':phase,'layout':layout,
                    'structure':'lv','surface_dice':dice,'hd95':hd,'assd':hd,'band_mae':None,
                    'reference_volume_voxels':None,'status':status}
        observations = [row(2,'A',1.,0.,'ok'),row(2,'B',1.,0.,'ok'),row(9,'A',0.,None,'empty_prediction'),
                        row(9,'B',None,None,'missing_prediction')]
        animals,_ = summarize_contours(observations)
        summary = animals[0]
        self.assertEqual((summary['reference_states'],summary['surface_dice_states'],summary['hd95_states']),(2,2,1))
        self.assertEqual((summary['reference_observations'],summary['surface_dice_observations'],
                          summary['hd95_observations']),(4,3,2))
        self.assertEqual((summary['empty_predictions'],summary['missing_predictions']),(1,1))
        self.assertAlmostEqual(summary['surface_dice'],.5)

    def test_projection_counts_and_means_use_original_cases(self):
        from rstar4d.evaluation.report import projection_summary
        def item(case,layout,value):
            p = {'status':'not_applicable','reason':'unknown geometry'} if value is None else {
                'status':'ok','mae':value,'rmse':value,'relative_l2':value,'ncc':value,'constant_views':0}
            return {'case_id':case,'animal_id':'pig','sample_id':case+layout,'projection':p}
        rows = [item('x','A',1.),item('x','B',1.),item('y','A',0.),item('y','B',None),
                item('z','A',None),item('z','B',None)]
        p = projection_summary(rows)
        self.assertEqual((p['applicable_cases'],p['skipped_cases'],p['applicable_samples'],p['skipped_samples']),(2,1,3,3))
        self.assertAlmostEqual(p['metrics']['mae']['animal_equal']['mean'],.5)
        self.assertEqual(p['metrics']['mae']['case_equal']['n'],2)

    def test_global_image_summary_keeps_phase_denominators_when_layouts_lack_a_phase(self):
        from rstar4d.evaluation.report import image_summary
        def obs(phase,error,n):
            return {'animal_id':'pig','case_id':'scan','phase_label':phase,'mae':error,'rmse':error,
                    'ssim':error,'valid_voxels':n,'ssim_centers':n}
        rows = [obs(2,1.,16),obs(2,1.,16),obs(9,0.,4),obs(9,None,0)]
        result = image_summary(rows)
        for key,expected in [('mae',.8),('ssim',.8),('rmse',np.sqrt(.8))]:
            self.assertAlmostEqual(result[key]['animal_equal']['mean'],expected)
            self.assertEqual(result[key]['case_equal']['n'],1)
            self.assertEqual((result[key]['valid_states'],result[key]['valid_state_layout_pairs']),(2,3))

    def test_one_original_case_cannot_enter_two_animal_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = fixture(root,t=2)
            path = root/'manifest.jsonl'
            write_manifest(path,[{**row,'sample_id':'layout-A'},
                                 {**row,'sample_id':'layout-B','animal_id':'other'}])
            with self.assertRaisesRegex(ValueError,'different animals'):
                validate_manifest(path,{'train':['a'],'val':['other'],'test':[]})

    def test_export_can_reference_files_on_another_windows_volume(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory(dir=Path.cwd()) as dst:
            row = fixture(Path(src),t=2)
            manifest = Path(dst)/'manifest.jsonl'
            write_manifest(manifest,[row])
            self.assertEqual(Path(validate_manifest(manifest)[0]['path']),Path(row['path']))
            if Path(src).drive!=manifest.drive:
                self.assertTrue(Path(read_jsonl(manifest)[0]['path']).is_absolute())

    def test_original_cases_keep_their_sampling_weight_and_epoch_budget(self):
        from rstar4d.data import sampling
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = fixture(root,t=2,case='a')
            b = fixture(root,t=2,case='b')
            c = fixture(root,t=2,animal='other',case='c')
            rows = [a,b,c,{**a,'sample_id':'a-second'}]
            path = root/'joint.jsonl'
            write_manifest(path,rows)
            parsed = read_manifest(path)
            self.assertEqual([r['sample_id'] for r in parsed],['a','b','c','a-second'])
            cfg = validate_train({'stage2_patches':[[4,16,16]]})
            dataset = PatchDataset(parsed,cfg,2,1,5000)
            counts = Counter()
            rng = np.random.default_rng(71)
            for i in range(len(dataset)):
                rec = dataset.records[dataset.proposal(i,rng)[0]]
                counts[rec['sample_id']] += 1
            for sample,expected in [('a',.125),('a-second',.125),('b',.25),('c',.5)]:
                self.assertAlmostEqual(counts[sample]/len(dataset),expected,delta=.025)
            for stage,expected in [(1,12),(2,45)]:
                self.assertEqual(sampling.epoch_draw_budget(parsed,cfg,stage),expected)
            specs = validation_samples(parsed,cfg,2)
            replay = PatchDataset(parsed,cfg,2,0,len(specs),specs)
            self.assertEqual({replay[i]['sample_id'] for i in range(len(replay))},{'a','a-second','b','c'})
            write_manifest(path,[a,a])
            with self.assertRaisesRegex(ValueError,'Duplicate sample_id'):
                read_manifest(path)

    def test_phase_denominators_and_intersected_states(self):
        target = np.zeros((2,1,12,12))
        valid = np.ones_like(target,dtype=bool)
        valid[1,:,1:] = False
        pred = np.ones_like(target)
        pred[1] = 3
        metrics = image_metrics(pred,target,valid,phase_labels=[2,9])
        self.assertAlmostEqual(metrics['mae'],180/156)
        self.assertEqual([(r['phase_label'],r['valid_voxels'],r['ssim_centers'])
                          for r in metrics['by_phase']],[(2,144,4),(9,12,0)])
        self.assertIsNone(metrics['by_phase'][1]['ssim'])
        self.assertIsNone(metrics['by_phase'][1]['mae_reason'])
        self.assertEqual(metrics['by_phase'][1]['ssim_reason'],'no complete SSIM window')
        wrap = lambda r: {'global':{'prediction':r}}
        case = {'case_id':'scan','animal_id':'pig','paired_sha256':'shared','regions':wrap(metrics),
                'phases':[{'phase_label':r['phase_label'],'regions':wrap(r)} for r in metrics['by_phase']]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def save(name,row):
                folder = root/name/'evaluation'
                folder.mkdir(parents=True)
                atomic_json(folder/'summary.json',{})
                atomic_jsonl(folder/'cases.jsonl',[row])
            save('base',case)
            save('both',case)
            save('one',{**case,'phases':case['phases'][1:]})
            save('old',{k:v for k,v in case.items() if k!='phases'})
            path = compare(root,root/'report',[('both','base'),('one','base'),('old','base')])
            data = read_json(path)
            rows = {(r['candidate'],r['metric']):r for r in data['paired'] if r['structure']=='global'}
            for pair,states in [(('both','mae'),2),(('both','ssim'),1),(('one','mae'),1),
                                (('one','ssim'),0),(('old','mae'),None)]:
                self.assertEqual(rows[pair]['valid_states'],states)
                self.assertEqual((rows[pair]['valid_animals'],rows[pair]['valid_cases']),(1,1))
            phase = [r for r in data['paired_by_phase'] if r['candidate']=='one' and r['metric']=='mae']
            self.assertEqual([(r['phase_label'],r['delta']) for r in phase],[(9,0.)])

    def test_case_equal_global_summary_counts_original_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = fixture(root,t=2,case='a')
            b = fixture(root,t=2,case='b')
            records = [{**a,'sample_id':'a-A','layout':'A'},{**a,'sample_id':'a-B','layout':'B'},b]
            manifest = root/'source.jsonl'
            write_manifest(manifest,records)
            predictions = []
            for row,error in zip(validate_manifest(manifest),[0.,.004,.008]):
                path = root/(row['sample_id']+'-predict.h5')
                with h5py.File(path,'w') as out,h5py.File(row['path'],'r') as source:
                    out['prediction'] = source['target'][:]+error
                predictions.append({'sample_id':row['sample_id'],'case_id':row['case_id'],'animal_id':row['animal_id'],
                                    'source_metadata_hash':canonical_hash(row['metadata']),'path':str(path),
                                    'sha256':sha256(path),'source_sha256':row['sha256']})
            path = root/'predictions.jsonl'
            atomic_jsonl(path,predictions)
            summary = evaluate_predictions(path,manifest,metric_defaults(),root/'report')
            mean = summary['metrics']['global/prediction/mu']['mae']['case_equal']
            self.assertEqual((summary['cases'],summary['samples'],mean['n']),(2,3,2))
            self.assertAlmostEqual(mean['mean'],.005)

    @unittest.skipUnless(torch.cuda.is_available(),'Synthetic projection/FDK requires CUDA')
    def test_joint_fdk_train_infer_and_independent_contour_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records,layouts,cfg = run_joint_phantom(root)
            self.assertEqual({r['layout'] for r in records},{'A','B'})
            samples = {r['sample_id'] for r in records}
            self.assertEqual(len(samples),2)
            from rstar4d.training.cardiac_benchmark import benchmark_manifest
            measured = benchmark_manifest(cfg,layouts['joint-correct'],None,1)['results'][0]
            self.assertEqual(measured['status'],'ok')
            self.assertIn(measured['sample_id'],samples)
            draws = read_jsonl(root/'s2/cardiac_samples.jsonl')
            self.assertEqual({r['sample_id'] for r in draws},samples)
            fixed = read_json(root/'s2/validation_samples.json')
            self.assertEqual({r['case_id'] for r in fixed},{'val'})
            predictions = read_jsonl(root/'predictions/manifest.jsonl')
            self.assertEqual(len({p['path'] for p in predictions}),2)
            self.assertEqual({tuple(p['geometry_source']['encoded_indices']) for p in predictions},{(0,2),(2,4)})
            cases = read_jsonl(root/'evaluation/cases.jsonl')
            self.assertEqual({c['case_id'] for c in cases},{'scan'})
            self.assertEqual(len({c['cardiac_layout']['reference_paired_sha256'] for c in cases}),1)
            self.assertEqual({r['sample_id'] for r in read_jsonl(root/'evaluation/contours_by_phase.jsonl')},samples)
            invalid = {**records[0],'angle_summary':records[1]['angle_summary']}
            with self.assertRaisesRegex(ValueError,'paired volume'):
                load_geometry(invalid,[2,9],allow_diagnostic_swap=False)
            with self.assertRaisesRegex(ValueError,'inference diagnostics'):
                PatchDataset(validate_manifest(layouts['A-swapped']),cfg,2,1,1)

    def test_joint_reports_pair_within_layout_unless_an_explicit_contrast_is_requested(self):
        def sample(layout,value):
            metrics = {'global':{'prediction':{'mae':value,'ssim':None}}}
            return {'case_id':'scan','animal_id':'pig','sample_id':'scan-'+layout,'layout':layout,
                    'paired_sha256':layout,'regions':metrics,'phases':[{'phase_label':9,'regions':metrics}],
                    'cardiac_layout':{'layout':layout,'phase':9,'reference_paired_sha256':'parent',
                                     'source_projections_sha256':'views','reference_annotations_sha256':None}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name,rows in [('base',[sample('A',1.),sample('B',5.)]),('new',[sample('B',7.),sample('A',4.)])]:
                folder = root/name/'evaluation'
                folder.mkdir(parents=True)
                atomic_json(folder/'summary.json',{})
                atomic_jsonl(folder/'cases.jsonl',rows)
            def differences(**options):
                out = compare(root,root/'report',[('new','base')],**options)
                data = read_json(out)
                return [r for r in data['paired_by_phase'] if r['metric']=='mae']
            default = differences()
            self.assertEqual({(r['candidate_layout'],r['reference_layout'],r['delta']) for r in default},
                             {('A','A',3.),('B','B',2.)})
            cross = differences(layout_pair=('B','A'))
            self.assertEqual([(r['candidate_layout'],r['reference_layout'],r['delta']) for r in cross],[('B','A',6.)])
            broken = sample('B',7.)
            broken['cardiac_layout']['reference_paired_sha256']='another'
            atomic_jsonl(root/'new/evaluation/cases.jsonl',[broken])
            with self.assertRaisesRegex(ValueError,'reference image'):
                differences(layout_pair=('B','A'))

    def test_two_layouts_are_one_anatomical_state_and_have_one_state_weight(self):
        def row(layout,phase,value):
            return {'case_id':'scan','animal_id':'pig','layout':layout,'phase_label':phase,
                    'structure':'lv','surface_dice':value}
        old = [row('A',9,0.),row('B',9,0.),row('A',2,0.)]
        new = [row('A',9,.2),row('B',9,.4),row('A',2,1.)]
        matched = paired_differences(old[:2],new[:2],'surface_dice','lv')
        self.assertEqual(matched['valid_states'],1)
        self.assertEqual(matched['valid_state_layout_pairs'],2)
        all_phases = paired_differences(old,new,'surface_dice','lv')
        self.assertEqual(all_phases['valid_states'],2)
        self.assertAlmostEqual(all_phases['mean_delta'],.65)
