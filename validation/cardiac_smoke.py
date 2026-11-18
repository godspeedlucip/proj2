"""Exercise all eight Stage II arms with independent thresholded phantom contours."""
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from experiments.cardiac_ablation import compare, matrix
from rstar4d.config import metric_defaults, validate_train
from rstar4d.data.cardiac_annotations import prepare_annotations, STRUCTURES
from rstar4d.data.contracts import basic_metadata, validate_case, write_case, write_manifest
from rstar4d.evaluation.report import evaluate_predictions
from rstar4d.inference.predict import predict_manifest
from rstar4d.inference.sliding_window import predict_volume, starts
from rstar4d.models import model_from_checkpoint
from rstar4d.training.checkpoint import load_checkpoint
from rstar4d.training.trainer import train
from rstar4d.utils.io import atomic_json, atomic_jsonl, read_jsonl, resolve, sha256, read_json


def fixture(root):
    shape = (8,24,24)
    affine = np.eye(4)
    meta = basic_metadata(2,(1.,1.,1.),phase_labels=[2,9],geometry={
        'shape_zyx':list(shape),'voxel_spacing_zyx_mm':[1.]*3,
        'voxel_to_scanner_zyx':affine.tolist(),'center_zyx_mm':((np.array(shape)-1)/2).tolist()})
    z,y,x = np.indices(shape).astype(np.float32)
    rng = np.random.default_rng(28)
    rows = []
    for index,animal in enumerate(('train','val','test')):
        phases = []
        for j in range(2):
            shift = j + index*.25
            heart = ((z-3.5)/2.7)**2+((y-12+shift)/7)**2+((x-12)/8)**2 < 1
            lv = ((z-3.5)/1.9)**2+((y-12+shift)/4)**2+((x-13)/3)**2 < 1
            phases.append(heart.astype(np.int32)+lv.astype(np.int32))
        roi = np.stack(phases)
        clean = (.003 + .016*(roi>0) + .019*(roi==2) + .001*np.sin(y/3)*np.cos(x/4)).astype(np.float32)
        sparse = clean + (rng.normal(size=clean.shape)*.002).astype(np.float32)
        path = root/(animal+'.h5')
        write_case(path,sparse,clean.mean(axis=0),clean,np.ones_like(clean),meta,roi_labels=roi)
        projections = root/(animal+'-angles.h5')
        with h5py.File(projections,'w') as out:
            out['angles_rad'] = np.deg2rad([5.,25.,50.,80.,120.,150.])
            out['state_label'] = [2,9,2,9,2,9]
        record = {'case_id':animal,'animal_id':animal,'path':str(path),'sha256':sha256(path),
                  'projections':str(projections),'projections_sha256':sha256(projections),**validate_case(path)}
        declared = {(label,structure):{'available':True,'whole_fov_confirmed':True,'closed':True}
                    for label in meta['phase_labels'] for structure in STRUCTURES}
        record.update(prepare_annotations(record,{},declared,root/(animal+'-annotations.h5')))
        rows.append(record)
    path = root/'manifest.jsonl'
    write_manifest(path,rows)
    return path


def contours(predictions):
    folder = predictions.parent
    rows = []
    thresholds = {'heart_outer':.014,'lv':.027}
    protocol = {'method':'frozen_segmenter','version':'phantom_absolute_threshold_v1',
                'weights_sha256':sha256(Path(__file__)),'threshold':thresholds}
    for row in read_jsonl(predictions):
        source_path = resolve(row['path'],folder)
        output = folder/(row['case_id']+'-contours.h5')
        with h5py.File(source_path,'r') as source, h5py.File(output,'x') as dest:
            for phase,label in enumerate((2,9)):
                for structure in STRUCTURES:
                    key = f'{label}/{structure}'
                    dest[key] = (source['prediction'][phase]>=thresholds[structure]).astype(np.uint8)
                    rows.append({'case_id':row['case_id'],'phase_label':label,'structure':structure,
                                 'prediction_sha256':row['sha256'],'dataset':key,'path':output.name,
                                 'delineation':protocol})
        for item in rows:
            if item['case_id']==row['case_id']:
                item['sha256'] = sha256(output)
    path = folder/'contours.jsonl'
    atomic_jsonl(path,rows)
    return path


def stitch_error(checkpoint):
    saved = load_checkpoint(checkpoint)
    model = model_from_checkpoint(saved)
    options = saved['stage2_resolved']
    rng = np.random.default_rng(779)
    source = rng.uniform(.002,.043,size=(2,8,24,112)).astype(np.float32)
    prior = source.mean(axis=0)
    context = {'source_fov':np.ones_like(source,dtype=bool),'phase_labels':[2,9],
               'geometry':np.ones((2,14),dtype=np.float32)*.1}
    tile = (8,24,32)
    overlap = (0,0,16)
    kwargs = {'amp':False,'recipe':options,'tile':tile,'overlap':overlap,'context':context}
    full = predict_volume(model,source,prior,saved['normalization'],mode='full',**kwargs)
    tiled = predict_volume(model,source,prior,saved['normalization'],mode='tiled',**kwargs)
    visits = np.zeros(source.shape[-1],dtype=int)
    for origin in starts(len(visits),tile[-1],overlap[-1],8):
        visits[origin:origin+tile[-1]] += 1
    error = np.abs(full-tiled)
    return {name:{'mae':float(error[...,mask].mean()),'max_abs':float(error[...,mask].max()),
                  'voxels':int(error[...,mask].size)}
            for name,mask in [('tile_interior',visits==1),('overlap_seams',visits>1)]}


def run(root):
    root.mkdir(parents=True)
    manifest = fixture(root)
    split = {'train':['train'],'val':['val'],'test':['test']}
    atomic_json(root/'split.json',split)
    base = validate_train({'device':'cpu','amp':False,'threads':2,'log_every':1,
        'stage1_patch':[1,16,16],'stage2_patches':[[6,16,16]],'stage2_crop_mode':'core_with_context',
        'epochs':{'1':1,'2':2},'steps_per_epoch':{'1':1,'2':1},'loss':{'kind':'l1'},
        'stage2_context':{'lncc_radius_vox':[1,1,1]},
        'extensions':{'boundary':{'ramp_epochs':1}}})
    stage1 = train(base,manifest,split,root/'shared-stage1',1)['checkpoint']
    cfg = metric_defaults()
    cfg['contours']['surface_dice_tolerance_vox'] = {'heart_outer':1.,'lv':1.}
    smoke = {}
    for name,options in matrix(base).items():
        out = root/name
        first = train(options,manifest,split,out,2,init={'path':stage1,'source_stage':1},stop_after_epoch=1)
        run = train(options,manifest,split,out,2,resume=first['last'])
        pred = predict_manifest(run['checkpoint'],manifest,out/'predictions',device='cpu',amp=False,split=split)
        evaluate_predictions(pred,manifest,cfg,out/'evaluation',contours=contours(pred))
        epochs = [row for row in read_jsonl(out/'train.jsonl') if row['event']=='epoch']
        smoke[name] = [row['sampling_sha256'] for row in epochs]
    if any(value!=smoke['baseline'] for value in smoke.values()):
        raise AssertionError('The eight arms used different training samples')
    report = compare(root,root/'report')
    if read_json(report)['missing_groups']:
        raise AssertionError('A smoke arm is missing')
    atomic_json(root/'smoke.json',{'status':'complete','arms':smoke,'shared_stage1_sha256':sha256(stage1)})
    atomic_json(root/'stitch_error.json',stitch_error(root/'MPB/checkpoints/best.pt'))
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',required=True,type=Path)
    print(run(parser.parse_args().out))
