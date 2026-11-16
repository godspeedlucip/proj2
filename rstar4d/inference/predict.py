from __future__ import annotations
import json
import os
from pathlib import Path
import h5py
import numpy as np
import nibabel as nib
import tifffile
from PIL import Image, ImageDraw
from ..training.checkpoint import load_checkpoint
from ..training.trainer import select_device
from ..models import create_model
from ..data.contracts import validate_manifest, validate_split
from ..utils.io import atomic_json, atomic_jsonl, sha256, canonical_hash
from .sliding_window import predict_volume


def preview(path,cbct,prediction,target,window=(0.,.05)):
    t=prediction.shape[0]//2; z=prediction.shape[1]//2; lo,hi=window
    arrays=[np.asarray(cbct[t,z]),np.asarray(prediction[t,z])]
    labels=['Sparse CBCT','RSTAR4D']
    if target is not None: arrays.append(np.asarray(target[t,z])); labels.append('Reference GT')
    h,w=arrays[0].shape; canvas=Image.new('RGB',(w*len(arrays),h+24),'white'); draw=ImageDraw.Draw(canvas)
    for i,(a,label) in enumerate(zip(arrays,labels)):
        gray=np.uint8(np.clip((a-lo)/(hi-lo),0,1)*255)
        canvas.paste(Image.fromarray(gray).convert('RGB'),(i*w,24)); draw.text((i*w+3,4),label,fill='black')
    canvas.save(path)


def predict_manifest(checkpoint,manifest,out,split_name='test',mode='tiled',tile=(32,128,128),overlap=(16,64,64),device='cuda',amp=True,export=('h5',),split=None):
    ck=load_checkpoint(checkpoint,require_training=False); dev=select_device(device)
    if not all(k in ck for k in ['stage','model_profile','normalization','intensity_domain']): raise ValueError('Checkpoint lacks inference contracts')
    model=create_model(ck['stage'],ck['model_profile']).to(dev); model.load_state_dict(ck['state_dict'],strict=True)
    rows=validate_manifest(manifest,require_target=False)
    if split_name!='all':
        selected_split=validate_split(rows,split or ck.get('split',{}))
        rows=[r for r in rows if str(r['animal_id']) in selected_split[split_name]]
    if not rows: raise ValueError('No cases in requested inference split')
    out=Path(out).resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError('Prediction output must be empty')
    out.mkdir(parents=True,exist_ok=True); records=[]; checkpoint_sha=sha256(checkpoint)
    for row in rows:
        meta=row['metadata']
        if meta['normalization']!=ck['normalization'] or meta['intensity_domain']!=ck['intensity_domain']:
            raise ValueError('Checkpoint/data intensity or normalization mismatch')
        if meta['calibration_application']!=ck.get('calibration_application',meta['calibration_application']): raise ValueError('Calibration application mismatch')
        folder=out/row['case_id']; folder.mkdir()
        path=folder/'prediction.h5'; temp=folder/'prediction.partial'
        with h5py.File(row['path'],'r') as f:
            predicted=predict_volume(model,f['cbct'],f['prior'],ck['normalization'],mode,tile,overlap,dev,amp)
            output_meta={**meta,'inference':{'mode':mode,'tile':list(tile),'overlap':list(overlap),'stage':ck['stage'],
                'model_profile':ck['model_profile'],'checkpoint_sha256':checkpoint_sha,'source_sha256':row['sha256']}}
            with h5py.File(temp,'w') as dest:
                dest.create_dataset('prediction',data=predicted,chunks=(1,min(predicted.shape[1],16),min(predicted.shape[2],128),min(predicted.shape[3],128)),compression='gzip',compression_opts=1)
                dest.attrs['metadata_json']=json.dumps(output_meta,ensure_ascii=False)
            os.replace(temp,path)
            preview(folder/'comparison.png',f['cbct'],predicted,f.get('target'))
            for ti,label in enumerate(meta['phase_labels']):
                if 'tif' in export: tifffile.imwrite(folder/f'phase_{label:02d}_prediction.tif',predicted[ti])
                if 'nii' in export:
                    affine=np.asarray(meta.get('geometry',{}).get('voxel_to_scanner_zyx',np.diag([*meta['output_spacing_zyx_mm'],1.])))
                    nib.save(nib.Nifti1Image(predicted[ti],affine),str(folder/f'phase_{label:02d}_prediction.nii.gz'))
            output_meta['nifti_affine_semantics']='array ZYX to scanner ZYX; not patient LPS/RAS'
            atomic_json(folder/'metadata.json',output_meta)
        record={'case_id':row['case_id'],'animal_id':row['animal_id'],'path':path.relative_to(out).as_posix(),
            'sha256':sha256(path),'source_sha256':row['sha256'],'checkpoint_sha256':checkpoint_sha,
            'source_metadata_hash':canonical_hash(meta),'mode':mode}
        records.append(record); atomic_jsonl(out/'manifest.jsonl',records)
        print(json.dumps({'case_id':row['case_id'],'prediction':str(path)},ensure_ascii=False),flush=True)
    return out/'manifest.jsonl'
