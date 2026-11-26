from __future__ import annotations
from collections import defaultdict
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from .contracts import validate_case


class PatchDataset(Dataset):
    def __init__(self,records,config,stage,epoch,length,samples=None):
        records=sorted(records,key=lambda r:r['case_id'])
        self.records=records; self.config=config; self.stage=stage; self.epoch=epoch; self.length=length; self.samples=samples
        self.shapes={}; self.meta={}; self.animals=defaultdict(list)
        self.by_id={r['case_id']:i for i,r in enumerate(records)}
        if not records: raise ValueError('No cases in dataset')
        self.patches=[config['stage1_patch']] if stage==1 else config['stage2_patches']
        for i,r in enumerate(records):
            info=validate_case(r['path'],full=False); self.shapes[i]=info['shape']; self.meta[i]=info['metadata']
            if config['temporal_roll'] and self.meta[i]['temporal_topology']!='cyclic': raise ValueError('Phase roll requires cyclic data')
            if config['small_volume']=='error' and any(any(a<b for a,b in zip(info['shape'][1:],p)) for p in self.patches):
                raise ValueError(f'{r["case_id"]} smaller than configured patch; explicitly use small_volume=pad')
            if info['metadata']['normalization']!=config['normalization']: raise ValueError('Data/training normalization mismatch')
            self.animals[str(r['animal_id'])].append(i)
        self.animal_ids=sorted(self.animals)

    def __len__(self): return len(self.samples) if self.samples is not None else self.length

    def proposal(self,index,rng):
        if self.samples is not None:
            s=self.samples[index]; return self.by_id[s['case_id']],s['shape'],s['origin']
        animal=self.animal_ids[int(rng.integers(len(self.animal_ids)))]; choices=self.animals[animal]
        i=choices[int(rng.integers(len(choices)))]; shape=self.patches[int(rng.integers(len(self.patches)))]
        origin=[int(rng.integers(max(1,n-k+1))) for n,k in zip(self.shapes[i][1:],shape)]
        return i,shape,origin

    def __getitem__(self,index):
        if index<0 or index>=len(self): raise IndexError(index)
        rng=np.random.default_rng(np.random.SeedSequence([self.config['seed'],self.epoch,index,self.stage]))
        i,shape,origin=self.proposal(index,rng); r=self.records[i]
        for attempt in range(1 if self.samples is not None else self.config['max_attempts']):
            if attempt:
                origin=[int(rng.integers(max(1,n-k+1))) for n,k in zip(self.shapes[i][1:],shape)]
            slices=tuple(slice(a,min(a+k,n)) for a,k,n in zip(origin,shape,self.shapes[i][1:]))
            with h5py.File(r['path'],'r') as f:
                mask=np.asarray(f['valid_mask'][(slice(None),)+slices],dtype=bool)
                pads=[(0,0)]+[(0,k-n) for k,n in zip(shape,mask.shape[1:])]
                mask=np.pad(mask,pads,mode='constant')
                if float(mask.mean())<self.config['min_valid_fraction']: continue
                cbct=np.asarray(f['cbct'][(slice(None),)+slices],dtype=np.float32)
                target=np.asarray(f['target'][(slice(None),)+slices],dtype=np.float32)
                prior=np.asarray(f['prior'][slices],dtype=np.float32)
                cbct=np.pad(cbct,pads,mode='edge'); target=np.pad(target,pads,mode='edge')
                prior=np.pad(prior,pads[1:],mode='edge')
            if self.config['temporal_roll'] and self.samples is None:
                shift=int(rng.integers(cbct.shape[0])); cbct=np.roll(cbct,shift,0); target=np.roll(target,shift,0); mask=np.roll(mask,shift,0)
            scale=self.config['normalization']['scale']; offset=self.config['normalization']['offset']
            x=np.stack([(cbct+offset)/scale,np.broadcast_to((prior+offset)/scale,cbct.shape)])
            return {'input':torch.from_numpy(x.copy()),'target':torch.from_numpy(((target+offset)/scale)[None].copy()),
                    'valid':torch.from_numpy(mask[None].copy()),'case_id':r['case_id'],'animal_id':str(r['animal_id']),
                    'temporal_topology':self.meta[i]['temporal_topology'],
                    'origin':torch.tensor(origin),'patch_shape':torch.tensor(shape)}
        raise ValueError(f'{r["case_id"]}: no patch {list(shape)} with valid fraction >= {self.config["min_valid_fraction"]} after {self.config["max_attempts"]} attempts (draw {index}); animal/case/shape were not resampled')


def validation_samples(records,config,stage):
    result=[]
    for r in sorted(records,key=lambda r:r['case_id']):
        # Seed is based on sorted case position, never Python's salted hash.
        case_index=sorted(x['case_id'] for x in records).index(r['case_id'])
        subset=PatchDataset([r],config,stage,0,config['validation_samples_per_case'])
        for draw in range(config['validation_samples_per_case']):
            rng=np.random.default_rng(np.random.SeedSequence([config['seed'],stage,case_index,draw,104729]))
            i,shape,origin=subset.proposal(0,rng)
            for attempt in range(config['max_attempts']):
                if attempt: origin=[int(rng.integers(max(1,n-k+1))) for n,k in zip(subset.shapes[i][1:],shape)]
                spec={'case_id':r['case_id'],'shape':list(shape),'origin':list(origin)}
                try:
                    fixed=PatchDataset([r],config,stage,0,1,[spec]); fixed[0]
                    result.append(spec); break
                except ValueError: continue
            else: raise ValueError(f'No valid fixed validation patch for {r["case_id"]}')
    return result
