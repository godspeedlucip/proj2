from __future__ import annotations
from collections import defaultdict
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from .contracts import sample_id, validate_case
from .acquisition import load_geometry
from .context import plan_window, check_support
from ..config import resolve_stage2, validate_train


class InvalidCore(ValueError):
    """The deterministic proposal may be retried with another origin."""


def epoch_draw_budget(records, config, stage):
    explicit = config['steps_per_epoch'][str(stage)]
    if explicit:
        return explicit
    depths = {}
    for record in records:
        case = (str(record['animal_id']),record['case_id'])
        if stage==1:
            shape = record.get('shape') or validate_case(record['path'],full=False)['shape']
            if case in depths and depths[case]!=shape[1]:
                raise ValueError(f'{case}: layouts require the same reference depth')
            depths[case] = shape[1]
        else:
            depths[case] = 15
    return sum(depths.values())


class PatchDataset(Dataset):
    def __init__(self,records,config,stage,epoch,length,samples=None):
        records=sorted(records,key=sample_id)
        self.records=records; self.config=config; self.stage=stage; self.epoch=epoch; self.length=length; self.samples=samples
        self.shapes={}; self.meta={}; self.animals=defaultdict(dict)
        self.by_id={sample_id(r):i for i,r in enumerate(records)}
        if len(self.by_id)!=len(records): raise ValueError('Duplicate sample_id')
        if not records: raise ValueError('No cases in dataset')
        self.patches=[config['stage1_patch']] if stage==1 else config['stage2_patches']
        self.resolved=resolve_stage2(config if 'extensions' in config else validate_train(config),stage)
        self.geometry={}
        self.use_annotations=(self.resolved['boundary']['enabled'] or self.resolved['motion']['surface_weight'] > 0)
        self.contextual=(self.resolved['crop_mode'] == 'core_with_context'
                         or self.resolved['motion']['mode'] != 'off' or self.resolved['prior']['mode'] != 'off'
                         or self.use_annotations)
        for i,r in enumerate(records):
            if r.get('diagnostic_swap'):
                raise ValueError('Swapped angle summaries are reserved for inference diagnostics')
            info=validate_case(r['path'],full=False); self.shapes[i]=info['shape']; self.meta[i]=info['metadata']
            if config['temporal_roll'] and self.meta[i]['temporal_topology']!='cyclic': raise ValueError('Phase roll requires cyclic data')
            if config['small_volume']=='error' and any(any(a<b for a,b in zip(info['shape'][1:],p)) for p in self.patches):
                raise ValueError(f'{r["case_id"]} smaller than configured patch; explicitly use small_volume=pad')
            if info['metadata']['normalization']!=config['normalization']: raise ValueError('Data/training normalization mismatch')
            if self.resolved['prior']['mode']=='geometry':
                self.geometry[i]=load_geometry(r,self.meta[i]['phase_labels'],allow_diagnostic_swap=False)
            self.animals[str(r['animal_id'])].setdefault(r['case_id'],[]).append(i)
        self.animal_ids=sorted(self.animals)
        self.case_ids={animal:sorted(cases) for animal,cases in self.animals.items()}

    def __len__(self): return len(self.samples) if self.samples is not None else self.length

    def proposal(self,index,rng):
        if self.samples is not None:
            s=self.samples[index]; return self.by_id[sample_id(s)],s['shape'],s['origin']
        animal=self.animal_ids[int(rng.integers(len(self.animal_ids)))]
        cases=self.case_ids[animal]
        case=cases[int(rng.integers(len(cases)))]
        layouts=self.animals[animal][case]
        i=layouts[int(rng.integers(len(layouts)))] if len(layouts)>1 else layouts[0]
        shape=self.patches[int(rng.integers(len(self.patches)))]
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
                if self.contextual:
                    return self._context_patch(i,shape,origin,r,f,rng)
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
                    'valid':torch.from_numpy(mask[None].copy()),'case_id':r['case_id'],'sample_id':sample_id(r),'animal_id':str(r['animal_id']),
                    'origin':torch.tensor(origin),'patch_shape':torch.tensor(shape)}
        raise InvalidCore(f'{r["case_id"]}: no patch {list(shape)} with valid fraction >= {self.config["min_valid_fraction"]} after {self.config["max_attempts"]} attempts (draw {index}); animal/case/shape were not resampled')

    def _context_patch(self,i,shape,origin,record,paired,rng):
        recipe = self.resolved['recipe']
        policy = recipe if self.resolved['crop_mode']=='core_with_context' else {
            'max_displacement_vox':[0]*3,'lncc_radius_vox':[0]*3,'support_margin_vox':[0]*3,'alignment':[1]*3}
        window = plan_window(self.shapes[i][1:],origin,shape,policy)
        if (self.resolved['crop_mode']=='core_with_context' or self.resolved['motion']['struct_weight']
                or self.resolved['motion']['surface_weight']):
            check_support(window,recipe,paired['valid_mask'],record['case_id'],prior_fov=paired.get('prior_valid_mask'))
        slices = window.slices
        pads = [(0,max(0,core.stop-(stop-start))) for core,start,stop in zip(window.core,window.origin,window.stop)]
        valid = np.asarray(paired['valid_mask'][(slice(None),*slices)],dtype=bool)
        prior_valid = np.asarray(paired['prior_valid_mask'][slices],dtype=bool) if 'prior_valid_mask' in paired else valid.all(axis=0)
        cbct = np.pad(np.asarray(paired['cbct'][(slice(None),*slices)],np.float32),[(0,0),*pads],mode='edge')
        target = np.pad(np.asarray(paired['target'][(slice(None),*slices)],np.float32),[(0,0),*pads],mode='edge')
        prior = np.pad(np.asarray(paired['prior'][slices],np.float32),pads,mode='edge')
        valid = np.pad(valid,[(0,0),*pads],mode='constant')
        phase_indices = np.arange(len(cbct))
        if self.config['temporal_roll'] and self.samples is None:
            phase_indices = np.roll(phase_indices,int(rng.integers(len(cbct))))
        phase_labels = np.asarray(self.meta[i]['phase_labels'])[phase_indices]
        ctx = {'phase_labels':torch.from_numpy(phase_labels), 'phase_indices':torch.from_numpy(phase_indices),
               'source_fov':torch.from_numpy(valid[phase_indices]),
               'prior_fov':torch.from_numpy(np.pad(prior_valid,pads,mode='constant')),
               'origin':torch.tensor(window.origin), 'volume_shape':torch.tensor(self.shapes[i][1:]),
               'core_bounds':torch.tensor([[axis.start,axis.stop] for axis in window.core])}
        if i in self.geometry:
            ctx['geometry']=torch.from_numpy(self.geometry[i][phase_indices])
        scale,offset = (self.config['normalization'][key] for key in ('scale','offset'))
        source = np.stack(((cbct[phase_indices]+offset)/scale,np.broadcast_to((prior+offset)/scale,cbct.shape)))
        batch = {'input':torch.from_numpy(source),'target':torch.from_numpy(((target[phase_indices]+offset)/scale)[None]),
                 'valid':ctx['source_fov'][None],'context':ctx,'case_id':record['case_id'],
                 'sample_id':sample_id(record),'animal_id':str(record['animal_id']),'source_path':record['path'],
                 'origin':torch.tensor(origin),'patch_shape':torch.tensor(shape)}
        if self.use_annotations:
            batch.update(self._annotation_patch(record,phase_indices,slices,pads,ctx['source_fov']))
        return batch

    def _annotation_patch(self,record,phase_indices,slices,pads,valid):
        shape = (len(phase_indices),2,*valid.shape[-3:])
        band = np.zeros(shape,dtype=bool)
        fields = {}
        use_surfaces = bool(self.resolved['motion']['surface_weight'])
        if use_surfaces:
            fields = {key:np.zeros(shape,dtype=np.float32 if key=='distance' else bool)
                      for key in ('surface','distance','correspondence')}
        if record.get('annotations'):
            import json
            from .cardiac_annotations import STRUCTURES, VERSION
            with h5py.File(record['annotations'],'r') as file:
                meta = json.loads(file.attrs['metadata_json'])
                if (meta['paired_sha256']!=record['sha256'] or meta['case_id']!=record['case_id']
                        or str(meta['animal_id'])!=str(record['animal_id'])
                        or meta['phase_labels']!=self.meta[self.by_id[sample_id(record)]]['phase_labels']
                        or meta['structures']!=list(STRUCTURES) or meta['version']!=VERSION
                        or meta['distance_unit']!='output_voxel'):
                    raise ValueError(f'{record["case_id"]}: incompatible contour annotation cache')
                for t,i in enumerate(phase_indices):
                    selector = (int(i),slice(None),*slices)
                    for key in fields:
                        fields[key][t] = np.pad(file[key][selector],[(0,0),*pads])
                    if self.resolved['boundary']['enabled']:
                        width = self.resolved['boundary']['half_width_vox']
                        local = (file['domain'][selector].astype(bool) & (file['distance'][selector]<=width)
                                 & file['annotation_available'][i,:].astype(bool)[:,None,None,None])
                        band[t] = np.pad(local,[(0,0),*pads])
        result = {key:torch.from_numpy(value) for key,value in fields.items()}
        result['bands'] = torch.from_numpy(band).transpose(0,1) & valid[None]
        return result


def validation_samples(records,config,stage):
    result=[]
    # Layouts of the same case use identical fixed validation coordinates.
    cases = sorted({(r['case_id'],str(r['animal_id'])) for r in records})
    case_positions = {case:i for i,case in enumerate(cases)}
    for r in sorted(records,key=sample_id):
        case_index=case_positions[r['case_id'],str(r['animal_id'])]
        subset=PatchDataset([r],config,stage,0,config['validation_samples_per_case'])
        for draw in range(config['validation_samples_per_case']):
            rng=np.random.default_rng(np.random.SeedSequence([config['seed'],stage,case_index,draw,104729]))
            i,shape,origin=subset.proposal(0,rng)
            for attempt in range(config['max_attempts']):
                if attempt: origin=[int(rng.integers(max(1,n-k+1))) for n,k in zip(subset.shapes[i][1:],shape)]
                spec={'case_id':r['case_id'],'sample_id':sample_id(r),'shape':list(shape),'origin':list(origin)}
                try:
                    fixed=PatchDataset([r],config,stage,0,1,[spec]); fixed[0]
                    result.append(spec); break
                except InvalidCore: continue
            else: raise ValueError(f'No valid fixed validation patch for {r["case_id"]}')
    return result
