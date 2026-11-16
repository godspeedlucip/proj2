from __future__ import annotations
from pathlib import Path
import csv
import numpy as np
import nibabel as nib
import yaml
from ..utils.io import atomic_json, atomic_jsonl


def make_phantom(out,views=240,phase_counts=(10,11,10),shape=(16,32,32)):
    out=Path(out).resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError('Phantom output must be empty')
    if views<200 or len(phase_counts)!=3 or any(t<2 for t in phase_counts): raise ValueError('Need >=200 views and three independent phantom animals')
    out.mkdir(parents=True,exist_ok=True)
    cycle=out/'cycle.csv'
    with cycle.open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f); writer.writerow(['cycle','max','min']); writer.writerow([0,'',0])
        bounds=np.linspace(0,views,5,dtype=int)
        for i in range(4): writer.writerow([i+1,int((bounds[i]+bounds[i+1])/2),int(bounds[i+1])])
    records=[]
    z,y,x=np.indices(shape,dtype=np.float32)
    z-=(shape[0]-1)/2; y-=(shape[1]-1)/2; x-=(shape[2]-1)/2
    for animal,t in enumerate(phase_counts):
        case=f'phantom_{animal}'; directory=out/case; directory.mkdir()
        phases={}
        for phase in range(1,t+1):
            hu=np.full(shape,-1000.,np.float32)
            body=(z/(shape[0]*.32))**2+(y/(8+animal*.5))**2+(x/(9-animal*.4))**2<1
            hu[body]=animal*25.
            moving=(z-(phase-(t+1)/2)*.3)**2+(y+1)**2+(x-2+animal*.3)**2<2.5**2
            hu[moving]=400.+animal*70
            path=directory/f'IM{phase:05d}_ct.nii.gz'
            nib.save(nib.Nifti1Image(hu,np.eye(4)),str(path)); phases[str(phase)]=str(path)
        records.append({'case_id':case,'animal_id':f'synthetic_{animal}','round':0,'series':case,
            'phase_paths':phases,'phase_labels':list(range(1,t+1)),
            'settings':{'max_phase':t},'synthetic':True})
    atomic_jsonl(out/'raw_manifest.jsonl',records)
    atomic_json(out/'split.json',{'train':['synthetic_0'],'val':['synthetic_1'],'test':['synthetic_2']})
    config={'profile':'dynamic_phantom_v1','cycle_csv':'cycle.csv','calibration':{'mode':'none'},
        'settings':{'n_projections':views,'angle_span_deg':300.,'voxel_spacing_multiplier':1.,
            'n_detector_rows':64,'n_detector_cols':64,'d_detector_row_mm':1.,'d_detector_col_mm':1.,
            'dso_mm':750.,'dsd_mm':1250.,'padding_row_pixels':4,'padding_col_pixels':4,
            'fov_output_pixels':shape[1],'fov_source_pixels':shape[1],'fov_mask_shrink_pixels':2},
        'prior_mode':'phase_weighted_fdk','full_object_fov':True}
    (out/'simulation.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    return {'manifest':str(out/'raw_manifest.jsonl'),'split':str(out/'split.json'),'config':str(out/'simulation.yaml')}
