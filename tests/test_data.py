import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from rstar4d.data.contracts import write_case, read_manifest, validate_manifest, write_manifest, basic_metadata
from rstar4d.data.sampling import PatchDataset, validation_samples
from rstar4d.data.paired import import_paired
from rstar4d.config import train_defaults
from rstar4d.utils.io import atomic_jsonl, atomic_json, sha256


def fixture(root,t=10,animal='a',case='case',shape=(4,16,16)):
    a=np.random.default_rng(1).random((t,*shape),dtype=np.float32)*.02
    p=root/(case+'.h5')
    meta=basic_metadata(t,[1,1,1]); meta['normalization']={'scale':.02,'offset':0.}
    affine=np.eye(4); affine[:3,3]=-(np.asarray(shape)-1)/2
    meta['geometry']={'shape_zyx':list(shape),'voxel_spacing_zyx_mm':[1,1,1],
        'center_zyx_mm':[0,0,0],'voxel_to_scanner_zyx':affine.tolist()}
    write_case(p,a,a.mean(0),a+0.002,np.ones_like(a,dtype=np.uint8),meta)
    return {'case_id':case,'animal_id':animal,'path':str(p),'sha256':sha256(p)}


class DataTests(unittest.TestCase):
    def test_contract_hash_and_animal_split(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); r=fixture(p); write_manifest(p/'manifest.jsonl',[r])
            validate_manifest(p/'manifest.jsonl',{'train':['a'],'val':[],'test':[]})
            with self.assertRaises(ValueError): validate_manifest(p/'manifest.jsonl',{'train':['a'],'val':['a'],'test':[]})
            r['sha256']='0'*64; write_manifest(p/'bad.jsonl',[r])
            with self.assertRaises(ValueError): validate_manifest(p/'bad.jsonl')

    def test_sampling_variable_t_and_synchronization(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); rows=[fixture(p),fixture(p,11,'b','case2')]
            write_manifest(p/'m.jsonl',rows); rows=read_manifest(p/'m.jsonl')
            cfg=train_defaults(); cfg.update(stage1_patch=[1,16,16],stage2_patches=[[4,16,16]])
            data=PatchDataset(rows,cfg,2,epoch=3,length=12)
            phases=set()
            for i in range(12):
                a=data[i]; b=data[i]
                self.assertTrue(torch.equal(a['input'],b['input']))
                torch.testing.assert_close(a['target']-a['input'][:1],torch.full_like(a['target'],.1),atol=1e-6,rtol=1e-6)
                phases.add(a['input'].shape[1])
            self.assertEqual(phases,{10,11})
            self.assertEqual(validation_samples(rows,cfg,2),validation_samples(rows,cfg,2))

    def test_small_volume_pad_and_empty_mask(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); r=fixture(p); write_manifest(p/'m.jsonl',[r]); rows=read_manifest(p/'m.jsonl')
            cfg=train_defaults(); cfg['stage2_patches']=[[6,16,16]]
            with self.assertRaises(ValueError): PatchDataset(rows,cfg,2,0,1)
            cfg['small_volume']='pad'
            sample=PatchDataset(rows,cfg,2,0,1)[0]
            self.assertEqual(sample['input'].shape,(2,10,6,16,16))
            self.assertEqual(int(sample['valid'][:,:,4:].sum()),0)

    def test_paired_entry_roundtrip_and_metadata_required(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); r=fixture(p)
            atomic_jsonl(p/'sources.jsonl',[{**r,'paired_h5':r['path']}])
            dest=import_paired(p/'sources.jsonl',p/'prepared')
            validate_manifest(dest)
            atomic_jsonl(p/'invalid.jsonl',[{'case_id':'x','animal_id':'a','cbct':['missing.tif'],'prior':'missing.tif'}])
            with self.assertRaises(ValueError): import_paired(p/'invalid.jsonl',p/'bad')

if __name__=='__main__': unittest.main()
