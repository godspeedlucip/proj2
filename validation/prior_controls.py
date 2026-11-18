"""Replay the negative prior control and exercise a continuous-support positive phantom.

From a checkout, first run python -m validation.cardiac_smoke --out runs/smoke,
then python -m validation.prior_controls --smoke runs/smoke --out runs/prior-controls.
Pass --historical-manifest to additionally replay the archived negative NPZ.
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from rstar4d.data.acquisition import angle_summary
from rstar4d.evaluation.prior_diagnostics import diagnose_manifest
from rstar4d.utils.io import atomic_jsonl, read_json, read_jsonl, resolve, sha256


def negative_control(smoke):
    # Recreate the original control from the deterministic cardiac_smoke validation animal.
    with h5py.File(smoke/'val.h5','r') as case:
        dynamic = case['cbct'][:]/.02
        matched = case['prior'][:]/.02
    if dynamic.shape!=(2,8,24,24) or matched.shape!=(8,24,24):
        raise ValueError('Expected the two-phase cardiac_smoke validation phantom')
    with h5py.File(smoke/'val-angles.h5','r') as angles:
        geometry = angle_summary(angles['angles_rad'][:],angles['state_label'][:],[2,9])
    source = np.zeros_like(dynamic,bool)
    source[:,0,8,8] = True
    prior = np.zeros_like(matched,bool)
    prior[2:,12:,12:] = True
    return dict(dynamic=dynamic,matched=matched,mismatched=np.roll(matched,3,axis=1),
                source_fov=source,prior_fov=prior,geometry=geometry)


def positive_control(geometry):
    # A central valid cuboid leaves eight exterior x2 target positions whose
    # 3x3x3 search boxes intersect only a single corner. Beyond each corner,
    # the cuboid supplies real neighbors for trilinear fine sampling.
    z,y,x = np.ogrid[:16,:32,:32]
    radius = ((z-7)/5)**2 + ((y-16)/9)**2 + ((x-16)/8)**2
    matched = (.25+1.25*np.exp(-radius)+.12*np.sin(y/3)*np.cos(x/4)).astype(np.float32)
    dynamic = np.stack((matched,np.roll(matched,1,axis=1)))
    prior = np.zeros_like(matched,bool)
    prior[4:12,8:24,8:24] = True
    return dict(dynamic=dynamic,matched=matched,mismatched=np.roll(matched,8,axis=1),
                source_fov=np.ones_like(dynamic,bool),prior_fov=prior,geometry=geometry)


def assert_negative(case):
    counts = case['filter_counts']
    measured = tuple(counts[key] for key in ('target_sites','target_valid','single_coarse_both',
                                             'positive_fine_both','comparable'))
    if measured!=(288,2,2,0,0):
        raise AssertionError(f'Historical control target/valid/single/fine/comparable counts: {measured}')


def run(smoke, out, historical_manifest=None):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=False)
    checkpoint = smoke/'MPB/checkpoints/best.pt'
    negative = negative_control(smoke)
    controls = [('historical-negative',negative),('continuous-positive',positive_control(negative['geometry']))]
    rows = []
    for name,arrays in controls:
        path = out/(name+'.npz')
        np.savez_compressed(path,**arrays)
        rows.append({'case_id':name,'animal_id':'synthetic-'+name,'phase_labels':[2,9],
                     'path':path.name,'sha256':sha256(path)})
    manifest = out/'controls.jsonl'
    atomic_jsonl(manifest,rows)
    report = out/'diagnostics.json'
    diagnose_manifest(checkpoint,manifest,report)
    negative_case,positive_case = read_json(report)['cases']
    assert_negative(negative_case)
    if positive_case['points']<=0 or any(phase['points']<=0 for phase in positive_case['phases']):
        raise AssertionError('Expected comparable points for both synthetic positive-control phases')
    if historical_manifest is not None:
        row, = read_jsonl(historical_manifest)
        source = resolve(row['path'],historical_manifest.parent)
        with np.load(source,allow_pickle=False) as saved:
            for key,expected in negative.items():
                np.testing.assert_array_equal(saved[key],expected,err_msg=key)
        original = out/'historical-replay.json'
        diagnose_manifest(checkpoint,historical_manifest,original)
        assert_negative(read_json(original)['cases'][0])
    return report


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--historical-manifest',type=Path)
    args = parser.parse_args()
    print(run(args.smoke,args.out,args.historical_manifest))
