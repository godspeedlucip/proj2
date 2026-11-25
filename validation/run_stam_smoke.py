"""Run STAM's CLI on small paired volumes, optionally checking an existing pig scan crop."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from rstar4d.cli import main
from rstar4d.config import load_config
from rstar4d.data.contracts import read_manifest, write_case, write_manifest
from rstar4d.inference.sliding_window import starts
from rstar4d.training.checkpoint import load_checkpoint
from rstar4d.utils.io import atomic_json, atomic_jsonl, read_json, read_jsonl, sha256
from tests.test_training import inputs


PUBLIC_WEIGHTS = Path('RSTAR/githubcode/checkpoints/checkpoints_Tetris2.pth')


def cli(*args):
    if main([str(arg) for arg in args]):
        raise RuntimeError(f'CLI failed: {args[0]}')


class ThresholdFixture(torch.nn.Module):
    def forward(self, volume):
        return volume > 0.005


def seam_report(full_path, tiled_path, tile, overlap):
    full = read_jsonl(Path(full_path)/'manifest.jsonl')
    tiled = read_jsonl(Path(tiled_path)/'manifest.jsonl')
    result = {}
    for a, b in zip(full, tiled):
        if (a['case_id'], a['source_sha256']) != (b['case_id'], b['source_sha256']):
            raise ValueError('Full/tiled inputs differ')
        with h5py.File(Path(full_path)/a['path'], 'r') as af, h5py.File(Path(tiled_path)/b['path'], 'r') as bf:
            diff = np.abs(af['prediction'][:] - bf['prediction'][:])
        shape = diff.shape[1:]
        center = tuple(slice((n-1)//2 - min(4, n//4), (n-1)//2 + min(4, n//4) + 1) for n in shape)
        seam = np.zeros(shape, dtype=bool)
        for axis, (n, width, stride_overlap, align) in enumerate(zip(shape, tile, overlap, (2,8,8))):
            origins = starts(n, width, stride_overlap, align)
            edges = set(origins[1:]) | {o+width for o in origins[:-1]}
            for edge in edges:
                if 0 < edge < n:
                    sl = [slice(None)] * 3
                    sl[axis] = slice(edge - 1, edge + 1)
                    seam[tuple(sl)] = True
        result[a['case_id']] = {
            'full_seconds': a['inference_seconds'], 'tiled_seconds': b['inference_seconds'],
            'center_mae': float(diff[(slice(None),)+center].mean()),
            'seam_mae': float(diff[:, seam].mean()) if seam.any() else None,
            'global_mae': float(diff.mean()),
        }
    return result


def infer_both(checkpoint, manifest, root, device, tile, overlap, selection, amp=False):
    for mode in ('full', 'tiled'):
        args = ('infer', '--checkpoint', checkpoint, '--manifest', manifest, '--out', root/mode,
                '--split-name', selection, '--mode', mode, '--tile', *tile, '--overlap', *overlap,
                '--device', device)
        cli(*args) if amp else cli(*args, '--fp32')
    return seam_report(root/'full', root/'tiled', tile, overlap)


def phantom_smoke(root, device, sam):
    root.mkdir()
    common, manifest, split = inputs(root)
    source = root/'public_stage2_fixture.pt'
    cli('convert-weights', '--stage', 2, '--input', PUBLIC_WEIGHTS, '--out', source,
        '--intensity-domain', 'mu', '--normalization-scale', .02, '--split', split,
        '--temporal-topology', 'ordered_states')
    rows = {row['case_id']: row for row in read_manifest(manifest)}
    masks = root/'synthetic_lv.h5'
    z, y, x = np.indices((4, 16, 16)) - np.array([1.5, 7.5, 7.5])[:, None, None, None]
    with h5py.File(masks, 'w') as f:
        f['ED'] = (z * 2) ** 2 + y ** 2 + x ** 2 <= 10
        f['ES'] = (z * 2) ** 2 + y ** 2 + x ** 2 <= 5
    annotations = root/'synthetic_lv.jsonl'
    atomic_jsonl(annotations, [
        {'case_id': 'b', 'source_sha256': rows['b']['sha256'], 'phase_label': label,
         'role': role, 'path': masks.name, 'dataset': role, 'sha256': sha256(masks)}
        for label, role in ((3, 'ED'), (9, 'ES'))
    ])
    extractor = root/'threshold_fixture.pt'
    torch.jit.trace(ThresholdFixture(), torch.zeros((1,1,4,16,16))).save(str(extractor))
    results = {}
    for policy in ('frozen', 'joint'):
        config = load_config(f'configs/train/pig_stam_{policy}.yaml')
        common_stam = dict(config['stam'], sam_checkpoint=str(sam.resolve()))
        # Use the named A6000 execution settings with a deterministic two-update fixture.
        config.update(common, device=device, amp=False, stam=common_stam)
        config['deterministic'] = device == 'cpu'
        config_path = root/f'{policy}.yaml'
        config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
        run = root/policy
        cli('train', '--stage', 2, '--config', config_path, '--manifest', manifest, '--split', split,
            '--init-from-stage2', source, '--run-dir', run, '--stop-after-epoch', 1)
        last = run/'checkpoints/last.pt'
        checkpoint = load_checkpoint(last)
        reduced = {k: checkpoint[k] for k in ('step', 'stam', 'initialization')}
        reduced['full_vs_tiled'] = infer_both(last, manifest, run/'predictions', device,
                                             (4,16,8), (2,8,0), 'val')
        cli('evaluate', '--predictions', run/'predictions/tiled', '--manifest', manifest,
            '--out', run/'evaluation', '--lv-annotations', annotations, '--lv-extractor', extractor)
        reduced['lv'] = read_json(run/'evaluation/summary.json')['lv']
        results[policy] = reduced
    comparison = root/'paired_smoke.json'
    cli('compare-stam', '--candidate', root/'joint/evaluation', '--reference', root/'frozen/evaluation',
        '--seed', 2026, '--out', comparison)
    return results


def pig_crop_smoke(manifest, root, device):
    root.mkdir()
    row = read_manifest(manifest)[0]
    if sha256(row['path']) != row['sha256']:
        raise ValueError('Existing pig source checksum differs')
    with h5py.File(row['path'], 'r') as f:
        metadata = json.loads(f.attrs['metadata_json'])
        shape = np.minimum(f['cbct'].shape[1:], (17,129,129))
        origin = (np.asarray(f['cbct'].shape[1:]) - shape) // 2
        sl = tuple(slice(int(start), int(start + n)) for start, n in zip(origin, shape))
        affine = np.asarray(metadata['geometry']['voxel_to_scanner_zyx'])
        affine[:3, 3] = (affine @ np.r_[origin, 1])[:3]
        geometry = dict(metadata['geometry'], shape_zyx=shape.tolist(), voxel_to_scanner_zyx=affine.tolist(),
                        center_zyx_mm=(affine @ np.r_[(shape - 1) / 2, 1])[:3].tolist())
        metadata.update(geometry=geometry, full_object_fov=False)
        metadata['source_hashes']['stam_smoke_parent'] = row['sha256']
        metadata['stam_smoke_origin_zyx'] = origin.tolist()
        crop = root/'paired_crop.h5'
        ts = (slice(None),) + sl
        write_case(crop, f['cbct'][ts], f['prior'][sl], f['target'][ts], f['valid_mask'][ts], metadata)
    manifest = root/'manifest.jsonl'
    write_manifest(manifest, [{'case_id': row['case_id'], 'animal_id': row['animal_id'],
                               'path': crop, 'sha256': sha256(crop)}])
    source = root/'public_stage2_cross_domain.pt'
    cli('convert-weights', '--stage', 2, '--input', PUBLIC_WEIGHTS, '--out', source,
        '--intensity-domain', metadata['intensity_domain'],
        '--normalization-scale', metadata['normalization']['scale'],
        '--normalization-offset', metadata['normalization']['offset'],
        '--calibration-application', metadata['calibration_application'])
    report = infer_both(source, manifest, root/'predictions', device, (16,96,96), (8,48,48), 'all', amp=True)
    for mode in ('full', 'tiled'):
        cli('evaluate', '--predictions', root/'predictions'/mode, '--manifest', manifest,
            '--out', root/f'evaluation_{mode}')
    return {'public_weights_sha256': sha256(PUBLIC_WEIGHTS), 'paired_source_sha256': row['sha256'],
            'calibration': metadata['calibration'], 'normalization': metadata['normalization'],
            'phase_labels': metadata['phase_labels'], 'temporal_topology': metadata['temporal_topology'],
            'crop_zyx': shape.tolist(), 'origin_zyx': origin.tolist(), 'full_vs_tiled': report,
            'full_metrics': read_json(root/'evaluation_full/summary.json')['metrics'],
            'tiled_metrics': read_json(root/'evaluation_tiled/summary.json')['metrics']}


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--sam', type=Path, default=Path('weights/sam_vit_b_01ec64.pth'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--real-manifest', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    summary = {'scope': 'Synthetic CLI checks and optional cross-domain public-weight crop inspection',
               'phantom': phantom_smoke(args.out/'phantom', args.device, args.sam)}
    if args.real_manifest:
        summary['pig_crop_public_weights'] = pig_crop_smoke(args.real_manifest, args.out/'pig_crop', args.device)
    atomic_json(args.out/'summary.json', summary)


if __name__ == '__main__':
    run()
