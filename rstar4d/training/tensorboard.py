"""Publish complete, replayable TensorBoard epochs from committed training archives."""
from collections import Counter
from contextlib import contextmanager
from functools import lru_cache
import json
from pathlib import Path
import shutil
import tempfile

import h5py
import torch

from .checkpoint import atomic_file, load_checkpoint
from ..data.acquisition import load_geometry
from ..data.contracts import validate_manifest
from ..inference.sliding_window import predict_volume
from ..utils.io import sha256


class CardiacAverages:
    """Pool motion terms by their support, and the joint B objective by supervised draws."""
    def __init__(self):
        self.sums = Counter()
        self.counts = Counter()

    def update(self, diagnostics):
        structures = diagnostics.get('structures', {})
        weights = {
            'struct': diagnostics.get('windows', 0),
            'smooth': diagnostics.get('smooth_pairs', 0),
            'surface': diagnostics.get('surface_pairs', 0),
            'boundary': int(any(item['voxels'] for item in structures.values())),
        }
        for name, count in weights.items():
            if count:
                for key in (name, name + '_weighted'):
                    self.sums[key] += diagnostics[key] * count
                    self.counts[key] += count
        for name, item in structures.items():
            for field, denom in [('l1', 'voxels'), ('gradient', 'pairs')]:
                count = item.get(denom, 0)
                if count:
                    key = name + '_' + field
                    self.counts[key] += count
                    self.sums[key] += item[field] * count

    def values(self):
        return {k: self.sums[k] / n for k, n in self.counts.items()}


@lru_cache(maxsize=16)
def _reference_hash(path, size, mtime_ns):
    # The same fixed sidecar can be large. Refresh its hash when its file changes.
    return sha256(path)


def validation_preview(x, pred, target, batch, record, config):
    """Capture one fixed validation phase and core slice; intensities share a fixed scale."""
    _, _, phases, depth, height, width = pred.shape
    t, z = phases // 2, depth // 2
    local = (0, slice(None), t, z)
    scale, offset = (config['normalization'][k] for k in ('scale', 'offset'))
    lo, hi = config['loss']['window']
    if hi <= lo:
        raise ValueError('TensorBoard display window requires increasing intensity bounds')
    def display(array):
        return (array.clamp(0, 1) * 255).round().to(dtype=torch.uint8, device='cpu')
    images = {}
    for key, plane in [('sparse', x[local][0:1]), ('prior', x[local][1:2]),
                       ('prediction', pred[local]), ('reference', target[local])]:
        images[key] = display((plane.float() * scale - offset - lo) / (hi - lo))
    images['absolute_error'] = display((pred[local].float() - target[local]).abs() * scale / (hi - lo))
    origin = batch['origin'][0].tolist()
    gz, gy, gx = origin[0] + z, origin[1], origin[2]
    labels = record['metadata']['phase_labels']
    preview = {'images': images, 'case_id': record['case_id'],
               'sample_id': record.get('sample_id', record['case_id']),
               'phase_label': labels[t], 'slice_zyx': [gz, gy, gx], 'window': [lo, hi]}
    # The display reads only two annotated planes. A missing optional sidecar never
    # becomes an input requirement for a reconstruction-only run.
    sidecar = record.get('annotations')
    if sidecar and Path(sidecar).is_file():
        try:
            stat = Path(sidecar).stat()
            if _reference_hash(sidecar, stat.st_size, stat.st_mtime_ns) != record.get('annotations_sha256'):
                preview['overlay_status'] = 'reference_hash_mismatch'
                return preview
            with h5py.File(sidecar, 'r') as ref:
                meta = json.loads(ref.attrs['metadata_json'])
                if (meta.get('paired_sha256') != record['sha256'] or meta.get('phase_labels') != labels
                        or meta.get('case_id') != record['case_id']
                        or str(meta.get('animal_id')) != str(record['animal_id'])
                        or meta.get('structures') != ['heart_outer', 'lv']):
                    preview['overlay_status'] = 'reference_metadata_mismatch'
                    return preview
                available = torch.from_numpy(ref['annotation_available'][t]).bool()
                if gz >= ref['surface'].shape[2] or not available.any():
                    return preview
                mask = torch.from_numpy(ref['surface'][t, :, gz, gy:gy+height, gx:gx+width]).bool()
                mask &= available[:, None, None]
        except (OSError, KeyError, IndexError, ValueError) as exc:
            preview['overlay_status'] = f'optional surface unavailable ({type(exc).__name__})'
            return preview
        overlay = images['prediction'].expand(3, -1, -1).clone()
        cropped = overlay[:, :mask.shape[1], :mask.shape[2]]
        for outline, color in zip(mask, ((255, 105, 0), (0, 215, 255))):
            cropped[:, outline] = torch.tensor(color, dtype=torch.uint8)[:, None]
        images['prediction_reference_contours'] = overlay
    return preview


def measured_preview_source(options, resolved):
    """Validate the separately provided, unlabeled reconstruction preview."""
    sidecars = ('projections','angle_summary') if resolved['prior']['mode']=='geometry' else ()
    rows = validate_manifest(options['manifest'],require_target=False,full=False,
                             sidecars=sidecars,prefer_angle_summary=True)
    if len(rows)!=1:
        raise ValueError('TensorBoard measured preview requires exactly one manifest case')
    record=rows[0]
    labels=record['metadata']['phase_labels']
    if options['phase_label'] is not None and options['phase_label'] not in labels:
        raise ValueError('TensorBoard measured preview phase is not in the manifest')
    if options['axial_slice'] is not None and options['axial_slice']>=record['shape'][1]:
        raise ValueError('TensorBoard measured preview axial index is outside the volume')
    if sidecars:
        load_geometry(record,labels)
    return record


def axial_preview(model, record, config, resolved, device, *, phase_label=None, axial_slice=None):
    """Read only the tiles contributing to a fixed full axial slice."""
    labels = record['metadata']['phase_labels']
    phase = len(labels)//2 if phase_label is None else labels.index(phase_label)
    z = record['shape'][1]//2 if axial_slice is None else axial_slice
    lo, hi = config['loss']['window']
    tile = config['stage1_patch'] if model.stage==1 else resolved['recipe']['core_shapes'][-1]
    alignment = (1,8,8) if model.stage==1 else (2,8,8)
    tile = tuple(((k+a-1)//a)*a for k,a in zip(tile,alignment))

    def display(plane):
        return ((plane-lo)/(hi-lo)).clamp(0,1).mul(255).round().to(torch.uint8)[None]

    with h5py.File(record['path'],'r') as source:
        context = {'case_id':record['case_id'],'source_fov':source['valid_mask'],
                   'prior_fov':source.get('prior_valid_mask'),'phase_labels':labels}
        if resolved['prior']['mode']=='geometry':
            context['geometry'] = load_geometry(record,labels)
        prediction = predict_volume(model,source['cbct'],source['prior'],config['normalization'],
                                    tile=tile,overlap=(0,0,0),device=device,amp=config['amp'],
                                    context=context,recipe=resolved,axial_slice=z)
        predicted = torch.from_numpy(prediction[phase])
        images = {'prediction':display(predicted),
                  'sparse':display(torch.from_numpy(source['cbct'][phase,z]))}
        if 'target' in source:
            target = torch.from_numpy(source['target'][phase,z])
            valid = torch.from_numpy(source['valid_mask'][phase,z]).bool()
            difference = (predicted-target).masked_fill(~valid,0)
            unit = (difference/(hi-lo)).clamp(-1,1)
            positive, negative = unit.clamp_min(0), (-unit).clamp_min(0)
            images['reference'] = display(target)
            images['absolute_error'] = (unit.abs()*255).round().to(torch.uint8)[None]
            images['residual'] = (torch.stack((positive,torch.zeros_like(unit),negative))*255).round().to(torch.uint8)
    return {'images':images,'case_id':record['case_id'],'sample_id':record['sample_id'],
            'source_sha256':record['sha256'],'phase_label':labels[phase],'axial_slice':z,
            'window':[lo,hi],'residual_display_range':[lo-hi,hi-lo],
            'residual_convention':'prediction minus reference; red positive, blue negative'}


@contextmanager
def _event_file(directory, step):
    """TensorBoard recognizes the final name only after the writer has flushed and closed."""
    from torch.utils.tensorboard import SummaryWriter
    directory.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(prefix='rstar-tb-') as tmp:
        with SummaryWriter(tmp) as writer:
            yield writer
        file, = Path(tmp).glob('events.out.tfevents.*')
        destination = directory / f'events.out.tfevents.{step:010d}.rstar'
        with file.open('rb') as source, atomic_file(destination, 'wb', prefix='.tb-pending-') as output:
            shutil.copyfileobj(source, output)


def write_training_epoch(run_dir, payload):
    if not payload['config'].get('tensorboard', True) or 'epoch_metrics' not in payload:
        return
    epoch = payload['completed_epoch']
    entry = payload['epoch_metrics']
    tags = {'train_total': 'train/total', 'train_l1': 'train/l1',
            'validation_l1': 'validation/case_l1', 'validation_animal_l1': 'validation/animal_l1',
            'lr': 'optimizer/learning_rate', 'steps': 'optimizer/total_updates',
            'optimizer_updates': 'optimizer/epoch_updates', 'amp_overflows': 'optimizer/amp_overflows',
            'seconds': 'resources/epoch_seconds', 'peak_allocated_gib': 'resources/peak_allocated_gib',
            'skipped_ssim': 'train/skipped_ssim', 'valid_fraction': 'train/valid_fraction'}
    with _event_file(Path(run_dir) / 'tensorboard', epoch) as writer:
        for key, tag in tags.items():
            writer.add_scalar(tag, entry[key], epoch)
        for key, value in entry.get('cardiac_means', {}).items():
            writer.add_scalar('loss/' + key, value, epoch)
        counts = entry.get('cardiac', {})
        for key in ('windows', 'outside', 'low_contrast', 'cross_core', 'surface_pairs', 'surface_points',
                    'heart_outer_hit_cores', 'heart_outer_voxels', 'lv_hit_cores', 'lv_voxels',
                    'boundary_overlap', 'boundary_dual_hit'):
            if key in counts:
                writer.add_scalar('support/' + key, counts[key], epoch)
        if counts.get('windows'):
            writer.add_scalar('motion/outside_fraction', counts['outside'] / counts['windows'], epoch)
        if 'boundary_weight' in counts:
            writer.add_scalar('loss/boundary_coefficient', counts['boundary_weight'] / entry['sample_count'], epoch)
        if epoch == 1:
            settings = {'stage': payload['stage'], 'model_profile': payload['model_profile'],
                        'seed': payload['config']['seed'], 'initialization': payload['initialization']['kind'],
                        'source_sha256': payload['initialization']['sha256'],
                        'stage2': payload.get('stage2_resolved')}
            writer.add_text('run/configuration', json.dumps(settings, ensure_ascii=False, sort_keys=True), epoch)
        preview = payload.get('tensorboard_preview')
        if preview:
            for key, image in preview['images'].items():
                writer.add_image('validation/' + key, image, epoch)
            metadata = {k: v for k, v in preview.items() if k != 'images'}
            writer.add_text('validation/sample', json.dumps(metadata, ensure_ascii=False), epoch)
        for group, plane in payload.get('tensorboard_slices', {}).items():
            for key, image in plane['images'].items():
                writer.add_image(group + '/' + key, image, epoch)
            metadata = {k:v for k,v in plane.items() if k!='images'}
            writer.add_text(group + '/sample',json.dumps(metadata,ensure_ascii=False),epoch)


def recover_training_events(run_dir, completed_epoch, enabled=True):
    folder = Path(run_dir)
    events = folder / 'tensorboard'
    for event in events.glob('events.out.tfevents.*.rstar'):
        suffix = event.name.split('.')[-2]
        if suffix.isdecimal() and int(suffix) > completed_epoch:
            event.unlink()
    if not enabled:
        return
    for path in sorted((folder / 'checkpoints').glob('epoch-*.pt')):
        suffix = path.stem.removeprefix('epoch-')
        if not suffix.isdecimal():
            continue
        epoch = int(suffix)
        event = events / f'events.out.tfevents.{epoch:010d}.rstar'
        if epoch <= completed_epoch and not event.is_file():
            write_training_epoch(folder, load_checkpoint(path))


def write_evaluation_events(out, summary):
    if 'contours' not in summary:
        return
    with _event_file(Path(out) / 'tensorboard', 0) as writer:
        for structure, metrics in summary['contours']['structures'].items():
            for key in ('surface_dice', 'hd95', 'assd', 'band_mae'):
                stats = metrics[key]
                prefix = f'evaluation/{structure}/{key}'
                writer.add_scalar(prefix + '_valid_animals', stats['n'], 0)
                if stats['n']:
                    writer.add_scalar(prefix, stats['mean'], 0)
