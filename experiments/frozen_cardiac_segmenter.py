"""One independent mask predictor shared by all reconstruction arms."""
import argparse
import json
from pathlib import Path
import tempfile

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rstar4d.data.contracts import read_manifest, validate_split
from rstar4d.training.checkpoint import atomic_save
from rstar4d.utils.io import append_jsonl, atomic_json, read_json, sha256


VERSION = 'cardiac-axial-2p5d-v1'
WINDOW = (0.0, 0.05)
IGNORE = 255


def block(in_channels, channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
        nn.GroupNorm(4, channels), nn.SiLU(),
        nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        nn.GroupNorm(4, channels), nn.SiLU(),
    )


class Segmenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = block(3, 16)
        self.second = block(16, 32)
        self.third = block(32, 64)
        self.center = block(64, 96)
        self.decode_third = block(96 + 64, 64)
        self.decode_second = block(64 + 32, 32)
        self.decode_first = block(32 + 16, 16)
        self.classes = nn.Conv2d(16, 3, 1)

    def forward(self, image):
        first = self.first(image)
        second = self.second(F.max_pool2d(first, 2))
        third = self.third(F.max_pool2d(second, 2))
        feature = self.center(F.max_pool2d(third, 2))
        for skip, decoder in ((third, self.decode_third), (second, self.decode_second),
                              (first, self.decode_first)):
            feature = F.interpolate(feature, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            feature = decoder(torch.cat((skip, feature), dim=1))
        return self.classes(feature)


def normalized(volume):
    lo, hi = WINDOW
    return (np.clip(np.asarray(volume, dtype=np.float32), lo, hi) - lo) / (hi - lo)


def axial_input(volume, indices):
    neighbors = np.asarray(indices)[:, None] + np.arange(-1, 2)
    return volume[np.clip(neighbors, 0, len(volume)-1)]


def predict_mask(model, volume, device, batch=12):
    """Predict 0 background, 1 heart outside LV, 2 LV on the source ZYX grid."""
    volume = normalized(volume)
    result = np.empty(volume.shape, dtype=np.uint8)
    with torch.inference_mode():
        for z in range(0, len(volume), batch):
            indices = np.arange(z, min(z + batch, len(volume)))
            images = torch.from_numpy(axial_input(volume, indices)).to(device)
            result[indices] = model(images).argmax(1).cpu().numpy()
    return result


def cached_phases(manifest, split_path, cache):
    rows = read_manifest(manifest)
    split = validate_split(rows, split_path)
    dataset = {'train': [], 'val': []}
    for part in dataset:
        for row in rows:
            if row['animal_id'] not in split[part] or not row.get('annotations'):
                continue
            with h5py.File(row['path'], 'r') as source, h5py.File(row['annotations'], 'r') as anno:
                phases = json.loads(source.attrs['metadata_json'])['phase_labels']
                for t, phase in enumerate(phases):
                    if not anno['annotation_available'][t].all():
                        continue
                    key = f"{row['case_id']}_p{phase:02d}"
                    directory = cache / key
                    contract = {'paired': row['sha256'], 'annotations': row['annotations_sha256'],
                                'phase': phase, 'split': part, 'window': list(WINDOW)}
                    if directory.exists():
                        if read_json(directory / 'source.json') != contract:
                            raise ValueError(f'Mask segmenter cache changed at {directory}')
                    else:
                        with tempfile.TemporaryDirectory(dir=cache, prefix='.phase-') as scratch:
                            temporary = Path(scratch) / key
                            temporary.mkdir()
                            region = anno['region'][t].astype(bool)
                            valid = anno['domain'][t].astype(bool).all(0) & source['valid_mask'][t].astype(bool)
                            label = (region[0] | region[1]).astype(np.uint8) + region[1].astype(np.uint8)
                            label[~valid] = IGNORE
                            if not ((label == 1).any() and (label == 2).any()):
                                raise ValueError(f'Expected both trusted anatomical classes in {key}')
                            np.save(temporary/'labels.npy', label)
                            for name in ('target', 'cbct'):
                                np.save(temporary/f'{name}.npy', normalized(source[name][t]).astype(np.float16))
                            z_valid = np.flatnonzero(valid.any(axis=(1, 2))).tolist()
                            z_heart = np.flatnonzero(((label > 0) & (label < IGNORE)).any(axis=(1, 2))).tolist()
                            atomic_json(temporary/'slices.json', {'valid': z_valid, 'heart': z_heart})
                            atomic_json(temporary/'source.json', contract)
                            temporary.rename(directory)
                        print(f'Cached segmenter {part} {key}', flush=True)
                    dataset[part].append({
                        'target': np.load(directory/'target.npy', mmap_mode='r'),
                        'cbct': np.load(directory/'cbct.npy', mmap_mode='r'),
                        'label': np.load(directory/'labels.npy', mmap_mode='r'),
                        'slices': read_json(directory/'slices.json'),
                        'case_id': row['case_id'], 'phase': phase,
                    })
    if not all(dataset.values()):
        raise ValueError('Segmenter needs labeled train and validation scans')
    return dataset


def sample_batch(rows, rng, batch):
    images, labels = [], []
    for _ in range(batch):
        row = rows[int(rng.integers(len(rows)))]
        slice_set = 'heart' if rng.random()<0.7 else 'valid'
        z = int(rng.choice(row['slices'][slice_set]))
        clean = axial_input(row['target'], [z])[0].astype(np.float32)
        noisy = axial_input(row['cbct'], [z])[0].astype(np.float32)
        mix = float(rng.uniform(0., .65)) if rng.random()<.65 else 0.
        image = clean + mix * (noisy-clean)
        gain = float(rng.uniform(.9, 1.1))
        offset = float(rng.uniform(-.025, .025))
        image = np.clip(image*gain+offset, 0, 1)
        label = row['label'][z]
        if rng.random()<.5:
            image, label = image[..., ::-1], label[..., ::-1]
        images.append(image)
        labels.append(label)
    return torch.from_numpy(np.stack(images)), torch.from_numpy(np.stack(labels).astype(np.int64))


def objective(logits, labels):
    valid = labels != IGNORE
    ce = F.cross_entropy(logits, labels, weight=logits.new_tensor([.4, 1., 2.]), ignore_index=IGNORE)
    probs = logits.float().softmax(1)
    p = torch.stack((probs[:, 1] + probs[:, 2], probs[:, 2]), dim=1)
    gt = torch.stack(((labels > 0) & valid, labels == 2), dim=1)
    weighted = p * valid[:, None]
    dice = (2*(weighted * gt).sum((0, 2, 3))+1) / (weighted.sum((0, 2, 3))+gt.sum((0, 2, 3))+1)
    return ce + .5 * (1-dice.mean())


def validation(model, rows, device, batch):
    scores = []
    with torch.inference_mode():
        for row in rows:
            available = row['slices']['valid']
            indices = [available[i] for i in np.unique(np.linspace(0, len(available)-1, 24).round().astype(int))]
            target = row['label'][indices]
            valid = target != IGNORE
            prediction = []
            for start in range(0,len(indices),batch):
                images = axial_input(row['target'],indices[start:start+batch]).astype(np.float32)
                prediction.append(model(torch.from_numpy(images).to(device)).argmax(1).cpu().numpy())
            pred = np.concatenate(prediction)
            dice = []
            for threshold in (1,2):
                y = (target>=threshold)&valid
                p = (pred>=threshold)&valid
                total = int(y.sum()+p.sum())
                dice.append(float(2*np.count_nonzero(y&p)/total) if total else np.nan)
            scores.append(dice)
    result = np.nanmean(scores, axis=0)
    if not np.isfinite(result).all():
        raise ValueError('Missing structure coverage in segmenter validation')
    return {'heart_outer':float(result[0]),'lv':float(result[1])}


def train(manifest, split, out, epochs, steps, batch, device):
    out.mkdir(parents=True, exist_ok=True)
    cache = out/'cache'
    cache.mkdir(exist_ok=True)
    identity = {'version':VERSION,'manifest_sha256':sha256(manifest),'split_sha256':sha256(split),
                'epochs':epochs,'steps':steps,'batch':batch,'seed':2026}
    if (out/'identity.json').exists() and read_json(out/'identity.json')!=identity:
        raise ValueError('Segmenter run configuration changed')
    atomic_json(out/'identity.json',identity)
    dataset = cached_phases(manifest,split,cache)
    torch.manual_seed(identity['seed'])
    model = Segmenter().to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.0003)
    last, best_file = out/'last.pt', out/'best.pt'
    best, start = -1., 1
    if last.exists():
        saved = torch.load(last,map_location=device,weights_only=True)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        best, start = saved['best'], saved['epoch']+1
    for epoch in range(start,epochs+1):
        model.train()
        rng = np.random.default_rng(np.random.SeedSequence([identity['seed'],epoch]))
        loss_total = 0.
        for _ in range(steps):
            x,y = sample_batch(dataset['train'],rng,batch)
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model(x.to(device)), y.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite segmenter objective')
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach())
        model.eval()
        scores = validation(model,dataset['val'],device,batch)
        score = sum(scores.values())/2
        if score>best:
            best = score
            atomic_save(best_file,{'model':model.state_dict(),'version':VERSION,'epoch':epoch,'validation':scores})
        atomic_save(last,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'best':best})
        record = {'epoch':epoch,'loss':loss_total/steps,'validation_dice':scores}
        append_jsonl(out/'train.jsonl',record)
        print(json.dumps(record),flush=True)
    frozen = torch.load(best_file,map_location='cpu',weights_only=True)
    atomic_json(out/'frozen.json',{'method':'frozen_segmenter','version':VERSION,
        'weights_sha256':sha256(best_file),'threshold':'three-class argmax',
        'best_epoch':frozen['epoch'],'validation_dice':frozen['validation'],
        'labeled_train_phases':len(dataset['train']),'labeled_validation_phases':len(dataset['val']),
        'identity':identity})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--split',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=24)
    parser.add_argument('--steps',type=int,default=64)
    parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--device',default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(2)
    train(args.manifest,args.split,args.out,args.epochs,args.steps,args.batch,torch.device(args.device))
