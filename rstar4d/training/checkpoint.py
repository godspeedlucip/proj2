from __future__ import annotations
from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import uuid
import torch
from ..utils.io import atomic_json, atomic_jsonl, canonical_hash, read_json, sha256


def basic_types(value):
    """Keep checkpoints compatible with weights_only loading, including LR Counter state."""
    if torch.is_tensor(value): return value
    if isinstance(value,dict): return {basic_types(k):basic_types(v) for k,v in value.items()}
    if isinstance(value,tuple): return tuple(basic_types(v) for v in value)
    if isinstance(value,list): return [basic_types(v) for v in value]
    if value is None or type(value) in (int,float,bool,str): return value
    if isinstance(value,str): return str(value)
    raise TypeError(f'Unsupported checkpoint value type: {type(value)}')


@contextmanager
def atomic_file(path, mode, prefix=None, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(mode, dir=path.parent, prefix=prefix or path.name+'.', suffix='.tmp',
                                            delete=False, **kwargs)
    try:
        with temporary:
            yield temporary
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary.name, path)
        if os.name == 'posix':
            fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        Path(temporary.name).unlink(missing_ok=True)


def atomic_save(path, payload):
    with atomic_file(path, 'wb') as output:
        torch.save(basic_types(payload), output)


def load_checkpoint(path,require_training=True):
    value=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(value,dict) or value.get('schema_version')!=1 or 'state_dict' not in value:
        raise ValueError('Expected a versioned RSTAR4D checkpoint; raw public weights are warm starts, not resume checkpoints')
    if require_training and value.get('checkpoint_kind')!='training': raise ValueError('Weights-only checkpoint cannot resume training')
    if require_training:
        required={'optimizer','scheduler','scaler','rng','config','data_snapshot','stage','completed_epoch','next_epoch','signature'}
        if not required.issubset(value): raise ValueError('Incomplete training checkpoint')
    return value


@contextmanager
def run_lock(run_dir):
    """Serialise writers. The OS releases the lock if a worker dies."""
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir.with_name('.'+run_dir.name+'.training.lock')
    with lock_path.open('a+b') as lock:
        try:
            if os.name == 'nt':
                import msvcrt
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f'Training directory is already in use: {run_dir}') from exc
        yield


def archive_path(ckdir, epoch):
    return ckdir / f'epoch-{epoch:04d}.pt'


def archives(ckdir):
    result = {}
    for path in ckdir.glob('epoch-*.pt'):
        match = re.fullmatch(r'epoch-(\d+)\.pt', path.name)
        if match:
            epoch = int(match[1])
            if epoch > 0 and path == archive_path(ckdir, epoch):
                result[epoch] = path
    return dict(sorted(result.items()))


def resume_path(request):
    """Resolve last.pt even when its alias is missing or one epoch behind."""
    request = Path(request).resolve()
    if request.parent.name != 'checkpoints':
        raise ValueError('Resume requires a checkpoint from its owning run')
    history = archives(request.parent)
    if request.name == 'last.pt':
        return history[max(history)] if history else request
    if request not in history.values():
        raise ValueError('Resume requires last.pt or a committed epoch archive')
    return request


def resume_defaults(request):
    """Read just the CLI arguments needed for an epoch or a pre-epoch restart."""
    path = resume_path(request)
    if path.is_file():
        ck = load_checkpoint(path)
        return {k: ck[k] for k in ('config', 'manifest', 'split', 'stage')}
    run = path.parent.parent
    identity = read_json(run / 'run_identity.json')
    return {'manifest': identity['manifest'], 'stage': identity['stage'],
            'config': read_json(run / 'resolved_config.json'), 'split': read_json(run / 'split.json')}


def _copy_checkpoint(src, dest):
    with src.open('rb') as source, atomic_file(dest, 'wb') as output:
        shutil.copyfileobj(source, output, length=1024*1024)


def _repair_alias(src, dest):
    if not dest.is_file() or sha256(dest) != sha256(src):
        _copy_checkpoint(src, dest)


def publish_epoch(ckdir, payload):
    epoch = payload['completed_epoch']
    path = archive_path(ckdir, epoch)
    if path.exists():
        raise FileExistsError(f'Epoch {epoch} already committed')
    atomic_save(path, payload)
    if payload['best_epoch'] == epoch:
        _copy_checkpoint(path, ckdir / 'best.pt')
    _copy_checkpoint(path, ckdir / 'last.pt')


def _check_owner(payload, identity):
    if any(payload.get(key) != identity.get(key) for key in ('stage', 'signature', 'run_id')):
        raise ValueError('Resume run ownership or signature mismatch')
    expected = identity.get('validation_samples_sha256')
    if expected and canonical_hash(payload['validation_samples']) != expected:
        raise ValueError('Resume fixed validation sample mismatch')


def read_resume(request, run_dir):
    path = resume_path(request)
    source = path.parent.parent
    identity_path = source / 'run_identity.json'
    if not identity_path.is_file():
        raise ValueError('Missing resume run identity')
    identity = read_json(identity_path)
    if not path.is_file():
        status = source / 'status.json'
        if (source != run_dir or 'initialization' not in identity or
                (path.parent / 'best.pt').exists() or
                (status.exists() and read_json(status)['completed_epoch'] > 0)):
            raise ValueError('Missing committed checkpoint or first-epoch initialization record')
        return None, identity, source
    saved = load_checkpoint(path)
    _check_owner(saved, identity)
    history = archives(path.parent)
    if history:
        if saved['completed_epoch'] != int(path.stem.split('-')[1]) or 'epoch_metrics' not in saved:
            raise ValueError('Invalid epoch archive')
        historical = saved['completed_epoch'] < max(history)
        if historical and source == run_dir:
            raise ValueError('A historical epoch requires --run-dir pointing to a new run')
        if source != run_dir:
            if not historical or source in run_dir.parents:
                raise ValueError('Only a historical epoch can be forked to a separate new run')
            if run_dir.exists() and any(run_dir.iterdir()):
                raise FileExistsError('Historical resume requires an empty new run directory')
    else:
        if source != run_dir or 'epoch_metrics' in saved:
            raise ValueError('Missing owning epoch archives; migrate the entire run or use a warm start')
        best_path = path.parent / 'best.pt'
        if not best_path.is_file():
            raise ValueError('Legacy resume requires the historical best.pt from the same run')
        best = load_checkpoint(best_path)
        _check_owner(best, identity)
        if best['best'] != saved['best'] or best['completed_epoch'] > saved['completed_epoch']:
            raise ValueError('Resume best/last checkpoint lineage mismatch')
        saved['best_epoch'] = best['completed_epoch']
        identity['legacy_through_epoch'] = saved['completed_epoch']
    if saved['next_epoch'] != saved['completed_epoch']+1 or not 1 <= saved['best_epoch'] <= saved['completed_epoch']:
        raise ValueError('Invalid checkpoint epoch boundary')
    return saved, identity, source


CSV_FIELDS = ('epoch', 'stage', 'steps', 'train_l1', 'train_total', 'validation_l1',
              'validation_animal_l1', 'lr', 'seconds', 'peak_allocated_gib', 'skipped_ssim')


def _csv_writer(output):
    return csv.DictWriter(output, fieldnames=CSV_FIELDS, extrasaction='ignore')


def append_metrics_csv(path, row):
    exists = path.exists()
    with path.open('a', encoding='utf-8', newline='') as output:
        writer = _csv_writer(output)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _committed_rows(path, through):
    if path.exists():
        with path.open('rb') as source:
            for line in source:
                try:
                    row = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    if line.endswith(b'\n'):
                        raise
                    break  # a process can leave its final write incomplete
                if row['epoch'] <= through:
                    yield row


def _repair_logs(source, dest, summaries, through):
    legacy_rows = {}
    remaining = iter(sorted(summaries))

    def train_rows():
        pending = next(remaining, None)
        for row in _committed_rows(source / 'train.jsonl', through):
            epoch = row['epoch']
            while pending is not None and pending < epoch:
                yield summaries[pending]
                pending = next(remaining, None)
            if row.get('event') == 'epoch':
                if epoch in summaries or epoch in legacy_rows:
                    continue
                legacy_rows[epoch] = row
            yield row
        while pending is not None:
            yield summaries[pending]
            pending = next(remaining, None)

    atomic_jsonl(dest / 'train.jsonl', train_rows())
    cardiac = source / 'cardiac_samples.jsonl'
    if cardiac.exists():
        atomic_jsonl(dest / cardiac.name, _committed_rows(cardiac, through))
    # An old run may have only CSV for some of its pre-archive epochs.
    old_csv = source / 'metrics.csv'
    if old_csv.exists() and (not summaries or min(summaries) > 1):
        with old_csv.open(encoding='utf-8', newline='') as prior:
            for row in csv.DictReader(prior):
                number = row.get('epoch', '')
                if number and number.isdigit() and int(number) <= through and all(row.get(k) for k in CSV_FIELDS):
                    legacy_rows.setdefault(int(number), row)
    with atomic_file(dest / 'metrics.csv', 'w', encoding='utf-8', newline='') as output:
        writer = _csv_writer(output)
        writer.writeheader()
        rows = {**legacy_rows, **summaries}
        writer.writerows(rows[e] for e in sorted(rows))


def _restore_history(source, dest, saved, parent_identity, identity):
    source_ck, dest_ck = (p / 'checkpoints' for p in (source, dest))
    through = saved['completed_epoch']
    legacy = identity.get('legacy_through_epoch', 0)
    best_epoch = saved['best_epoch']
    history = archives(source_ck)
    best_source = archive_path(dest_ck, best_epoch)
    if legacy:
        anchor = source_ck / 'legacy-best.pt'
        if not anchor.exists():
            if history:
                raise ValueError('Missing historical legacy-best.pt anchor')
            anchor = source_ck / 'best.pt'
        best = load_checkpoint(anchor)
        _check_owner(best, parent_identity)
        local_anchor = dest_ck / 'legacy-best.pt'
        if best_epoch <= legacy:
            if best['completed_epoch'] != best_epoch or best['best'] != saved['best']:
                raise ValueError('Historical best checkpoint lineage mismatch')
            best_source = local_anchor
        if source != dest:
            atomic_save(local_anchor, {**best, 'run_id': identity['run_id']})
        elif anchor != local_anchor:
            _copy_checkpoint(anchor, local_anchor)
    summaries = {}
    for epoch in range(legacy+1, through+1):
        path = archive_path(source_ck, epoch)
        if epoch not in history:
            raise ValueError(f'Missing committed epoch archive: {path}')
        ck = saved if epoch == through else load_checkpoint(path)
        _check_owner(ck, parent_identity)
        row = ck['epoch_metrics']
        if (ck['completed_epoch'] != epoch or ck['next_epoch'] != epoch+1 or
                row['epoch'] != epoch or row['stage'] != identity['stage'] or row['steps'] != ck['step']):
            raise ValueError(f'Inconsistent epoch archive: {path}')
        if epoch == best_epoch and (ck['best'] != saved['best'] or row['validation_l1'] != saved['best']):
            raise ValueError('Historical best archive lineage mismatch')
        summaries[epoch] = row
        if source != dest:
            atomic_save(archive_path(dest_ck, epoch), {**ck, 'run_id': identity['run_id']})
    if history:
        _repair_alias(archive_path(dest_ck, through), dest_ck / 'last.pt')
        _repair_alias(best_source, dest_ck / 'best.pt')
    _repair_logs(source, dest, summaries, through)
    atomic_json(dest / 'run_identity.json', identity)
    atomic_json(dest / 'status.json', {'stage': saved['stage'], 'completed_epoch': through,
                                       'complete': saved['stage_complete']})


def restore_history(source, run_dir, saved, identity):
    if saved is None:
        _repair_logs(source, run_dir, {}, 0)
        return identity
    if source == run_dir:
        _restore_history(source, run_dir, saved, identity, identity)
        return identity
    child = {**identity, 'run_id': str(uuid.uuid4()), 'parent_run_id': identity['run_id'],
             'parent_run': str(source), 'fork_epoch': saved['completed_epoch']}
    # A failure while copying earlier epochs cannot expose half of a new lineage.
    with tempfile.TemporaryDirectory(dir=run_dir.parent, prefix='.rstar-fork-') as temporary:
        staged = Path(temporary) / 'run'
        staged.mkdir()
        _restore_history(source, staged, saved, identity, child)
        if run_dir.exists():
            run_dir.rmdir()  # succeeds only if the destination remains empty
        os.rename(staged, run_dir)
    return child
