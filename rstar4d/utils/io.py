from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import importlib.metadata


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def atomic_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False); f.flush(); os.fsync(f.fileno())
    os.replace(temp,path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def atomic_jsonl(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('w',encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        f.flush(); os.fsync(f.fileno())
    os.replace(temp,path)


def append_jsonl(path,row):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n'); f.flush()


def resolve(path,base):
    p=Path(path).expanduser()
    return (p if p.is_absolute() else Path(base)/p).resolve()


def safe_id(value):
    import re
    value=str(value)
    if not value or not re.fullmatch(r'[\w.\-]+',value) or value in ('.','..'):
        raise ValueError(f'Unsafe case/run identifier: {value!r}')
    return value


def environment():
    import torch
    versions={}
    for name in ['torch','numpy','scipy','h5py','nibabel','tifffile','pytorch-msssim','pytigre']:
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]=None
    return {'python':platform.python_version(),'platform':platform.platform(),'versions':versions,
            'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def code_fingerprint():
    root=Path(__file__).resolve().parents[1]
    return canonical_hash({str(p.relative_to(root)):sha256(p) for p in sorted(root.rglob('*.py'))})
