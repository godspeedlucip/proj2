from __future__ import annotations
import os
from pathlib import Path
import torch
from ..config import validate_cycn


def cycn_settings(checkpoint):
    """Read the saved architecture, including the distinct center-only graph."""
    value = checkpoint.get('cycn')
    if value is None:
        return validate_cycn(None)
    if value.get('version') != 1:
        raise ValueError('Unsupported CycN structure version')
    settings = {key: item for key, item in value.items() if key != 'version'}
    if set(settings) != set(validate_cycn(None)):
        raise ValueError('Incomplete CycN structure metadata')
    return validate_cycn(settings)


def basic_types(value):
    """Keep checkpoints compatible with weights_only loading, including LR Counter state."""
    if torch.is_tensor(value): return value
    if isinstance(value,dict): return {basic_types(k):basic_types(v) for k,v in value.items()}
    if isinstance(value,tuple): return tuple(basic_types(v) for v in value)
    if isinstance(value,list): return [basic_types(v) for v in value]
    if value is None or type(value) in (int,float,bool,str): return value
    if isinstance(value,str): return str(value)
    raise TypeError(f'Unsupported checkpoint value type: {type(value)}')


def atomic_save(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as f:
        torch.save(basic_types(payload),f); f.flush(); os.fsync(f.fileno())
    os.replace(temporary,path)


def load_checkpoint(path,require_training=True):
    value=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(value,dict) or value.get('schema_version')!=1 or 'state_dict' not in value:
        raise ValueError('Expected a versioned RSTAR4D checkpoint; raw public weights are warm starts, not resume checkpoints')
    if require_training and value.get('checkpoint_kind')!='training': raise ValueError('Weights-only checkpoint cannot resume training')
    if require_training:
        required={'optimizer','scheduler','scaler','rng','config','data_snapshot','stage','completed_epoch','next_epoch','signature'}
        if not required.issubset(value): raise ValueError('Incomplete training checkpoint')
    return value
