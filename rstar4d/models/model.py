import torch
import torch.nn.functional as F
from torch import nn
from . import official_stage1, official_stage2


def is_z_parameter(name):
    return '.conv2.' in name or '.conv2_2.' in name


def create_model(stage, profile='official_v1', stam=None, ttl=None):
    if stage not in (1,2): raise ValueError('stage must be 1 or 2')
    if profile not in ('official_v1','pig_ordered_v1'): raise ValueError('Unknown model profile')
    model=(official_stage1 if stage==1 else official_stage2).myUNet()
    model.stage=stage; model.profile=profile
    if profile=='pig_ordered_v1':
        for module in model.modules():
            if isinstance(module,nn.Conv3d) and module.padding_mode=='circular': module.padding_mode='reflect'
    if stage==1:
        for name,p in model.named_parameters():
            if is_z_parameter(name): p.requires_grad_(False)
    attach_stam(model, stam)
    attach_ttl(model, ttl)
    return model


def attach_stam(model, config, load_sam=False):
    if not config or config['mode'] == 'off':
        return None
    if model.stage != 2:
        raise ValueError('STAM is supported only in Stage II')
    from .stam import STAMJointBlock
    block = STAMJointBlock(config['mode'], config.get('z_chunk_size', 1), config.get('activation_checkpointing', True))
    source = block.sam.load_pretrained(config.get('sam_checkpoint')) if load_sam else None
    model.stam = block
    return source


def attach_ttl(model, config):
    if not config or config['mode'] == 'off':
        return None
    if model.stage != 2 or getattr(model, 'stam', None) is None:
        raise ValueError('TTL requires a Stage II model with STAM')
    if model.ttl is not None:
        raise ValueError('This model already has TTL')
    from .ttl import TemporalTransformerSkip
    model.ttl = TemporalTransformerSkip(config['mode'], config.get('z_chunk_size', 0))
    return model.ttl


def model_from_checkpoint(value, stage=None, profile=None):
    stage = value.get('stage', stage)
    profile = value.get('model_profile', profile or 'official_v1')
    metadata = value.get('stam')
    structure = None
    if metadata is not None:
        from .stam import architecture
        structure = metadata.get('architecture', {})
        mode = structure.get('mode')
        if mode not in ('joint', 'temporal_only') or structure != architecture(mode):
            raise ValueError('Unsupported STAM architecture in checkpoint')
        execution = value.get('config', {}).get('stam', {})
        structure = {**execution, **structure}
    model = create_model(stage, profile, structure)
    model.load_state_dict(value.get('state_dict', value), strict=True)
    return model


def initialize_stage2(model,state):
    if model.stage!=2: raise ValueError('Stage II model required')
    model.load_state_dict(state,strict=True)
    for name,p in model.named_parameters():
        if is_z_parameter(name):
            if name in [f'conv{i}.conv2.weight' for i in range(1,5)]: nn.init.dirac_(p)
            else: nn.init.zeros_(p)
            p.requires_grad_(True)


def forward_volume(model,x,temporal_topology=None):
    if x.ndim!=6 or x.shape[1]!=2: raise ValueError('Expected [B,2,T,Z,Y,X]')
    if x.shape[2]<2: raise ValueError('At least two phases are required')
    b,c,t,z,y,w=x.shape
    pad_z=(-z)%2 if model.stage==2 else 0
    pad=(0,(-w)%8,0,(-y)%8,0,pad_z)
    if any(pad):
        x=F.pad(x.reshape(b,c*t,z,y,w),pad,mode='replicate').reshape(b,c,t,z+pad_z,y+pad[3],w+pad[1])
    x = x.contiguous()
    out = model(x, temporal_topology=temporal_topology) if getattr(model, 'ttl', None) is not None else model(x)
    return out[:,:,:,:z,:y,:w]


def load_weights(path,stage,profile='official_v1',device='cpu'):
    state=torch.load(path,map_location='cpu',weights_only=True)
    return model_from_checkpoint(state, stage, profile).to(device)
