import torch
import torch.nn.functional as F
from torch import nn
from ..config import merge_strict, train_defaults, validate_pida
from . import official_stage1, official_stage2
from .pida_da import PIDADeformableAttention, PIDAInputConv, architecture as pida_architecture


def is_z_parameter(name):
    return '.conv2.' in name or '.conv2_2.' in name


def create_model(stage, profile='official_v1', stam=None, pida_da=None):
    if stage not in (1,2): raise ValueError('stage must be 1 or 2')
    if profile not in ('official_v1','pig_ordered_v1'): raise ValueError('Unknown model profile')
    pida_da = merge_strict(train_defaults()['pida_da'], pida_da or {}, 'pida_da.')
    validate_pida(pida_da, (stam or {}).get('mode', 'off'), stage)
    model=(official_stage1 if stage==1 else official_stage2).myUNet()
    model.stage=stage; model.profile=profile
    if profile=='pig_ordered_v1':
        for module in model.modules():
            if isinstance(module,nn.Conv3d) and module.padding_mode=='circular': module.padding_mode='reflect'
    if stage==1:
        for name,p in model.named_parameters():
            if is_z_parameter(name): p.requires_grad_(False)
    attach_stam(model, stam)
    attach_pida(model, pida_da)
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


def attach_pida(model, config):
    if not config or not config.get('enabled'):
        return
    config = merge_strict(train_defaults()['pida_da'], config, 'pida_da.')
    validate_pida(config, getattr(getattr(model, 'stam', None), 'mode', 'off'), model.stage)
    if model.pida_da is not None:
        raise ValueError('Model already has PIDA DA')
    arch = pida_architecture(config)
    reference = model.conv5_2.conv1.weight
    model.pida_da = PIDADeformableAttention(
        arch['channels'], config['query_chunk_size'], config['plane_batch_size'], arch['fusion']
    ).to(device=reference.device, dtype=reference.dtype)
    if arch['fusion'] == 'residual':
        model.requires_grad_(False)
        model.pida_da.requires_grad_(True)
    else:
        for name in ('conv1', 'conv2', 'conv3'):
            conv = getattr(model.conv5_2, name)
            setattr(model.conv5_2, name, PIDAInputConv(conv, arch['channels']))
        model.conv5_2.in_ch += arch['channels']
    model.train(model.training)


def _pida_from_checkpoint(value):
    metadata = value.get('pida_da', {'enabled': False})
    if not isinstance(metadata, dict):
        raise ValueError('Invalid pida_da architecture in checkpoint')
    enabled = metadata.get('enabled')
    train_policy = 'pida_only' if metadata.get('fusion') == 'residual' else 'stam'
    saved_config = value.get('config', {}).get('pida_da', {})
    config = merge_strict(train_defaults()['pida_da'], saved_config, 'config.pida_da.')
    expected = {'enabled': enabled, 'train_policy': train_policy}
    if any(key in saved_config and config[key] != val for key, val in expected.items()):
        raise ValueError('Checkpoint pida_da architecture and config disagree')
    config.update(expected)
    if type(enabled) is not bool or metadata != pida_architecture(config):
        raise ValueError('Unsupported pida_da architecture in checkpoint')
    return config


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
    model = create_model(stage, profile, structure, _pida_from_checkpoint(value))
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


def forward_volume(model,x):
    if x.ndim!=6 or x.shape[1]!=2: raise ValueError('Expected [B,2,T,Z,Y,X]')
    if x.shape[2]<2: raise ValueError('At least two phases are required')
    b,c,t,z,y,w=x.shape
    pad_z=(-z)%2 if model.stage==2 else 0
    pad=(0,(-w)%8,0,(-y)%8,0,pad_z)
    if any(pad):
        x=F.pad(x.reshape(b,c*t,z,y,w),pad,mode='replicate').reshape(b,c,t,z+pad_z,y+pad[3],w+pad[1])
    out=model(x.contiguous())
    return out[:,:,:,:z,:y,:w]


def load_weights(path,stage,profile='official_v1',device='cpu'):
    state=torch.load(path,map_location='cpu',weights_only=True)
    return model_from_checkpoint(state, stage, profile).to(device)
