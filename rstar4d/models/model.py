import torch
import torch.nn.functional as F
from torch import nn
from . import official_stage1, official_stage2


def is_z_parameter(name):
    return '.conv2.' in name or '.conv2_2.' in name


def create_model(stage,profile='official_v1'):
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
    if isinstance(state,dict) and 'state_dict' in state: state=state['state_dict']
    model=create_model(stage,profile); model.load_state_dict(state,strict=True)
    return model.to(device)
