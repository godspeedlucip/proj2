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


def attach_stage2_extensions(model, resolved):
    if model.stage != 2 or resolved['version'] != 'cardiac_v1':
        raise ValueError('Unsupported model stage or Stage II extension version')
    from .motion import AdjacentMotionFusion
    from .prior_fusion import GeometryPriorFusion
    motion,prior = resolved['motion'],resolved['prior']
    model.activation_checkpointing = (resolved['crop_mode'] in ('core_with_context','fixed_input')
                                      and resolved['recipe'].get('activation_checkpointing',False))
    device = next(model.parameters()).device
    if motion['mode'] != 'off':
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(resolved['motion_seed'])
            model.motion = AdjacentMotionFusion(motion['mode'],motion['max_displacement_vox']).to(device)
    if prior['mode'] != 'off':
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(resolved['prior_seed'])
            model.prior_fusion = GeometryPriorFusion(prior['mode'],resolved['recipe']['match_chunk_size']).to(device)


def model_from_checkpoint(checkpoint):
    model = create_model(checkpoint['stage'],checkpoint.get('model_profile','official_v1'))
    if model.stage == 2 and checkpoint.get('stage2_resolved') is not None:
        attach_stage2_extensions(model,checkpoint['stage2_resolved'])
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    return model


def forward_volume(model,x,*,context=None,return_aux=False):
    if x.ndim!=6 or x.shape[1]!=2: raise ValueError('Expected [B,2,T,Z,Y,X]')
    if x.shape[2]<2: raise ValueError('At least two phases are required')
    b,c,t,z,y,w=x.shape
    pad_z=(-z)%2 if model.stage==2 else 0
    pad=(0,(-w)%8,0,(-y)%8,0,pad_z)
    if any(pad):
        x=F.pad(x.reshape(b,c*t,z,y,w),pad,mode='replicate').reshape(b,c,t,z+pad_z,y+pad[3],w+pad[1])
    if context is not None and any(pad):
        context = {**context, **{name:F.pad(context[name],pad,value=False)
                                for name in ('source_fov','prior_fov') if name in context}}
    requires_context = getattr(model,'motion',None) is not None or getattr(model,'prior_fusion',None) is not None
    if requires_context and context is None:
        raise ValueError('Stage II branch requires sampling context')
    if return_aux and model.stage == 2:
        out,aux = model(x.contiguous(),context=context,return_aux=True)
        if 'flow_vox' in aux:
            feature_slices = tuple(slice(0,(n+s-1)//s) for n,s in zip((z,y,w),aux['stride']))
            for name in ('flow_vox','valid_samples'):
                aux[name] = aux[name][(...,*feature_slices)]
        return out[:,:,:,:z,:y,:w],aux
    out = model(x.contiguous(),context=context) if requires_context else model(x.contiguous())
    if return_aux:
        return out[:,:,:,:z,:y,:w],{}
    return out[:,:,:,:z,:y,:w]


def load_weights(path,stage,profile='official_v1',device='cpu'):
    state=torch.load(path,map_location='cpu',weights_only=True)
    if isinstance(state,dict) and 'state_dict' in state:
        if state.get('stage',stage) != stage or state.get('model_profile',profile) != profile:
            raise ValueError('Requested model does not match checkpoint')
        return model_from_checkpoint({'stage':stage,'model_profile':profile,**state}).to(device)
    model=create_model(stage,profile); model.load_state_dict(state,strict=True)
    return model.to(device)
