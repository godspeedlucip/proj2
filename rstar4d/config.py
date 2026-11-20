from __future__ import annotations
import copy
import math
from pathlib import Path
import yaml
from .utils.io import resolve
from .data.context import STAGE2_ALIGNMENT, X2_STRIDE


def train_defaults():
    return {'seed':2026,'device':'cuda','workers':0,'pin_memory':False,'log_every':10,'threads':4,
        'deterministic':False,'amp':True,'tensorboard':True,'model_profile':'official_v1',
        'tensorboard_real':{'manifest':None,'phase_label':None,'axial_slice':None},
        'epochs':{'1':60,'2':30},'learning_rate':.00015,'milestones':{'1':[20,30,40],'2':[10,20,25]},
        'gamma':.5,'scheduler':'multistep','min_lr':.00001,'betas':[.9,.999],'weight_decay':0.,
        'loss':{'kind':'public_code','ssim_weight':.1,'window':[0.,.05]},
        'normalization':{'scale':.02,'offset':0.},
        'stage1_patch':[1,256,256],'stage2_patches':[[16,128,128],[32,96,96],[64,64,64]],
        'steps_per_epoch':{'1':0,'2':0},'validation_samples_per_case':1,
        'small_volume':'error','min_valid_fraction':.5,'max_attempts':32,
        'accumulation':1,'micro_batch_size':1,'clip_grad':None,'temporal_roll':False,'init_weights':None,
        'stage2_crop_mode':'legacy',
        'stage2_context':{'lncc_radius_vox':[2,2,2], 'support_margin_vox':[31,36,36],
                          'edge_threshold':1e-4, 'variance_threshold':1e-8, 'match_chunk_size':32768,
                          'activation_checkpointing':True},
        'extensions':{
            'motion':{'mode':'off','max_displacement_vox':[8,8,8],'struct_weight':.05,
                      'smooth_weight':.01,'surface_weight':0.,'surface_truncation_vox':5.},
            'prior':{'mode':'off'},
            'boundary':{'enabled':False,'half_width_vox':2.,'structure_weights':{'heart_outer':1.,'lv':1.},
                        'ramp_epochs':5,'weight':.5,'grad_weight_vox':0.}}}


def simulation_defaults():
    return {'profile':'pig_tigre_v1','settings_root':None,'settings_csv':None,'cycle_csv':None,
        'calibration':{'mode':'none','csv':None,'root':None},'settings':{},'view_batch':64,
        'noise':{'kind':'none','photons':50000.},'seed':2026,'prior_mode':'phase_weighted_fdk',
        'save_projections':True,'full_object_fov':False,'limit':None,'case_ids':[]}


def metric_defaults():
    return {'window':[0.,.05],'ssim_window':11,'sigma':1.5,'roi_labels':{},'projection':False,
            'contours':{'surface_dice_tolerance_vox':{'heart_outer':None,'lv':None},'half_width_vox':2.}}


def merge_strict(defaults,values,path=''):
    result=copy.deepcopy(defaults)
    for key,value in values.items():
        key=str(key)
        if key not in defaults: raise ValueError(f'Unknown configuration field: {path}{key}')
        if isinstance(defaults[key],dict) and defaults[key]:
            if not isinstance(value,dict): raise ValueError(f'{path}{key} must be a mapping')
            result[key]=merge_strict(defaults[key],value,path+key+'.')
        else: result[key]=copy.deepcopy(value)
    return result


def validate_train(cfg):
    cfg=merge_strict(train_defaults(),cfg)
    if type(cfg['tensorboard']) is not bool:
        raise ValueError('tensorboard must be a boolean')
    preview = cfg['tensorboard_real']
    if preview['manifest'] is not None and (not isinstance(preview['manifest'],str) or not preview['manifest'].strip()):
        raise ValueError('tensorboard_real.manifest must be a nonempty path')
    if preview['phase_label'] is not None and type(preview['phase_label']) is not int:
        raise ValueError('tensorboard_real.phase_label must be an integer')
    index = preview['axial_slice']
    if index is not None and (type(index) is not int or index<0):
        raise ValueError('tensorboard_real.axial_slice must be a nonnegative integer')
    if cfg['model_profile'] not in ('official_v1','pig_ordered_v1'): raise ValueError('Unknown model profile')
    if cfg['loss']['kind'] not in ('public_code','l1','physical_ssim'): raise ValueError('Unknown loss kind')
    if cfg['scheduler'] not in ('multistep','cosine'): raise ValueError('Unknown scheduler')
    if cfg['small_volume'] not in ('error','pad'): raise ValueError('small_volume must be error or pad')
    if cfg['workers']<0 or cfg['threads']<1 or cfg['accumulation']<1: raise ValueError('Invalid worker/thread/accumulation count')
    micro_batch = cfg['micro_batch_size']
    if type(micro_batch) is not int or micro_batch not in (1,2) or cfg['accumulation'] % micro_batch:
        raise ValueError('micro_batch_size must be 1 or 2 and divide accumulation')
    if cfg['normalization']['scale']<=0 or cfg['learning_rate']<=0: raise ValueError('scale and learning_rate must be positive')
    if not 0<cfg['min_valid_fraction']<=1 or cfg['max_attempts']<1: raise ValueError('Invalid validity sampling limits')
    if cfg['validation_samples_per_case']<1: raise ValueError('validation_samples_per_case must be positive')
    if len(cfg['betas'])!=2 or any(not 0<=v<1 for v in cfg['betas']): raise ValueError('Invalid Adam betas')
    for stage in ['1','2']:
        if cfg['epochs'][stage]<1 or cfg['steps_per_epoch'][stage]<0: raise ValueError('Invalid epoch count')
        m=cfg['milestones'][stage]
        if m!=sorted(set(m)) or any(v<1 for v in m): raise ValueError('Milestones must be unique increasing positive integers')
    if cfg['stage1_patch'][0]!=1: raise ValueError('Stage I patch Z must be 1')
    for patch in [cfg['stage1_patch']]+cfg['stage2_patches']:
        if len(patch)!=3 or any(not isinstance(v,int) or v<1 for v in patch): raise ValueError('Invalid patch shape')
    if not cfg['stage2_patches']: raise ValueError('No Stage II patches')
    _validate_stage2(cfg)
    return cfg


def _number(value, name, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f'Invalid {name}: {value}')


def _validate_stage2(config):
    motion, prior, boundary = (config['extensions'][key] for key in ('motion','prior','boundary'))
    recipe = config['stage2_context']
    if config['stage2_crop_mode'] not in ('legacy','core_with_context'):
        raise ValueError('Invalid stage2_crop_mode')
    if type(recipe['activation_checkpointing']) is not bool:
        raise ValueError('activation_checkpointing must be boolean')
    if motion['mode'] not in ('off','identity','learned') or prior['mode'] not in ('off','image_only','geometry'):
        raise ValueError('Invalid motion or prior mode (quote "off" in YAML)')
    for key in ('struct_weight','smooth_weight','surface_weight','surface_truncation_vox'):
        _number(motion[key], key, positive=key == 'surface_truncation_vox')
    if motion['surface_weight'] and motion['mode'] != 'learned':
        raise ValueError('Surface loss requires learned motion')
    if type(boundary['enabled']) is not bool:
        raise ValueError('boundary.enabled must be boolean')
    for key in ('weight','grad_weight_vox','half_width_vox'):
        _number(boundary[key], key, positive=key == 'half_width_vox')
    for key, value in boundary['structure_weights'].items():
        _number(value, key, positive=True)
    if boundary['enabled'] and boundary['weight'] == 0:
        raise ValueError('Enabled boundary loss requires a positive weight')
    for name, values in [('max_displacement_vox', motion['max_displacement_vox']),
                         ('lncc_radius_vox', recipe['lncc_radius_vox']),
                         ('support_margin_vox', recipe['support_margin_vox'])]:
        if not isinstance(values, (tuple,list)) or len(values) != 3:
            raise ValueError(f'{name} must contain three ZYX values')
        for value in values:
            _number(value, name, positive=name != 'support_margin_vox')
            if name != 'max_displacement_vox' and type(value) is not int:
                raise ValueError(f'{name} requires integer voxel counts')
    for name, value in [('ramp_epochs', boundary['ramp_epochs']), ('match_chunk_size', recipe['match_chunk_size'])]:
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    for name in ('edge_threshold','variance_threshold'):
        _number(recipe[name], name, positive=True)


def resolve_stage2(config: dict, stage: int) -> dict:
    """Expose effective weights without modifying the preset shared by the factorial runs."""
    motion, prior, boundary = (copy.deepcopy(config['extensions'][key]) for key in ('motion','prior','boundary'))
    if stage == 1:
        motion['mode'] = prior['mode'] = 'off'
        boundary['enabled'] = False
    if motion['mode'] != 'learned':
        for name in ('struct_weight','smooth_weight','surface_weight'):
            motion[name] = 0.
    if not boundary['enabled']:
        boundary['weight'] = boundary['grad_weight_vox'] = 0.
    recipe = {**copy.deepcopy(config['stage2_context']), 'max_displacement_vox': motion['max_displacement_vox'],
              'alignment': list(STAGE2_ALIGNMENT), 'feature_stride': list(X2_STRIDE),
              'core_shapes': copy.deepcopy(config['stage2_patches'])}
    return {'crop_mode': config['stage2_crop_mode'] if stage == 2 else 'legacy',
            'motion': motion, 'prior': prior, 'boundary': boundary, 'recipe': recipe,
            'version': 'cardiac_v1', 'distance_unit': 'output_voxel',
            'motion_seed': config['seed'] + 31013, 'prior_seed': config['seed'] + 67003,
            'architecture': {'feature_channels':64,'descriptor_channels':16,'search_window':3,
                             'prior_temperature':.2,'geometry_encoding':'angles_v1'}}


def load_config(path=None,kind='train'):
    defaults={'train':train_defaults,'simulation':simulation_defaults,'metrics':metric_defaults}[kind]()
    data={} if path is None else yaml.safe_load(Path(path).read_text(encoding='utf-8-sig')) or {}
    cfg=merge_strict(defaults,data)
    if path:
        base=Path(path).resolve().parent
        for key in ['settings_root','settings_csv','cycle_csv','init_weights']:
            if cfg.get(key): cfg[key]=str(resolve(cfg[key],base))
        if 'calibration' in cfg:
            for key in ['csv','root']:
                if cfg['calibration'].get(key): cfg['calibration'][key]=str(resolve(cfg['calibration'][key],base))
        if kind=='train' and cfg['tensorboard_real']['manifest']:
            cfg['tensorboard_real']['manifest']=str(resolve(cfg['tensorboard_real']['manifest'],base))
    if kind=='train': return validate_train(cfg)
    if kind=='simulation':
        if cfg['calibration']['mode'] not in ('none','post_fdk'): raise ValueError('Calibration must be none or post_fdk')
        if cfg['calibration']['mode']=='post_fdk' and not (cfg['calibration']['csv'] or cfg['calibration']['root']):
            raise ValueError('post_fdk requires calibration csv or root')
        if cfg['view_batch']<1 or cfg['noise']['kind'] not in ('none','poisson'): raise ValueError('Invalid simulation config')
        if cfg['noise']['photons']<=0: raise ValueError('photons must be positive')
        if cfg['prior_mode'] not in ('all_views_fdk','phase_weighted_fdk'): raise ValueError('Invalid prior_mode')
    return cfg
