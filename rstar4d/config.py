from __future__ import annotations
import copy
import math
from pathlib import Path
import yaml
from .utils.io import resolve


def train_defaults():
    return {'seed':2026,'device':'cuda','workers':0,'pin_memory':False,'log_every':10,'threads':4,
        'deterministic':False,'amp':True,'model_profile':'official_v1',
        'epochs':{'1':60,'2':30},'learning_rate':.00015,'milestones':{'1':[20,30,40],'2':[10,20,25]},
        'gamma':.5,'scheduler':'multistep','min_lr':.00001,'betas':[.9,.999],'weight_decay':0.,
        'loss':{'kind':'public_code','ssim_weight':.1,'window':[0.,.05]},
        'normalization':{'scale':.02,'offset':0.},
        'stage1_patch':[1,256,256],'stage2_patches':[[16,128,128],[32,96,96],[64,64,64]],
        'steps_per_epoch':{'1':0,'2':0},'validation_samples_per_case':1,
        'small_volume':'error','min_valid_fraction':.5,'max_attempts':32,
        'accumulation':1,'clip_grad':None,'temporal_roll':False,'init_weights':None,
        'stam':{'mode':'off','train_policy':'frozen','sam_checkpoint':None,
                'warmup_epochs':10,'backbone_lr_ratio':.1,'z_chunk_size':1,'activation_checkpointing':True},
        'ttl':{'mode':'off','z_chunk_size':0},
        'stage2_start_epoch':1,'stage2_include_initial':False}


def simulation_defaults():
    return {'profile':'pig_tigre_v1','settings_root':None,'settings_csv':None,'cycle_csv':None,
        'calibration':{'mode':'none','csv':None,'root':None},'settings':{},'view_batch':64,
        'noise':{'kind':'none','photons':50000.},'seed':2026,'prior_mode':'phase_weighted_fdk',
        'save_projections':True,'full_object_fov':False,'limit':None,'case_ids':[]}


def metric_defaults():
    return {'window':[0.,.05],'ssim_window':11,'sigma':1.5,'roi_labels':{},'projection':False}


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


def validate_train(cfg, stage=None):
    explicit_stam = cfg.get('stam', {})
    cfg=merge_strict(train_defaults(),cfg)
    ttl = cfg['ttl']
    if ttl['mode'] not in ('off', 'full', 'local3'):
        raise ValueError('Unknown ttl.mode')
    if type(ttl['z_chunk_size']) is not int or ttl['z_chunk_size'] < 0:
        raise ValueError('ttl.z_chunk_size must be a nonnegative integer')
    ttl_enabled = ttl['mode'] != 'off'
    if ttl_enabled:
        if stage == 1:
            raise ValueError('TTL is supported only in Stage II')
        if 'mode' not in explicit_stam:
            cfg['stam']['mode'] = 'from_parent'
    if cfg['model_profile'] not in ('official_v1','pig_ordered_v1'): raise ValueError('Unknown model profile')
    if cfg['loss']['kind'] not in ('public_code','l1','physical_ssim'): raise ValueError('Unknown loss kind')
    if cfg['scheduler'] not in ('multistep','cosine','poly'): raise ValueError('Unknown scheduler')
    if cfg['small_volume'] not in ('error','pad'): raise ValueError('small_volume must be error or pad')
    if cfg['workers']<0 or cfg['threads']<1 or cfg['accumulation']<1: raise ValueError('Invalid worker/thread/accumulation count')
    if cfg['normalization']['scale']<=0 or cfg['learning_rate']<=0: raise ValueError('scale and learning_rate must be positive')
    if not 0<cfg['min_valid_fraction']<=1 or cfg['max_attempts']<1: raise ValueError('Invalid validity sampling limits')
    if cfg['validation_samples_per_case']<1: raise ValueError('validation_samples_per_case must be positive')
    if len(cfg['betas'])!=2 or any(not 0<=v<1 for v in cfg['betas']): raise ValueError('Invalid Adam betas')
    for stage_key in ['1','2']:
        if cfg['epochs'][stage_key]<1 or cfg['steps_per_epoch'][stage_key]<0: raise ValueError('Invalid epoch count')
        m=cfg['milestones'][stage_key]
        if m!=sorted(set(m)) or any(v<1 for v in m): raise ValueError('Milestones must be unique increasing positive integers')
    if cfg['stage1_patch'][0]!=1: raise ValueError('Stage I patch Z must be 1')
    for patch in [cfg['stage1_patch']]+cfg['stage2_patches']:
        if len(patch)!=3 or any(not isinstance(v,int) or v<1 for v in patch): raise ValueError('Invalid patch shape')
    if not cfg['stage2_patches']: raise ValueError('No Stage II patches')
    stam = cfg['stam']
    allowed_modes = ('off', 'temporal_only', 'joint') + (('from_parent',) if ttl_enabled else ())
    if stam['mode'] not in allowed_modes:
        raise ValueError('Unknown stam.mode')
    if stam['mode'] != 'off' and stage == 1:
        raise ValueError('STAM is supported only in Stage II')
    if stam['train_policy'] not in ('frozen', 'joint', 'joint_after_warmup'):
        raise ValueError('Unknown stam.train_policy')
    if stam['sam_checkpoint'] is not None and not isinstance(stam['sam_checkpoint'], str):
        raise ValueError('stam.sam_checkpoint must be a path')
    if type(stam['z_chunk_size']) is not int or stam['z_chunk_size'] < 0:
        raise ValueError('stam.z_chunk_size must be a nonnegative integer')
    if type(stam['activation_checkpointing']) is not bool:
        raise ValueError('stam.activation_checkpointing must be a boolean')
    if ttl_enabled:
        if stam['sam_checkpoint'] or cfg['init_weights']:
            raise ValueError('TTL needs Stage II parent weights from --init-from-stage2 or --init-from-stam')
        stam['train_policy'] = 'frozen'
    if stam['mode'] != 'off' and stam['train_policy'] != 'frozen':
        ratio = stam['backbone_lr_ratio']
        if type(ratio) not in (int, float) or not math.isfinite(ratio) or ratio <= 0:
            raise ValueError('stam.backbone_lr_ratio must be finite and positive')
    if stam['mode'] != 'off' and stam['train_policy'] == 'joint_after_warmup':
        warmup = stam['warmup_epochs']
        if type(warmup) is not int or not 1 <= warmup < cfg['epochs']['2']:
            raise ValueError('stam.warmup_epochs must be between 1 and Stage II epochs minus 1')
    else:
        stam['warmup_epochs'] = 0
    start = cfg['stage2_start_epoch']
    if type(start) is not int or not 1 <= start <= cfg['epochs']['2']:
        raise ValueError('stage2_start_epoch must be inside the Stage II epoch budget')
    if type(cfg['stage2_include_initial']) is not bool:
        raise ValueError('stage2_include_initial must be a boolean')
    if ttl_enabled and start != 1:
        raise ValueError('TTL training must start at Stage II epoch 1')
    if ttl_enabled:
        cfg['stage2_include_initial'] = True
    elif (start != 1 or cfg['stage2_include_initial']) and stam['mode'] != 'off':
        raise ValueError('Stage II control options require stam.mode=off')
    return cfg


def load_config(path=None,kind='train'):
    defaults={'train':train_defaults,'simulation':simulation_defaults,'metrics':metric_defaults}[kind]()
    data={} if path is None else yaml.safe_load(Path(path).read_text(encoding='utf-8-sig')) or {}
    cfg=validate_train(data) if kind=='train' else merge_strict(defaults,data)
    if path:
        base=Path(path).resolve().parent
        for key in ['settings_root','settings_csv','cycle_csv','init_weights']:
            if cfg.get(key): cfg[key]=str(resolve(cfg[key],base))
        if kind == 'train' and isinstance(cfg['stam']['sam_checkpoint'], str) and cfg['stam']['sam_checkpoint']:
            cfg['stam']['sam_checkpoint'] = str(resolve(cfg['stam']['sam_checkpoint'], base))
        if 'calibration' in cfg:
            for key in ['csv','root']:
                if cfg['calibration'].get(key): cfg['calibration'][key]=str(resolve(cfg['calibration'][key],base))
    if kind=='train': return validate_train(cfg)
    if kind=='simulation':
        if cfg['calibration']['mode'] not in ('none','post_fdk'): raise ValueError('Calibration must be none or post_fdk')
        if cfg['calibration']['mode']=='post_fdk' and not (cfg['calibration']['csv'] or cfg['calibration']['root']):
            raise ValueError('post_fdk requires calibration csv or root')
        if cfg['view_batch']<1 or cfg['noise']['kind'] not in ('none','poisson'): raise ValueError('Invalid simulation config')
        if cfg['noise']['photons']<=0: raise ValueError('photons must be positive')
        if cfg['prior_mode'] not in ('all_views_fdk','phase_weighted_fdk'): raise ValueError('Invalid prior_mode')
    return cfg
