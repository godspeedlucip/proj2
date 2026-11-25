from __future__ import annotations
import copy
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
        'accumulation':1,'clip_grad':None,'temporal_roll':False,'init_weights':None}


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


def validate_train(cfg):
    cfg=merge_strict(train_defaults(),cfg)
    if cfg['model_profile'] not in ('official_v1','pig_ordered_v1'): raise ValueError('Unknown model profile')
    if cfg['loss']['kind'] not in ('public_code','l1','physical_ssim'): raise ValueError('Unknown loss kind')
    if cfg['scheduler'] not in ('multistep','cosine'): raise ValueError('Unknown scheduler')
    if cfg['small_volume'] not in ('error','pad'): raise ValueError('small_volume must be error or pad')
    if cfg['workers']<0 or cfg['threads']<1 or cfg['accumulation']<1: raise ValueError('Invalid worker/thread/accumulation count')
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
    return cfg


def load_config(path=None,kind='train'):
    from .moco.settings import defaults as moco_defaults, validate as validate_moco
    defaults={'train':train_defaults,'simulation':simulation_defaults,'metrics':metric_defaults,'moco':moco_defaults}[kind]()
    data=None if path is None else yaml.safe_load(Path(path).read_text(encoding='utf-8-sig'))
    if data is None: data={}
    if not isinstance(data,dict): raise ValueError('Configuration must be a mapping')
    cfg=merge_strict(defaults,data)
    if kind=='moco':
        return validate_moco(cfg, Path(path).resolve().parent if path else Path.cwd())
    if path:
        base=Path(path).resolve().parent
        for key in ['settings_root','settings_csv','cycle_csv','init_weights']:
            if cfg.get(key): cfg[key]=str(resolve(cfg[key],base))
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
