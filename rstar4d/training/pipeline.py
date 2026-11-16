from pathlib import Path
from .trainer import train
from ..utils.io import atomic_json


def pipeline(config,manifest,split,run_dir):
    run_dir=Path(run_dir).resolve(); run_dir.mkdir(parents=True,exist_ok=True)
    stage1=run_dir/'stage1'; last1=stage1/'checkpoints/last.pt'
    first=train(config,manifest,split,stage1,1,resume=last1 if last1.exists() else None)
    if not first['complete']: return first
    atomic_json(run_dir/'pipeline.json',{'stage1_complete':True,'stage2_initialized':False,'stage1_best':first['checkpoint']})
    stage2=run_dir/'stage2'; last2=stage2/'checkpoints/last.pt'
    second=train(config,manifest,split,stage2,2,resume=last2 if last2.exists() else None,
        init=None if last2.exists() else {'path':first['checkpoint'],'source_stage':1})
    atomic_json(run_dir/'pipeline.json',{'stage1_complete':True,'stage2_initialized':True,'stage2_complete':second['complete'],
        'stage1_best':first['checkpoint'],'stage2_best':second['checkpoint']})
    return second
