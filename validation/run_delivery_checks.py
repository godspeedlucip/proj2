"""Persist a small, real-CUDA acceptance run. Run from the repository root.

python -m validation.run_delivery_checks --out runs/my_acceptance
This is a software smoke run, not an accuracy experiment or a replacement for animal splits.
"""
import argparse
import contextlib
import gc
import json
from pathlib import Path
import torch
from rstar4d.cli import main
from rstar4d.utils.io import read_jsonl, atomic_jsonl, atomic_json, code_fingerprint


def real_patch_check(manifest,out):
    from rstar4d.config import load_config
    from rstar4d.data.contracts import read_manifest, validate_case
    from rstar4d.data.sampling import PatchDataset
    from rstar4d.models import create_model, forward_volume, initialize_stage2
    from rstar4d.training.losses import compute_loss
    rows=read_manifest(manifest)
    if len(rows)!=1: raise ValueError('Real patch check requires exactly one case')
    info=validate_case(rows[0]['path'],full=False)
    config=load_config('configs/train/pig_local_16gb.yaml'); config['stage2_patches']=[[16,128,128]]
    records=[]; previous=None
    for stage in [1,2]:
        sample=PatchDataset(rows,config,stage,1,1)[0]
        model=create_model(stage).cuda()
        if stage==1:
            model.load_state_dict(torch.load('RSTAR/githubcode/checkpoints/checkpoints_Tetris1.pth',map_location='cpu',weights_only=True),strict=True)
        else: initialize_stage2(model,previous)
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=config['learning_rate'])
        scaler=torch.cuda.amp.GradScaler(); before=model.conv1.conv1.weight.detach().clone()
        x=sample['input'][None].cuda(); target=sample['target'][None].cuda(); valid=sample['valid'][None].cuda()
        torch.cuda.reset_peak_memory_stats(); losses=[]; updates=0; scale_history=[]
        for draw in range(32):
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(): prediction=forward_volume(model,x)
            loss=compute_loss(prediction,target,valid,stage,config['loss'])
            if not torch.isfinite(loss['total']): raise RuntimeError('Nonfinite real patch loss')
            scaler.scale(loss['total']).backward(); scaler.unscale_(optimizer)
            finite=all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            scale_before=scaler.get_scale(); scaler.step(optimizer); scaler.update(); updates+=int(finite)
            scale_history.append({'draw':draw,'scale':scale_before,'finite_gradient':bool(finite)})
            print(json.dumps({'real_patch_stage':stage,'draw':draw,'scale':scale_before,'finite_gradient':bool(finite),'l1':float(loss['l1'].detach())}),flush=True)
            losses.append(float(loss['l1'].detach()))
            if updates>=2: break
        if updates<2 or torch.equal(before,model.conv1.conv1.weight):
            atomic_json(out/'real_patch_error.json',{'stage':stage,'updates':updates,'scales':scale_history,'l1':losses})
            raise RuntimeError(f'No real parameter updates in Stage {stage}')
        previous={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        records.append({'stage':stage,'shape':list(x.shape),'origin':sample['origin'].tolist(),
            'valid_fraction':float(valid.float().mean()),'l1_normalized':losses,'optimizer_updates':updates,'scale_history':scale_history,
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3})
        del model,optimizer,scaler,x,target,valid,prediction,loss,before
        gc.collect(); torch.cuda.empty_cache()
    result={'case_id':rows[0]['case_id'],'prepared_shape':info['shape'],'domain':info['metadata']['intensity_domain'],
        'purpose':'Single-case execution check, no validation/test or recovery-quality claim','stages':records,
        'code_fingerprint':code_fingerprint()}
    atomic_json(out/'real_patch_check.json',result)


def run(out,real_manifest=None):
    out=Path(out).resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError('Acceptance output must be empty')
    out.mkdir(parents=True,exist_ok=True)
    def command(*args):
        with (out/'cli.log').open('a',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            status=main(list(map(str,args)))
        if status: raise RuntimeError(f'Failed {args[0]}; see {out / "cli.log"}')
        print(json.dumps({'command':args[0],'status':'passed'}),flush=True)
    command('make-phantom','--out',out/'raw','--views',200)
    command('simulate-pig','--manifest',out/'raw/raw_manifest.jsonl','--config',out/'raw/simulation.yaml','--out',out/'prepared')
    for entry in ['prepared','imported']:
        if entry=='imported':
            sources=read_jsonl(out/'prepared/manifest.jsonl')
            for row in sources:
                row['paired_h5']=str(out/'prepared'/row['path'])
                if row.get('projections'): row['projections']=str(out/'prepared'/row['projections'])
            atomic_jsonl(out/'sources.jsonl',sources)
            command('import-paired','--manifest',out/'sources.jsonl','--out',out/entry)
        command('validate-data','--manifest',out/entry/'manifest.jsonl','--split',out/'raw/split.json')
        command('pipeline','--config','configs/train/smoke.yaml','--manifest',out/entry/'manifest.jsonl',
            '--split',out/'raw/split.json','--run-dir',out/('run_'+entry))
        checkpoint=out/('run_'+entry)/'stage2/checkpoints/best.pt'
        command('infer','--checkpoint',checkpoint,'--manifest',out/entry/'manifest.jsonl','--out',out/('pred_'+entry),
            '--tile',8,32,32,'--overlap',4,16,16,'--export','tif','nii')
        command('evaluate','--predictions',out/('pred_'+entry),'--manifest',out/entry/'manifest.jsonl',
            '--out',out/('eval_'+entry),'--projection')
    if real_manifest: real_patch_check(real_manifest,out)
    atomic_json(out/'acceptance_status.json',{'complete':True,'entries':['raw_ct','paired_h5'],
        'code_fingerprint':code_fingerprint(),'real_patch_checked':bool(real_manifest)})


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--out',required=True); parser.add_argument('--real-manifest')
    args=parser.parse_args(); run(args.out,args.real_manifest)
