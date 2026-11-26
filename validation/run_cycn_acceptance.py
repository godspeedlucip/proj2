"""Run a deterministic CLI train/resume/infer/evaluate check with synthetic cases."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from rstar4d.config import load_config
from rstar4d.data.contracts import basic_metadata, write_case
from rstar4d.models import create_model, forward_volume, load_weights
from rstar4d.utils.io import atomic_json, atomic_jsonl, read_json, read_jsonl, sha256


PROJECT = Path(__file__).resolve().parents[1]
PUBLIC_STAGE2 = PROJECT / 'RSTAR/githubcode/checkpoints/checkpoints_Tetris2.pth'


def run_cli(log, *args):
    with log.open('a', encoding='utf-8') as stream:
        command = [str(arg) for arg in args]
        stream.write('\n' + subprocess.list2cmdline(command) + '\n')
        stream.flush()
        subprocess.run([sys.executable, '-m', 'rstar4d', *command], cwd=PROJECT,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def make_fixture(root):
    shape = (5, 25, 33)
    spacing = (1.5, .9, .8)
    affine = np.diag([*spacing, 1.])
    geometry = {'shape_zyx':list(shape), 'voxel_spacing_zyx_mm':list(spacing),
                'voxel_to_scanner_zyx':affine.tolist(),
                'center_zyx_mm':((np.asarray(shape)-1)*spacing/2).tolist()}
    z, y, x = np.indices(shape)
    rng = np.random.default_rng(2026)
    rows, annotations = [], []
    split = {key:[] for key in ('train','val','test')}
    mask_path = root / 'lv_masks.h5'
    with h5py.File(mask_path,'w') as masks_h5:
        for cohort in split:
            for t,topology in [(10,'cyclic'),(11,'ordered_states')]:
                case = f'{cohort}-{topology}'
                split[cohort].append(case)
                labels = [10+i*7 for i in range(t)]
                ed, es = labels[0], labels[t//2]
                radius = 5.5 + 1.25 * np.cos(2*np.pi*np.arange(t)/t)
                lv = np.stack([((z-2)/1.8)**2 + ((y-12)/r)**2 +
                               ((x-16-.6*np.sin(2*np.pi*i/t))/r)**2 <= 1
                               for i,r in enumerate(radius)])
                target = (.007 + .022*lv + .0001*x + .00008*y).astype(np.float32)
                fdk = target + (.003*np.sin(x*.3+y*.2) + rng.normal(0,.001, target.shape)).astype(np.float32)
                path = root / f'{case}.h5'
                meta = basic_metadata(t,spacing,phase_labels=labels,ed_phase_label=ed,es_phase_label=es,
                                      temporal_topology=topology,geometry=geometry,
                                      generation_profile='cycn_synthetic_acceptance')
                write_case(path,fdk,target.mean(0),target,np.ones_like(lv),meta)
                digest=sha256(path)
                rows.append({'case_id':case,'animal_id':case,'path':path.name,'sha256':digest})
                if cohort!='train':
                    for i in (0,t//2,t-1):
                        key=f'{case}-{labels[i]}'
                        masks_h5[key]=lv[i]
                        annotations.append({'case_id':case,'source_sha256':digest,'phase_label':labels[i],
                                            'path':mask_path.name,'dataset':key})
    mask_sha=sha256(mask_path)
    atomic_jsonl(root/'manifest.jsonl',rows)
    atomic_jsonl(root/'annotations.jsonl',[{**row,'sha256':mask_sha} for row in annotations])
    atomic_json(root/'split.json',split)


def check_baseline():
    baseline_path = PROJECT/'runs/baseline/stage2_fp32.pt'
    if not baseline_path.exists():
        return None
    torch.manual_seed(2026)
    saved = torch.load(baseline_path,map_location='cpu',weights_only=True)
    vanilla = load_weights(PUBLIC_STAGE2,2).eval()
    enhanced = create_model(2,cycn={'mode':'five_phase'}).eval()
    enhanced.load_state_dict({**enhanced.state_dict(),**vanilla.state_dict()},strict=True)
    assert list(vanilla.state_dict()) == saved['keys']
    with torch.no_grad():
        off = forward_volume(vanilla,saved['input'])
        new = forward_volume(enhanced,saved['input'],'ordered_states')
        errors = {}
        for name,actual in [('off',off),('zero_projection',new)]:
            torch.testing.assert_close(actual,saved['output'],rtol=1e-5,atol=1e-6)
            errors[name] = float((actual-saved['output']).abs().max())
        enhanced.cycn_branch.fuse.weight.fill_(.05)
        enhanced.cycn_branch.z_chunk=1
        chunked = forward_volume(enhanced,saved['input'],'ordered_states')
        enhanced.cycn_branch.z_chunk=16
        full = forward_volume(enhanced,saved['input'],'ordered_states')
        torch.testing.assert_close(chunked,full,rtol=1e-5,atol=1e-6)
        errors['z_chunk_1_vs_16_nonzero'] = float((chunked-full).abs().max())
    return {'keys':len(saved['keys']),'rtol':1e-5,'atol':1e-6,'max_abs':errors}


def main(root):
    root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    make_fixture(root)
    log = root/'commands.log'
    manifest = root/'manifest.jsonl'
    split = root/'split.json'
    base = root/'A_weights.pt'
    extractor = root/'synthetic_extractor.pt'
    # The fixed synthetic segmentation rule has no fitted weights.
    script = torch.jit.trace(lambda volume: volume > .018, torch.zeros(1,1,5,25,33))
    with extractor.open('wb') as file:
        torch.jit.save(script, file)
    run_cli(log,'convert-weights','--input',PUBLIC_STAGE2,'--out',base,'--stage',2,
            '--intensity-domain','mu','--normalization-scale','.02')
    choices = {'B':'backbone','C':'center_only','D':'five_phase'}
    for arm,variant in choices.items():
        config=load_config(PROJECT/f'configs/train/pig_cycn_{variant}.yaml')
        config.update(device='cpu',amp=False,threads=1,deterministic=True,stage2_patches=[[4,16,24]],
                      log_every=1,loss={'kind':'l1','ssim_weight':0.})
        config['epochs']['2']=2
        config['steps_per_epoch']['2']=4
        config['milestones']['2']=[1]
        config['cycn']['z_chunk']=2
        path = root/f'{arm}.yaml'
        path.write_text(yaml.safe_dump(config),encoding='utf-8')
        train_args = ('train','--stage',2,'--config',path,'--manifest',manifest,'--split',split,
                      '--init-from-stage2',base,'--run-dir')
        run_cli(log,*train_args,root/arm)
        if arm=='D':
            run_cli(log,*train_args,root/'D_resumed','--stop-after-epoch',1)
            run_cli(log,'train','--resume',root/'D_resumed/checkpoints/last.pt','--stop-after-epoch',2)
    ck_paths = {'A':base, **{arm:root/arm/'checkpoints/best.pt' for arm in choices}}
    resumed = torch.load(root/'D_resumed/checkpoints/last.pt',map_location='cpu',weights_only=True)
    continuous = torch.load(root/'D/checkpoints/last.pt',map_location='cpu',weights_only=True)
    assert all(torch.equal(value,resumed['state_dict'][name]) for name,value in continuous['state_dict'].items())
    for field in ('step','best','scheduler','optimizer_group_names','cycn'):
        assert continuous[field] == resumed[field], field
    assert continuous['optimizer']['param_groups'] == resumed['optimizer']['param_groups']
    for idx, state in continuous['optimizer']['state'].items():
        assert all(torch.equal(value,resumed['optimizer']['state'][idx][key]) for key,value in state.items())
    draws = [[(row['epoch'], row['draw'],row['case_id'],row['shape'])
              for row in read_jsonl(root/arm/'train.jsonl') if row['event']=='batch'] for arm in choices]
    assert all(draw == draws[0] for draw in draws)
    assert {draw[2] for draw in draws[0]} == {'train-cyclic','train-ordered_states'}

    for arm,weights in ck_paths.items():
        for mode in ('tiled','full') if arm in ('C','D') else ('tiled',):
            out = root/f'pred_{arm}_{mode}'
            flags = ('--compare-full',) if mode=='tiled' else ()
            run_cli(log,'infer','--checkpoint',weights,'--manifest',manifest,'--split',split,
                    '--split-name','val','--mode',mode,'--tile',6,16,24,'--overlap',2,8,16,
                    '--device','cpu','--fp32','--out',out,*flags)
            for record in read_jsonl(out/'manifest.jsonl'):
                with h5py.File(out/record['path'],'r') as pred:
                    meta = json.loads(pred.attrs['metadata_json'])
                    assert pred['prediction'].shape == (len(meta['phase_labels']),5,25,33)
                    assert meta['phase_labels'] == [10+7*i for i in range(len(meta['phase_labels']))]
                    assert meta['inference']['cycn']['mode'] == {'C':'center_only','D':'five_phase'}.get(arm,'off')
            if mode=='tiled':
                run_cli(log,'evaluate','--predictions',out,'--manifest',manifest,
                        '--lv-annotations',root/'annotations.jsonl','--lv-extractor',extractor,'--out',root/f'eval_{arm}')
    run_cli(log,'compare-cycn','--a',root/'eval_A','--b',root/'eval_B','--c',root/'eval_C',
            '--d',root/'eval_D','--out',root/'synthetic_comparison.json')
    comparison=read_json(root/'synthetic_comparison.json')
    assert comparison['comparisons']['D_minus_C']['summary']['boundary_mae']['paired_animals'] == 2
    summary={'fixture':'synthetic engineering acceptance','baseline_fp32':check_baseline(),
             'train_cases':sorted({draw[2] for draw in draws[0]}),
             'validation_cases':sorted(row['case_id'] for row in continuous['validation_samples']),
             'resume':{'parameters_exact':True,'optimizer_exact':True,'effective_steps':continuous['step'],
                       'best':continuous['best'],
                       'learning_rates':{group['name']:group['lr'] for group in continuous['optimizer']['param_groups']}},
             'boundary_paired_animals':2,'source_public_weights_sha256':sha256(PUBLIC_STAGE2)}
    atomic_json(root/'summary.json',summary)
    print(json.dumps(summary))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    main(parser.parse_args().out.resolve())
