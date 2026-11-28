import copy
import tempfile
import unittest
from pathlib import Path
import torch
from tests.test_data import fixture
from rstar4d.data.contracts import write_manifest
from rstar4d.config import train_defaults
from rstar4d.training.trainer import train
from rstar4d.training.pipeline import pipeline
from rstar4d.training.checkpoint import load_checkpoint
from rstar4d.utils.io import atomic_json


def inputs(root):
    rows=[fixture(root,10,'a','a'),fixture(root,11,'b','b'),fixture(root,10,'c','c')]
    write_manifest(root/'manifest.jsonl',rows)
    atomic_json(root/'split.json',{'train':['a'],'val':['b'],'test':['c']})
    cfg=train_defaults(); cfg.update(device='cpu',amp=False,deterministic=True,threads=1,
        epochs={'1':2,'2':2},steps_per_epoch={'1':2,'2':2},stage1_patch=[1,16,16],stage2_patches=[[4,16,16]],
        loss={'kind':'l1','ssim_weight':0.,'window':[0,.05]})
    return cfg,root/'manifest.jsonl',root/'split.json'


def ttl_inputs(root):
    from rstar4d.data.contracts import snapshot, validate_manifest
    from rstar4d.models import create_model
    from rstar4d.models.stam import architecture
    from rstar4d.training.checkpoint import atomic_save
    from rstar4d.utils.io import read_json

    cfg, manifest, split = inputs(root)
    torch.manual_seed(11)
    mode = 'joint'
    parent = create_model(2, stam={'mode': mode})
    torch.nn.init.normal_(parent.stam.output_projection.weight, std=0.005)
    torch.nn.init.normal_(parent.stam.temporal_adapter.up.weight, std=0.01)
    path = root/'parent.pt'
    groups = read_json(split)
    atomic_save(path, dict(schema_version=1, checkpoint_kind='weights', stage=2,
        model_profile=cfg['model_profile'], state_dict=parent.state_dict(),
        normalization=cfg['normalization'], intensity_domain='mu', calibration_application='none',
        config=cfg, split=groups, data_snapshot=snapshot(validate_manifest(manifest), groups),
        temporal_topologies=['ordered_states'], stam={'architecture': architecture(mode)}))
    cfg.update(ttl={'mode': 'full'}, stam={'mode': 'from_parent'})
    return cfg, manifest, split, {'kind': 'stam_parent', 'source_stage': 2, 'path': path}


class TTLTrainingTests(unittest.TestCase):
    def test_ttl_epoch0_can_win(self):
        from unittest.mock import patch
        from rstar4d.models import model_from_checkpoint
        from rstar4d.utils.io import read_jsonl

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg, manifest, split, init = ttl_inputs(root)
            validation = [dict(l1=l1, animal_l1=l1, cases={'b': l1}) for l1 in (0.25, 0.25, 0.5)]
            with patch('rstar4d.training.trainer.validate_model', side_effect=validation) as checks:
                result = train(cfg, manifest, split, root/'run', 2, init=init)
            self.assertEqual(checks.call_count, 3)
            best = load_checkpoint(result['checkpoint'])
            self.assertEqual((best['completed_epoch'], best['step'], best['best']), (0, 0, 0.25))
            self.assertEqual(best['ttl']['freeze_policy'], 'ttl_only')
            self.assertEqual(best['ttl']['param_groups'], ['ttl'])
            init['path'].rename(root/'moved-parent.pt')
            model = model_from_checkpoint(best)
            self.assertEqual(model.ttl.mode, 'full')
            self.assertEqual(torch.count_nonzero(model.ttl.temporal.weight).item(), 0)
            self.assertEqual(read_jsonl(root/'run/train.jsonl')[0]['event'], 'initial_validation')

    def test_ttl_resume_matches_continuous(self):
        import subprocess
        import sys
        from rstar4d.training.checkpoint import atomic_save
        from rstar4d.utils.io import sha256

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg, manifest, split, init = ttl_inputs(root)
            cfg['ttl']['z_chunk_size'] = 1
            cfg['milestones']['2'] = [1]
            parent_hash = sha256(init['path'])
            continuous = train(cfg, manifest, split, root/'complete', 2, init=init)
            partial = train(cfg, manifest, split, root/'partial', 2, init=init, stop_after_epoch=1)
            saved = load_checkpoint(partial['last'])
            bad = dict(saved, ttl=dict(saved['ttl'], freeze_policy='stam'))
            atomic_save(partial['last'], bad)
            with self.assertRaisesRegex(ValueError, 'freeze'):
                train(cfg, manifest, split, root/'partial', 2, resume=partial['last'])
            atomic_save(partial['last'], saved)
            moved = init['path'].rename(root/'moved-parent.pt')
            result = subprocess.run([sys.executable, '-m', 'rstar4d', 'train', '--resume', partial['last']],
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for kind in ('checkpoint', 'last'):
                a, b = load_checkpoint(continuous[kind]), load_checkpoint(partial[kind])
                self.assertEqual(a['step'], b['step'])
                self.assertEqual(a['best'], b['best'])
                self.assertEqual(a['completed_epoch'], b['completed_epoch'])
                self.assertEqual(a['scheduler'], b['scheduler'])
                self.assertEqual(a['optimizer']['param_groups'], b['optimizer']['param_groups'])
                self.assertEqual(a['parent_checkpoint_sha256'], b['parent_checkpoint_sha256'])
                self.assertEqual(a['parent_checkpoint_sha256'], parent_hash)
                for name, tensor in a['state_dict'].items():
                    self.assertTrue(torch.equal(tensor, b['state_dict'][name]), name)
                for key, state in a['optimizer']['state'].items():
                    for field, value in state.items():
                        self.assertTrue(torch.equal(value, b['optimizer']['state'][key][field]), field)
            self.assertEqual(load_checkpoint(partial['last'])['step'], 4)
            self.assertEqual(sha256(moved), parent_hash)

    def test_ttl_only_two_updates(self):
        from types import MethodType
        from unittest.mock import patch
        from rstar4d.training.trainer import make_optimizer
        from rstar4d.utils.io import sha256

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg, manifest, split, init = ttl_inputs(root)
            cfg['epochs']['2'] = 1
            parent = load_checkpoint(init['path'], require_training=False)
            before_hash = sha256(init['path'])
            updates = []

            def audited_optimizer(model, *args):
                optimizer = make_optimizer(model, *args)
                self.assertEqual([group['name'] for group in optimizer.param_groups], ['ttl'])
                ids = {id(p) for p in optimizer.param_groups[0]['params']}
                self.assertEqual(ids, {id(p) for p in model.ttl.parameters()})
                self.assertEqual(ids, {id(p) for p in model.parameters() if p.requires_grad})
                self.assertEqual(sum(p.numel() for p in model.ttl.parameters()), 1_887_136)
                step = optimizer.step

                def audit_step(_optimizer, *args, **kwargs):
                    for name, module in model.named_modules():
                        self.assertEqual(module.training, name == 'ttl' or name.startswith('ttl.'), name)
                    for name, param in model.named_parameters():
                        if id(param) not in ids:
                            self.assertIsNone(param.grad, name)
                    params = dict(model.ttl.named_parameters())
                    active = ('temporal.weight',) if not updates else (
                        'embed.weight', 'blocks.0.qkv.weight', 'blocks.1.mlp.0.weight', 'unembed.weight')
                    for name in active:
                        grad = params[name].grad
                        self.assertIsNotNone(grad, name)
                        self.assertTrue(torch.isfinite(grad).all() and grad.abs().max() > 0, name)
                    before = params['temporal.weight'].detach().clone()
                    result = step(*args, **kwargs)
                    self.assertFalse(torch.equal(before, params['temporal.weight']))
                    for name, tensor in model.state_dict().items():
                        if not name.startswith('ttl.'):
                            self.assertTrue(torch.equal(tensor, parent['state_dict'][name]), name)
                    updates.append(True)
                    return result

                optimizer.step = MethodType(audit_step, optimizer)
                return optimizer

            with patch('rstar4d.training.trainer.make_optimizer', side_effect=audited_optimizer):
                result = train(cfg, manifest, split, root/'run', 2, init=init)
            self.assertEqual(len(updates), 2)
            child = load_checkpoint(result['last'])
            self.assertEqual(child['step'], 2)
            self.assertEqual(child['parent_checkpoint_sha256'], before_hash)
            self.assertEqual(sha256(init['path']), before_hash)
            self.assertEqual(child['stam']['architecture']['mode'], 'joint')


class TrainingTests(unittest.TestCase):
    def test_nonfinite_output_records_failure_without_checkpoint(self):
        from unittest.mock import patch
        from rstar4d.utils.io import read_json
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); cfg,manifest,split=inputs(root)
            with patch('rstar4d.training.trainer.forward_volume',side_effect=lambda model,x:x[:,:1]*float('nan')):
                with self.assertRaises(FloatingPointError): train(cfg,manifest,split,root/'failed',1)
            self.assertIn('Nonfinite loss',read_json(root/'failed/error.json')['error'])
            self.assertFalse((root/'failed/checkpoints/best.pt').exists())
            self.assertFalse((root/'failed/checkpoints/last.pt').exists())

    def test_accumulation_final_short_group_has_full_weight(self):
        from unittest.mock import patch
        from rstar4d.data.contracts import read_manifest
        from rstar4d.data.sampling import PatchDataset
        class LinearProbe(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight=torch.nn.Parameter(torch.zeros(()))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); cfg,manifest,split=inputs(root)
            cfg.update(accumulation=2,epochs={'1':1,'2':1},steps_per_epoch={'1':3,'2':3})
            rows=[r for r in read_manifest(manifest) if r['animal_id']=='a']
            dataset=PatchDataset(rows,cfg,1,1,3)
            gradients=[float(dataset[i]['input'][:1].mean()) for i in range(3)]
            def optimizer(params,lr,**kwargs): return torch.optim.SGD(params,lr=lr)
            with patch('rstar4d.training.trainer.create_model',side_effect=lambda *args:LinearProbe()), \
                 patch('rstar4d.training.trainer.forward_volume',side_effect=lambda model,x:x[:,:1]*model.weight), \
                 patch('rstar4d.training.trainer.torch.optim.Adam',side_effect=optimizer):
                train(cfg,manifest,split,root/'accumulation',1)
            actual=load_checkpoint(root/'accumulation/checkpoints/last.pt')
            expected=cfg['learning_rate']*((gradients[0]+gradients[1])/2+gradients[2])
            self.assertAlmostEqual(actual['state_dict']['weight'].item(),expected,places=9)
            self.assertEqual(actual['step'],2)

    def test_exact_epoch_resume_and_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); cfg,manifest,split=inputs(root)
            from rstar4d.data.contracts import read_manifest
            rows=read_manifest(manifest); rows.append(fixture(root,10,'a','a_second'))
            write_manifest(manifest,rows)
            train(cfg,manifest,split,root/'full',1)
            train(cfg,manifest,split,root/'resumed',1,stop_after_epoch=1)
            write_manifest(manifest,rows[::-1])
            ck=root/'resumed/checkpoints/last.pt'
            train(cfg,manifest,split,root/'resumed',1,resume=ck)
            a=load_checkpoint(root/'full/checkpoints/last.pt'); b=load_checkpoint(ck)
            for k,v in a['state_dict'].items(): self.assertTrue(torch.equal(v,b['state_dict'][k]),k)
            self.assertEqual(a['scheduler'],b['scheduler'])
            for i,state in a['optimizer']['state'].items():
                for k,v in state.items():
                    if torch.is_tensor(v): self.assertTrue(torch.equal(v,b['optimizer']['state'][i][k]))
            changed=copy.deepcopy(cfg); changed['learning_rate']*=2
            with self.assertRaises(ValueError): train(changed,manifest,split,root/'resumed',1,resume=ck)

    def test_two_stage_pipeline_reentry(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); cfg,manifest,split=inputs(root)
            result=pipeline(cfg,manifest,split,root/'run')
            last=load_checkpoint(root/'run/stage2/checkpoints/last.pt')
            self.assertEqual(last['stage'],2); self.assertTrue(last['stage_complete'])
            self.assertEqual(last['initialization']['source_stage'],1)
            self.assertTrue(Path(result['checkpoint']).is_file())
            second=pipeline(cfg,manifest,split,root/'run')
            self.assertEqual(result['checkpoint'],second['checkpoint'])

    def test_stam_first_and_second_effective_update(self):
        from rstar4d.models import create_model, forward_volume
        from rstar4d.training.trainer import make_optimizer

        torch.set_num_threads(1)
        torch.manual_seed(6)
        data = torch.rand(1, 2, 10, 3, 17, 19)
        target = torch.rand_like(data[:, :1])
        first = {}
        for policy in ('frozen', 'joint'):
            torch.manual_seed(99)
            config = train_defaults()
            config['stam'].update(mode='joint', train_policy=policy)
            model = create_model(2, stam=config['stam'])
            optimizer = make_optimizer(model, config, backbone_frozen=policy == 'frozen')
            names = ['stam'] if policy == 'frozen' else ['stam', 'backbone']
            self.assertEqual([g['name'] for g in optimizer.param_groups], names)
            parameters = [p for g in optimizer.param_groups for p in g['params']]
            self.assertEqual(len(parameters), len(set(parameters)))
            self.assertEqual(set(parameters), {p for p in model.parameters() if p.requires_grad})
            before = {key: value.clone() for key, value in model.state_dict().items()}
            for update in range(2):
                model.train()
                self.assertTrue(all(not p.requires_grad for p in model.stam.sam.parameters()))
                pred = forward_volume(model, data)
                if update == 0:
                    first[policy] = pred.detach()
                loss = (pred - target).abs().mean()
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters))
                block = model.stam
                early_layers = (block.input_projection, block.spatial_adapter.down,
                                block.temporal_adapter.down, block.fusion_adapter.down)
                self.assertTrue(all(bool(layer.weight.grad.any()) == bool(update) for layer in early_layers))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                self.assertFalse(torch.equal(block.output_projection.weight, before['stam.output_projection.weight']))
                backbone = [k for k in before if not k.startswith('stam.')]
                self.assertEqual(any(not torch.equal(model.state_dict()[k], before[k]) for k in backbone), policy == 'joint')
                for key, tensor in model.state_dict().items():
                    if key.startswith('stam.sam.') or (policy == 'frozen' and not key.startswith('stam.')):
                        self.assertTrue(torch.equal(tensor, before[key]), key)
        self.assertTrue(torch.equal(first['frozen'], first['joint']))

    def test_stam_resume_and_fork_match_uninterrupted_runs(self):
        from rstar4d.models import create_model
        from rstar4d.models.stam import SAMBlock
        from rstar4d.training.checkpoint import atomic_save
        from rstar4d.utils.io import read_jsonl, sha256

        def equal_state(actual, expected):
            self.assertEqual(actual['step'], expected['step'])
            self.assertEqual(actual['scheduler'], expected['scheduler'])
            self.assertEqual(actual['optimizer']['param_groups'], expected['optimizer']['param_groups'])
            for key, tensor in actual['state_dict'].items():
                self.assertTrue(torch.equal(tensor, expected['state_dict'][key]), key)
            for key, values in actual['optimizer']['state'].items():
                for name, tensor in values.items():
                    self.assertTrue(torch.equal(tensor, expected['optimizer']['state'][key][name]), name)

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg, manifest, split = inputs(root)
            cfg['epochs']['2'] = 3
            cfg['milestones']['2'] = [1, 2]
            cfg['steps_per_epoch']['2'] = 1
            torch.manual_seed(79)
            stage2 = root/'source.pt'
            atomic_save(stage2, dict(stage=2, schema_version=1, checkpoint_kind='weights',
                model_profile=cfg['model_profile'], state_dict=create_model(2).state_dict(),
                normalization=cfg['normalization'], calibration_application='none', intensity_domain='mu',
                temporal_topologies=['ordered_states'], split={'train':['a'], 'val':['b'], 'test':['c']}))
            source = {'path': stage2, 'source_stage': 2}
            sam = root/'sam.pt'
            torch.save({'image_encoder.blocks.5.' + k: v for k, v in SAMBlock().state_dict().items()}, sam)
            cfg['stam'].update(mode='joint', warmup_epochs=1, sam_checkpoint=str(sam))

            for policy in ('frozen', 'joint', 'joint_after_warmup'):
                cfg['stam']['train_policy'] = policy
                full, resumed = root/f'{policy}-full', root/f'{policy}-resumed'
                train(cfg, manifest, split, full, 2, init=source)
                train(cfg, manifest, split, resumed, 2, init=source, stop_after_epoch=1)
                saved = resumed/'checkpoints/last.pt'
                if policy == 'frozen':
                    parent = resumed/'checkpoints/prefix'
                    parent.mkdir()
                    for name in ('last.pt', 'best.pt'):
                        (parent/name).write_bytes((saved.with_name(name)).read_bytes())
                sam.rename(root/'sam.offline')
                train(cfg, manifest, split, resumed, 2, resume=saved, stop_after_epoch=2)
                train(cfg, manifest, split, resumed, 2, resume=saved)
                (root/'sam.offline').rename(sam)
                equal_state(load_checkpoint(saved), load_checkpoint(full/'checkpoints/last.pt'))
                def lrs(path):
                    return [r['group_lrs'] for r in read_jsonl(path/'train.jsonl') if r['event'] == 'epoch']
                self.assertEqual(lrs(full), lrs(resumed))
                if policy == 'joint_after_warmup':
                    joint_full = full
                    expected = [{'stam': .00015}, {'stam': .000075, 'backbone': .0000075},
                                {'stam': .0000375, 'backbone': .00000375}]
                    for actual, rates in zip(lrs(full), expected):
                        self.assertEqual(actual.keys(), rates.keys())
                        for name, rate in rates.items():
                            self.assertAlmostEqual(actual[name], rate, places=15)
                    self.assertEqual(load_checkpoint(full/'checkpoints/warmup/last.pt')['stam']['param_groups'], ['stam'])

            fork = root/'forked'
            train(cfg, manifest, split, fork, 2, fork_from=parent/'last.pt')
            forked = load_checkpoint(fork/'checkpoints/last.pt')
            self.assertEqual(forked['parent_checkpoint_sha256'], sha256(parent/'last.pt'))
            equal_state(forked, load_checkpoint(joint_full/'checkpoints/last.pt'))

            cfg['stam']['train_policy'] = 'frozen'
            frozen_fork = root/'forked-frozen'
            train(cfg, manifest, split, frozen_fork, 2, fork_from=parent/'last.pt')
            equal_state(load_checkpoint(frozen_fork/'checkpoints/last.pt'),
                        load_checkpoint(root/'frozen-full/checkpoints/last.pt'))
            with self.assertRaisesRegex(ValueError, 'snapshot differs'):
                train(cfg, manifest, {'train':['b'],'val':['a'],'test':['c']}, root/'bad-data', 2,
                      fork_from=parent/'last.pt')
            cfg['stam']['mode'] = 'temporal_only'
            with self.assertRaisesRegex(ValueError, 'modes must match'):
                train(cfg, manifest, split, root/'bad-mode', 2, fork_from=parent/'last.pt')

    def test_original_control_starts_on_absolute_epoch_and_lr(self):
        from unittest.mock import patch
        from rstar4d.data.sampling import PatchDataset
        from rstar4d.models import create_model
        from rstar4d.training.checkpoint import atomic_save
        from rstar4d.utils.io import read_jsonl

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg, manifest, split = inputs(root)
            cfg.update(stage2_start_epoch=2, stage2_include_initial=True, learning_rate=.000015)
            cfg['milestones']['2'] = [1]
            source = root/'source.pt'
            atomic_save(source, dict(stage=2, schema_version=1, checkpoint_kind='weights',
                model_profile=cfg['model_profile'], state_dict=create_model(2).state_dict(),
                normalization=cfg['normalization'], calibration_application='none', intensity_domain='mu',
                temporal_topologies=['ordered_states'], split={'train':['a'], 'val':['b'], 'test':['c']}))
            with patch('rstar4d.training.trainer.PatchDataset', wraps=PatchDataset) as datasets:
                train(cfg, manifest, split, root/'control', 2, init={'path':source,'source_stage':2})
            self.assertEqual([call.args[3] for call in datasets.call_args_list], [0, 2, 0])
            events = read_jsonl(root/'control/train.jsonl')
            self.assertEqual(events[0]['event'], 'initial_validation')
            self.assertEqual(events[-1]['epoch'], 2)
            self.assertAlmostEqual(events[-1]['lr'], .0000075)
            self.assertEqual(events[-1]['steps'], cfg['steps_per_epoch']['2'])

if __name__=='__main__': unittest.main()
