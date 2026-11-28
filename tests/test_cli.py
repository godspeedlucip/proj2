import contextlib
import io
import tempfile
import unittest
from pathlib import Path
import yaml
import torch
from rstar4d.cli import main, parser
from rstar4d.utils.io import read_json, read_jsonl, atomic_jsonl


class CliTests(unittest.TestCase):
    def test_all_commands_exposed(self):
        text=parser().format_help()
        for name in ['doctor','inventory-pig','validate-settings','make-phantom','simulate-pig','import-paired',
            'validate-data','convert-weights','train','pipeline','infer','evaluate','benchmark']:
            self.assertIn(name,text)

    @unittest.skipUnless(torch.cuda.is_available(),'End-to-end fixture uses actual TIGRE CUDA')
    def test_two_entry_points_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            def run(*args):
                with contextlib.redirect_stdout(io.StringIO()): status=main(list(map(str,args)))
                self.assertEqual(status,0,args)
            run('make-phantom','--out',root/'raw','--views',200)
            run('simulate-pig','--manifest',root/'raw/raw_manifest.jsonl','--config',root/'raw/simulation.yaml','--out',root/'prepared')
            cfg={'device':'cuda','threads':2,'amp':True,'epochs':{'1':1,'2':1},'steps_per_epoch':{'1':8,'2':8},
                'stage1_patch':[1,32,32],'stage2_patches':[[8,32,32]],'log_every':1}
            (root/'train.yaml').write_text(yaml.safe_dump(cfg),encoding='utf-8')
            for entry in ['prepared','imported']:
                if entry=='imported':
                    rows=read_jsonl(root/'prepared/manifest.jsonl')
                    for r in rows:
                        r['paired_h5']=str(root/'prepared'/r['path'])
                        if r.get('projections'): r['projections']=str(root/'prepared'/r['projections'])
                    atomic_jsonl(root/'sources.jsonl',rows)
                    run('import-paired','--manifest',root/'sources.jsonl','--out',root/'imported')
                run('validate-data','--manifest',root/entry/'manifest.jsonl','--split',root/'raw/split.json')
                run('pipeline','--config',root/'train.yaml','--manifest',root/entry/'manifest.jsonl','--split',root/'raw/split.json','--run-dir',root/('run_'+entry))
                checkpoint=root/('run_'+entry)/'stage2/checkpoints/best.pt'
                run('infer','--checkpoint',checkpoint,'--manifest',root/entry/'manifest.jsonl','--out',root/('pred_'+entry),
                    '--tile',8,32,32,'--overlap',4,16,16,'--export','tif','nii')
                run('evaluate','--predictions',root/('pred_'+entry),'--manifest',root/entry/'manifest.jsonl','--out',root/('eval_'+entry))
                self.assertEqual(read_json(root/('eval_'+entry)/'summary.json')['cases'],1)

class TTLSmokeTests(unittest.TestCase):
    def run_cli(self, *args):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stdout):
            status = main(list(map(str, args)))
        self.assertEqual(status, 0, stdout.getvalue())

    def test_benchmark_ttl_only_and_legacy_stam(self):
        from tests.test_training import ttl_inputs

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config, _, _, init = ttl_inputs(root)
            config_path = root/'train.yaml'
            for mode in ('local3', 'off'):
                cfg = dict(config, ttl={'mode': mode})
                if mode == 'off':
                    cfg['stam'] = {'mode': 'joint', 'train_policy': 'frozen'}
                config_path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
                out = root/f'{mode}.json'
                self.run_cli('benchmark', '--stage', 2, '--config', config_path, '--phase-count', 5,
                             '--steps', 2, '--weights', init['path'], '--temporal-topology', 'cyclic', '--out', out)
                row = read_json(out)['results'][0]
                self.assertEqual(row['steps'], 2)
                self.assertTrue(row['parameter_update'])
                if mode == 'local3':
                    self.assertEqual(row['trainable_parameters'], 1_887_136)
                    self.assertEqual(row['optimizer_groups'], ['ttl'])
                    self.assertEqual(len(row['step_seconds']), 2)
                    self.assertTrue(row['parent_state_unchanged'])
                    self.assertFalse(row['output_projection_changed'])
                    self.assertGreaterEqual(row['parent_inference']['seconds'], 0.)
                else:
                    self.assertTrue(row['output_projection_changed'])
                    self.assertFalse(row['backbone_changed'])

    def test_frozen_parent_end_to_end(self):
        import h5py
        import numpy as np
        from rstar4d.training.checkpoint import atomic_save, load_checkpoint
        from rstar4d.utils.io import sha256
        from tests.test_training import ttl_inputs
        from validation.run_stam_smoke import ThresholdFixture, seam_report

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            config, manifest, split, init = ttl_inputs(root)
            parent = init['path']
            value = load_checkpoint(parent, require_training=False)
            value['state_dict']['conv8.conv.conv3_2.bias'] += 2
            atomic_save(parent, value)
            digest = sha256(parent)
            config_path = root/'train.yaml'
            config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
            run = root/'B'
            last = run/'checkpoints/last.pt'
            self.run_cli('validate-data', '--manifest', manifest, '--split', split)
            self.run_cli('train', '--stage', 2, '--manifest', manifest, '--split', split, '--config', config_path,
                         '--init-from-stam', parent, '--run-dir', run, '--stop-after-epoch', 1)
            self.assertEqual(load_checkpoint(last)['step'], 2)
            self.run_cli('train', '--resume', last)
            self.assertEqual(load_checkpoint(last)['step'], 4)
            self.assertEqual(sha256(parent), digest)
            best = run/'checkpoints/best.pt'

            rows = {row['case_id']: row for row in read_jsonl(manifest)}
            masks = root/'synthetic_lv.h5'
            z, y, x = np.indices((4, 16, 16)) - np.array([1.5, 7.5, 7.5])[:, None, None, None]
            with h5py.File(masks, 'w') as handle:
                handle['ED'] = (z * 2) ** 2 + y ** 2 + x ** 2 <= 10
                handle['ES'] = (z * 2) ** 2 + y ** 2 + x ** 2 <= 5
            annotations = root/'synthetic_lv.jsonl'
            atomic_jsonl(annotations, [
                {'case_id': 'c', 'source_sha256': rows['c']['sha256'], 'phase_label': label,
                 'role': role, 'path': masks.name, 'dataset': role, 'sha256': sha256(masks)}
                for label, role in ((3, 'ED'), (9, 'ES'))
            ])
            extractor = root/'\u5de6\u5ba4\u63d0\u53d6\u5668.pt'
            with extractor.open('wb') as handle:
                torch.jit.save(torch.jit.trace(ThresholdFixture(), torch.zeros((1, 1, 4, 16, 16))), handle)
            tile, overlap = (4, 16, 8), (2, 8, 0)
            for name, checkpoint, modes in (('A', parent, ('tiled',)), ('B', best, ('full', 'tiled'))):
                for mode in modes:
                    self.run_cli('infer', '--checkpoint', checkpoint, '--manifest', manifest, '--split', split,
                                 '--mode', mode, '--tile', *tile, '--overlap', *overlap, '--device', 'cpu', '--fp32',
                                 '--out', root/name/mode)
                self.run_cli('evaluate', '--predictions', root/name/'tiled', '--manifest', manifest,
                             '--lv-annotations', annotations, '--lv-extractor', extractor, '--out', root/name/'eval')
            self.assertEqual(read_jsonl(root/'B/tiled/manifest.jsonl')[0]['parent_checkpoint_sha256'], digest)
            seams = seam_report(root/'B/full', root/'B/tiled', tile, overlap)
            self.assertTrue(np.isfinite(seams['c']['seam_mae']))
            comparison = root/'B_vs_A.json'
            self.run_cli('compare-stam', '--seed', 2026, '--candidate', root/'B/eval',
                         '--reference', root/'A/eval', '--out', comparison)
            summary = read_json(comparison)['summary']
            self.assertEqual(summary['boundary_mae']['paired_animals'], 1)
            self.assertEqual(summary['motion_error_ml']['paired_animals'], 1)


if __name__=='__main__': unittest.main()
