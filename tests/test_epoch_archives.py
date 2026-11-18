import contextlib
import csv
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from rstar4d.training import checkpoint
from rstar4d.training.pipeline import pipeline
from rstar4d.training.trainer import train
from rstar4d.utils.io import append_jsonl, read_json, read_jsonl, sha256
from tests.test_training import inputs


class EpochArchiveTests(unittest.TestCase):
    def setUp(self):
        contexts = contextlib.ExitStack()
        self.addCleanup(contexts.close)
        self.root = Path(contexts.enter_context(tempfile.TemporaryDirectory()))
        contexts.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.cfg, self.manifest, self.split = inputs(self.root)
        self.cfg.update(milestones={'1': [1], '2': [1]}, log_every=1)
        self.run = self.root / 'run'
        self.ckdir = self.run / 'checkpoints'

    def fit(self, **kwargs):
        return train(self.cfg, self.manifest, self.split, self.run, 1, **kwargs)

    def assert_state_equal(self, left, right):
        if torch.is_tensor(left):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                with self.subTest(field=key):
                    self.assert_state_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_state_equal(a, b)
        else:
            self.assertEqual(left, right)

    def resume_cli(self, path, dest=None):
        command = [sys.executable, '-B', '-m', 'rstar4d', 'train', '--resume', str(path)]
        if dest:
            command.extend(['--run-dir', str(dest)])
        result = subprocess.run(command, capture_output=True, text=True,
                                env={**os.environ, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_two_stages_resume_in_a_new_process_with_identical_training_state(self):
        reference = self.root / 'reference'
        pipeline(self.cfg, self.manifest, self.split, reference)
        for stage in (1, 2):
            run = self.root / f'resumed{stage}'
            full = reference / f'stage{stage}' / 'checkpoints'
            kwargs = {} if stage == 1 else {'init': {'path': reference / 'stage1/checkpoints/best.pt', 'source_stage': 1}}
            train(self.cfg, self.manifest, self.split, run, stage, stop_after_epoch=1, **kwargs)
            first = run / 'checkpoints/epoch-0001.pt'
            self.assertTrue(first.is_file())
            self.resume_cli(first)
            left, right = (checkpoint.load_checkpoint(path) for path in
                           (full / 'epoch-0002.pt', run / 'checkpoints/last.pt'))
            for field in ('state_dict', 'optimizer', 'scheduler', 'scaler', 'step', 'rng',
                          'best', 'best_epoch', 'validation_samples'):
                with self.subTest(stage=stage, field=field):
                    self.assert_state_equal(left[field], right[field])
            self.assertEqual(left['epoch_metrics']['sampling_sha256'], right['epoch_metrics']['sampling_sha256'])
            self.assertEqual((right['completed_epoch'], right['next_epoch'], right['step']), (2, 3, 4))
        for stage in ('stage1', 'stage2'):
            (reference / stage / 'checkpoints/last.pt').unlink()
            (reference / stage / 'checkpoints/best.pt').unlink()
        self.assertTrue(pipeline(self.cfg, self.manifest, self.split, reference)['complete'])

    def test_failed_archive_write_replays_epoch_and_discards_partial_logs(self):
        self.fit(stop_after_epoch=1)
        first = self.ckdir / 'epoch-0001.pt'
        self.assertTrue(first.is_file())
        original = sha256(first)
        def broken_write(payload, file):
            file.write(b'partial zip')
            raise OSError('disk full')
        with patch.object(torch, 'save', broken_write), self.assertRaisesRegex(OSError, 'disk full'):
            self.fit(resume=self.ckdir / 'last.pt')
        self.assertFalse((self.ckdir / 'epoch-0002.pt').exists())
        self.assertEqual(sha256(first), original)
        (self.ckdir / 'epoch-9999.pt.tmp').write_bytes(b'partial')
        self.fit(resume=self.ckdir / 'last.pt')
        rows = read_jsonl(self.run / 'train.jsonl')
        self.assertEqual([(r['epoch'], r['draw']) for r in rows if r['event'] == 'batch'],
                         [(1, 0), (1, 1), (2, 0), (2, 1)])

    def test_pipeline_replays_first_epoch_from_original_initialization(self):
        self.cfg['epochs'] = {'1': 1, '2': 1}
        save = torch.save
        def interrupted(payload, output):
            if payload['stage'] == 2:
                raise OSError('interrupted first archive')
            save(payload, output)
        with patch.object(torch, 'save', interrupted), self.assertRaisesRegex(OSError, 'interrupted first archive'):
            pipeline(self.cfg, self.manifest, self.split, self.run)
        stage = self.run / 'stage2'
        self.assertFalse((stage / 'checkpoints/last.pt').exists())
        initial = read_json(stage / 'initialization.json')
        # The regular CLI can replay this same first epoch without an existing last.pt.
        self.resume_cli(stage / 'checkpoints/last.pt')
        self.assertTrue(pipeline(self.cfg, self.manifest, self.split, self.run)['complete'])
        archive = checkpoint.load_checkpoint(stage / 'checkpoints/epoch-0001.pt')
        self.assertEqual(archive['initialization'], initial)
        self.assertEqual([r['draw'] for r in read_jsonl(stage / 'train.jsonl') if r['event'] == 'batch'], [0, 1])

    def test_committed_archive_repairs_stale_aliases_and_partial_summaries(self):
        self.fit(stop_after_epoch=1)
        last = self.ckdir / 'last.pt'
        replace = os.replace
        def stop_after_archive(src, dest):
            if Path(dest).name in ('last.pt', 'best.pt'):
                raise OSError('interrupted alias')
            return replace(src, dest)
        with patch.object(os, 'replace', stop_after_archive), self.assertRaisesRegex(OSError, 'interrupted alias'):
            self.fit(resume=last)
        second = self.ckdir / 'epoch-0002.pt'
        self.assertTrue(second.is_file())
        (self.ckdir / 'best.pt').unlink()
        last.unlink(missing_ok=True)
        append_jsonl(self.run / 'train.jsonl', {'event': 'batch', 'epoch': 3, 'draw': 0})
        with (self.run / 'train.jsonl').open('ab') as output:
            output.write(b'{"evt":"\xe4')
        (self.run / 'metrics.csv').write_text('torn', encoding='utf-8')
        self.resume_cli(last)
        self.assertEqual(sha256(second), sha256(last))
        final = checkpoint.load_checkpoint(last)
        self.assertEqual(sha256(self.ckdir / 'best.pt'), sha256(self.ckdir / f'epoch-{final["best_epoch"]:04d}.pt'))
        rows = read_jsonl(self.run / 'train.jsonl')
        self.assertEqual([r['epoch'] for r in rows if r['event'] == 'epoch'], [1, 2])
        self.assertTrue(all(r['epoch'] <= 2 for r in rows))
        with (self.run / 'metrics.csv').open() as f:
            self.assertEqual([r['epoch'] for r in csv.DictReader(f)], ['1', '2'])
        self.assertTrue(read_json(self.run / 'status.json')['complete'])

    def test_historical_epoch_requires_empty_new_run_and_inherits_best_at_fork(self):
        from rstar4d.training.trainer import validate_model
        self.cfg['epochs']['1'] = 3
        # Epoch two is worse than one; the parent's later best must not enter the child.
        scores = iter((0.5, 0.6, 0.1))
        def ranked_validation(*args, **kwargs):
            return {**validate_model(*args, **kwargs), 'l1': next(scores)}
        with patch('rstar4d.training.trainer.validate_model', ranked_validation):
            self.fit()
        second = self.ckdir / 'epoch-0002.pt'
        self.assertTrue(second.exists())
        before = {p: sha256(p) for p in self.run.rglob('*') if p.is_file()}
        with self.assertRaisesRegex(ValueError, 'new run'):
            self.fit(resume=second)
        used = self.root / 'used'
        used.mkdir()
        (used / 'existing').touch()
        with self.assertRaises((ValueError, FileExistsError)):
            train(self.cfg, self.manifest, self.split, used, 1, resume=second)
        fork = self.root / 'fork'
        fork.mkdir()
        train(self.cfg, self.manifest, self.split, fork, 1, resume=second, stop_after_epoch=2)
        identity = read_json(fork / 'run_identity.json')
        parent = read_json(self.run / 'run_identity.json')
        self.assertNotEqual(identity['run_id'], parent['run_id'])
        self.assertEqual((identity['parent_run_id'], identity['fork_epoch']), (parent['run_id'], 2))
        best = checkpoint.load_checkpoint(fork / 'checkpoints/best.pt')
        self.assertEqual((best['completed_epoch'], best['best']), (1, 0.5))
        self.assertEqual(best['run_id'], identity['run_id'])
        self.assertEqual(sorted(p.name for p in (fork / 'checkpoints').glob('epoch-*.pt')),
                         ['epoch-0001.pt', 'epoch-0002.pt'])
        self.resume_cli(fork / 'checkpoints/last.pt')
        self.assertEqual(before, {p: sha256(p) for p in self.run.rglob('*') if p.is_file()})

    def test_legacy_last_retains_best_and_rejects_changed_code_signature(self):
        self.fit(stop_after_epoch=1)
        ck = checkpoint.load_checkpoint(self.ckdir / 'last.pt')
        self.assertIn('epoch_metrics', ck)
        for key in ('epoch_metrics', 'best_epoch', 'tensorboard_preview'):
            ck.pop(key, None)
        for p in self.ckdir.glob('epoch-*.pt'):
            p.unlink()
        last = self.ckdir / 'last.pt'
        for p in (last, self.ckdir / 'best.pt'):
            checkpoint.atomic_save(p, ck)
        with patch('rstar4d.training.trainer.code_fingerprint', return_value='new code'):
            with self.assertRaisesRegex(ValueError, 'warm start'):
                self.fit(resume=last)
        self.fit(resume=last)
        self.assertTrue((self.ckdir / 'epoch-0002.pt').is_file())
        (self.ckdir / 'best.pt').unlink()
        self.assertTrue(self.fit(resume=last)['complete'])

    def test_rejects_changed_fixed_validation_patches_before_repair(self):
        self.fit(stop_after_epoch=1)
        path = self.ckdir / 'epoch-0001.pt'
        ck = checkpoint.load_checkpoint(path)
        ck['validation_samples'][0]['origin'][0] += 1
        checkpoint.atomic_save(path, ck)
        original_alias = sha256(self.ckdir / 'last.pt')
        with self.assertRaisesRegex(ValueError, 'validation sample'):
            self.fit(resume=path)
        self.assertEqual(sha256(self.ckdir / 'last.pt'), original_alias)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for AMP updates')
    def test_cuda_amp_restores_scale_and_next_optimizer_update(self):
        from rstar4d.models import create_model
        import yaml
        self.cfg.update(device='cuda', amp=True, deterministic=True, steps_per_epoch={'1': 1, '2': 1})
        # Center the initial prediction in the phantom's intensity range so the
        # default AMP scale can make an effective update on the very first draw.
        torch.manual_seed(self.cfg['seed'])
        model = create_model(1)
        with torch.no_grad():
            model.conv8.conv.conv1_2.bias.add_(0.9)
        initial = self.root / 'initial.pt'
        torch.save(model.state_dict(), initial)
        self.cfg['init_weights'] = str(initial)
        config = self.root / 'cuda.yaml'
        config.write_text(yaml.safe_dump(self.cfg), encoding='utf-8')
        def start(run, stop=None):
            command = [sys.executable, '-B', '-m', 'rstar4d', 'train', '--stage', '1', '--config', str(config),
                       '--manifest', str(self.manifest), '--split', str(self.split), '--run-dir', str(run)]
            if stop:
                command.extend(['--stop-after-epoch', str(stop)])
            result = subprocess.run(command, capture_output=True, text=True,
                                    env={**os.environ, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})
            self.assertEqual(result.returncode, 0, result.stderr)
        ref = self.root / 'cuda-ref'
        start(ref)
        start(self.run, 1)
        first = self.ckdir / 'epoch-0001.pt'
        self.assertTrue(first.is_file())
        self.assertGreater(checkpoint.load_checkpoint(first)['scaler']['scale'], 0)
        self.resume_cli(first)
        a, b = (checkpoint.load_checkpoint(run / 'checkpoints/last.pt') for run in (ref, self.run))
        self.assertEqual(b['epoch_metrics']['optimizer_updates'], 1)
        for key in ('scaler', 'step', 'optimizer', 'state_dict'):
            self.assert_state_equal(a[key], b[key])
