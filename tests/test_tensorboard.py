import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from rstar4d.training.checkpoint import load_checkpoint
from rstar4d.training.trainer import train
from tests.test_training import inputs


class TensorBoardTests(unittest.TestCase):
    def test_tensorboard_cannot_discover_uncommitted_event_files(self):
        from tensorboard.backend.event_processing.io_wrapper import GetLogdirSubdirectories
        from rstar4d.training.tensorboard import _event_file
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with _event_file(root/'run'/'tensorboard', 1) as writer:
                writer.add_scalar('train/total', 2., 1)
                writer.flush()
                self.assertEqual(list(GetLogdirSubdirectories(tmp)), [])
            self.assertEqual([Path(d) for d in GetLogdirSubdirectories(tmp)], [root/'run'/'tensorboard'])

    def test_resume_recreates_missing_events_and_preserves_training(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            config, manifest, split = inputs(root)
            run = root / 'visible'
            first = train(config, manifest, split, run, 1, stop_after_epoch=1)
            files = list((run / 'tensorboard').glob('events.out.tfevents.*'))
            self.assertEqual(len(files), 1)
            files[0].unlink()  # Interrupted immediately after the model archive was committed.
            orphan = run / 'tensorboard' / 'events.out.tfevents.0000000009.rstar'
            orphan.write_bytes(b'orphaned event')
            resumed = train(config, manifest, split, run, 1, resume=first['last'])
            self.assertFalse(orphan.exists())
            train(config, manifest, split, run, 1, resume=resumed['last'])
            events = EventAccumulator(str(run / 'tensorboard')).Reload()
            for tag in ('train/l1', 'validation/animal_l1', 'optimizer/learning_rate'):
                self.assertEqual([event.step for event in events.Scalars(tag)], [1, 2])
            for tag in ('sparse', 'prior', 'prediction', 'reference', 'absolute_error'):
                self.assertEqual([event.step for event in events.Images('validation/' + tag)], [1, 2])
            quiet = train({**config, 'tensorboard': False}, manifest, split, root/'quiet', 1)
            self.assertFalse((root/'quiet'/'tensorboard').exists())
            visible = load_checkpoint(resumed['last'])
            hidden = load_checkpoint(quiet['last'])
            for name, parameter in hidden['state_dict'].items():
                self.assertTrue(torch.equal(parameter, visible['state_dict'][name]), name)

    def test_cardiac_means_have_effective_denominators(self):
        from rstar4d.training import trainer
        self.assertTrue(hasattr(trainer, 'CardiacAverages'))
        means = trainer.CardiacAverages()
        means.update({'struct': 2., 'struct_weighted': .1, 'windows': 3, 'outside': 2})
        means.update({'struct': 0., 'struct_weighted': 0., 'windows': 1, 'outside': 0})
        means.update({'boundary': 0., 'boundary_weighted': 0., 'structures': {
            'heart_outer': {'voxels': 0}, 'lv': {'voxels': 0}}})
        self.assertEqual(means.values()['struct'], 1.5)
        self.assertAlmostEqual(means.values()['struct_weighted'], .075)
        self.assertNotIn('boundary', means.values())

    def test_preview_honors_fixed_window_and_reference_surfaces(self):
        from validation.cardiac_smoke import fixture
        from rstar4d.data.contracts import read_manifest
        from rstar4d.config import validate_train
        from rstar4d.training.tensorboard import validation_preview
        import h5py

        with tempfile.TemporaryDirectory() as tmp:
            row = read_manifest(fixture(Path(tmp)))[0]
            shape = (1, 1, 2, 8, 24, 24)
            reference = torch.ones(shape)
            predicted = reference * 2
            image = torch.cat((reference * 0, reference), 1)
            preview = validation_preview(image, predicted, reference,
                                         {'origin': torch.zeros(1, 3, dtype=torch.int64)},
                                         row, validate_train({}))
            self.assertEqual(preview['phase_label'], 9)
            self.assertEqual(preview['slice_zyx'], [4, 0, 0])
            panels = preview['images']
            self.assertEqual(panels['sparse'].count_nonzero().item(), 0)
            self.assertTrue(torch.equal(panels['absolute_error'], panels['reference']))
            self.assertTrue(torch.all(panels['prediction'] == 204))
            with h5py.File(row['annotations']) as labels:
                lv = torch.from_numpy(labels['surface'][1, 1, 4]).bool()
            self.assertTrue(lv.any())
            self.assertTrue(torch.all(panels['prediction_reference_contours'][2, lv] == 255))
            stale = validation_preview(image, predicted, reference,
                                       {'origin': torch.zeros(1, 3, dtype=torch.int64)},
                                       {**row, 'annotations_sha256': 'changed'}, validate_train({}))
            self.assertNotIn('prediction_reference_contours', stale['images'])

    def test_independent_contour_evaluation_writes_animal_counts(self):
        from rstar4d.training.tensorboard import write_evaluation_events
        with tempfile.TemporaryDirectory() as tmp:
            summary = {'contours': {'structures': {'lv': {
                key: {'n': count, 'mean': value} for key, count, value in (
                    ('surface_dice', 2, .75), ('assd', 1, 2.), ('hd95', 0, None), ('band_mae', 0, None))
            }}}}
            write_evaluation_events(tmp, summary)
            events = EventAccumulator(str(Path(tmp)/'tensorboard')).Reload()
            self.assertEqual(events.Scalars('evaluation/lv/surface_dice')[0].value, .75)
            self.assertEqual(events.Scalars('evaluation/lv/hd95_valid_animals')[0].value, 0)
            self.assertNotIn('evaluation/lv/hd95', events.Tags()['scalars'])
