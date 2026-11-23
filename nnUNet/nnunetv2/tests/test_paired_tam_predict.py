import json
import math

import numpy as np
import pytest
import SimpleITK as sitk
import torch

from nnunetv2.inference.paired_tam import PairedTAMPredictor
from nnunetv2.network_architecture.paired_tam import build_paired_network
from nnunetv2.utilities.paired_tam_contract import make_checkpoint_contract
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
DATASET_JSON = {
    'channel_names': {'0': 'CT'},
    'labels': {'background': 0, 'foreground': 1},
    'file_ending': '.mha',
}


def _plans(patch_size, transpose=(0, 1, 2), spacing=(1.0, 1.0, 1.0)):
    return PlansManager({
        'plans_name': 'paired_test',
        'label_manager': 'LabelManager',
        'transpose_forward': list(transpose),
        'transpose_backward': np.argsort(transpose).tolist(),
        'configurations': {'3d_fullres': {
            'patch_size': list(patch_size),
            'spacing': list(spacing),
            'resampling_fn_probabilities': 'resample_data_or_seg_to_shape',
            'resampling_fn_probabilities_kwargs': {'is_seg': False, 'order': 1},
            'architecture': {'network_class_name':
                             'dynamic_network_architectures.architectures.unet.PlainConvUNet',
                             'arch_kwargs': {}, '_kw_requires_import': []},
        }},
    })


def _predictor(network, patch_size, mirror_axes=None, transpose=(0, 1, 2), spacing=(1.0, 1.0, 1.0)):
    plans = _plans(patch_size, transpose, spacing)
    return PairedTAMPredictor(network, plans, plans.get_configuration('3d_fullres'), DATASET_JSON,
                              torch.device('cpu'), mirror_axes)


class _PairAndPositionModel(torch.nn.Module):
    def forward(self, patch):
        phases = patch[:, :, 0]
        opposite = phases.flip((1,))
        scales = phases.new_tensor([2, 3]).view(1, 2, 1, 1, 1)
        d, h, w = phases.shape[-3:]
        z = torch.arange(d, device=phases.device).view(d, 1, 1) - (d - 1) / 2
        y = torch.arange(h, device=phases.device).view(1, h, 1) - (h - 1) / 2
        x = torch.arange(w, device=phases.device).view(1, 1, w) - (w - 1) / 2
        return torch.stack((phases, scales * opposite + z + 2 * y + 3 * x), dim=2)


def test_sliding_windows_pad_and_mirror_both_phases_on_spatial_axes():
    predictor = _predictor(_PairAndPositionModel(), (4, 4, 4), mirror_axes=(0, 1, 2))
    phase1 = torch.arange(63, dtype=torch.float32).reshape(3, 3, 7)
    phase5 = phase1 + 80

    logits = predictor.predict_logits(torch.stack((phase1, phase5)))

    assert logits.shape == (2, 2, 3, 3, 7)
    torch.testing.assert_close(logits[0, 0], phase1)
    torch.testing.assert_close(logits[1, 0], phase5)
    torch.testing.assert_close(logits[0, 1], 2 * phase5)
    torch.testing.assert_close(logits[1, 1], 3 * phase1)


class _TilePositionModel(torch.nn.Module):
    def forward(self, patch):
        positions = torch.arange(patch.shape[-1], dtype=patch.dtype, device=patch.device)
        return positions.expand(*patch.shape[:2], 2, *patch.shape[-3:])


class _ThreeClassModel(torch.nn.Module):
    def forward(self, patch):
        return torch.cat((patch, patch + 1, patch + 2), dim=2)


def test_sliding_windows_restore_padding_without_cropping_segmentation_classes():
    plans = _plans((4, 4, 4))
    dataset = {**DATASET_JSON, 'labels': {'background': 0, 'atrium': 1, 'ventricle': 2}}
    predictor = PairedTAMPredictor(_ThreeClassModel(), plans, plans.get_configuration('3d_fullres'),
                                  dataset, 'cpu', mirror_axes=None)

    logits = predictor.predict_logits(torch.zeros((2, 2, 3, 4)))

    assert logits.shape == (2, 3, 2, 3, 4)
    torch.testing.assert_close(logits[:, 2], torch.full((2, 2, 3, 4), 2.0))


def test_sliding_windows_apply_gaussian_weights_independently_to_each_logit():
    predictor = _predictor(_TilePositionModel(), (2, 2, 4))

    logits = predictor.predict_logits(np.zeros((2, 2, 2, 6), dtype=np.float32))

    weights_at_overlap = 2 / (1 + math.exp(-8))
    expected = torch.tensor([0, 1, weights_at_overlap, 2, 2, 3]).expand(2, 2, 2, 2, 6)
    torch.testing.assert_close(logits, expected)


def _geometry(size, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), direction=IDENTITY):
    return {'size': size, 'spacing': spacing, 'origin': origin, 'direction': direction}


def _properties(common, phase5, bbox, transpose):
    shape_zyx = common['size'][::-1]
    return {
        'phase_order': ['phase1', 'phase5'],
        'original_geometries': {'phase1': common, 'phase5': phase5},
        'common_geometry': common,
        'spacing': common['spacing'][::-1],
        'shape_before_cropping': tuple(shape_zyx[i] for i in transpose),
        'bbox_used_for_cropping': bbox,
        'shape_after_cropping_and_before_resampling': tuple(upper - lower for lower, upper in bbox),
    }


def test_export_undoes_crop_and_transpose_into_each_original_physical_geometry(tmp_path):
    predictor = _predictor(_TilePositionModel(), (4, 4, 4), transpose=(2, 0, 1), spacing=(2.0, 4.0, 3.0))
    common = _geometry((7, 6, 5), (2.0, 3.0, 4.0), (10.0, 20.0, 30.0),
                       (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    phase5 = _geometry((6, 6, 5), (2.0, 3.0, 4.0), (10.0, 32.0, 30.0),
                       (0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    properties = _properties(common, phase5, ((2, 6), (1, 4), (1, 5)), (2, 0, 1))
    logits = torch.zeros((2, 2, 4, 3, 4), dtype=torch.float32)
    logits[:, 1] = -1
    logits[0, 1, 2, 1, 2] = 3
    logits[1, 1, 3, 0, 1] = 3

    outputs = predictor.export_prediction(logits, properties, 'scan_a', tmp_path, save_probabilities=True)

    assert set(outputs) == {'phase1', 'phase5'}
    for phase, geometry, point in [('phase1', common, [2, 3, 4]), ('phase5', phase5, [1, 2, 1])]:
        output = tmp_path / f'scan_a_{phase}.mha'
        assert str(output) == outputs[phase]
        image = sitk.ReadImage(str(output))
        assert image.GetSize() == geometry['size']
        assert image.GetSpacing() == geometry['spacing']
        assert image.GetOrigin() == geometry['origin']
        assert image.GetDirection() == geometry['direction']
        assert np.argwhere(sitk.GetArrayFromImage(image) == 1).tolist() == [point]
        with np.load(tmp_path / f'scan_a_{phase}.npz') as archive:
            probabilities = archive['probabilities']
        assert probabilities.shape == (2, *geometry['size'][::-1])
        np.testing.assert_allclose(probabilities[:, 0, 0, 0], [1.0, 0.0])
        assert probabilities[(1, *point)] == pytest.approx(0.9525741)


def test_export_resamples_logits_before_choosing_a_label(tmp_path):
    predictor = _predictor(_TilePositionModel(), (2, 2, 2), spacing=(1.0, 1.0, 0.5))
    common = _geometry((3, 2, 2))
    phase5 = _geometry((3, 2, 2), spacing=(1.25, 1.0, 1.0), origin=(0.25, 0.0, 0.0))
    properties = _properties(common, phase5, ((0, 2), (0, 2), (0, 3)), (0, 1, 2))
    logits = torch.zeros((2, 2, 2, 2, 6), dtype=torch.float32)
    logits[:, 1] = -1
    logits[1, 1, ..., 2:4] = 9

    outputs = predictor.export_prediction(logits, properties, 'scan_b', tmp_path)

    phase5_seg = sitk.GetArrayFromImage(sitk.ReadImage(outputs['phase5']))
    np.testing.assert_array_equal(phase5_seg[0, 0], [1, 1, 0])


def _write_checkpoint(model_dir, mirror_axes=(0, 1, 2)):
    plans = _plans((8, 4, 4))
    plans.plans['dataset_name'] = 'Dataset001_Paired'
    config = plans.plans['configurations']['3d_fullres']
    config.update({'normalization_schemes': ['CTNormalization'], 'batch_size': 1})
    config['architecture']['arch_kwargs'] = {
        'n_stages': 3,
        'features_per_stage': [8, 8, 8],
        'conv_op': 'torch.nn.Conv3d',
        'kernel_sizes': [[3, 3, 3]] * 3,
        'strides': [[1, 1, 1], [2, 2, 2], [2, 2, 2]],
        'n_conv_per_stage': [1, 1, 1],
        'n_conv_per_stage_decoder': [1, 1],
    }
    config['architecture']['_kw_requires_import'] = ['conv_op']
    configuration = plans.get_configuration('3d_fullres')
    network = build_paired_network(configuration.network_arch_class_name, configuration.network_arch_init_kwargs,
                                   configuration.network_arch_init_kwargs_req_import, 1, 2, True, 'C3')
    checkpoint = {
        'trainer_name': 'nnUNetTrainerTAMC3',
        'init_args': {'configuration': '3d_fullres'},
        'inference_allowed_mirroring_axes': mirror_axes,
        'network_weights': network.state_dict(),
        'paired_tam': make_checkpoint_contract(plans, configuration, DATASET_JSON, network, 0, mirror_axes),
        'paired_tam_training': {'split_sha256': 'example'},
    }
    (model_dir / 'fold_0').mkdir()
    (model_dir / 'plans.json').write_text(json.dumps(plans.plans), encoding='utf-8')
    (model_dir / 'dataset.json').write_text(json.dumps(DATASET_JSON), encoding='utf-8')
    torch.save(checkpoint, model_dir / 'fold_0' / 'checkpoint_final.pth')
    return checkpoint


def test_reload_restores_paired_weights_and_training_provenance(tmp_path):
    checkpoint = _write_checkpoint(tmp_path)

    predictor = PairedTAMPredictor.from_model_folder(tmp_path, 0, 'checkpoint_final.pth', 'cpu')

    assert predictor.network.tam_metadata['stages'] == [2]
    assert predictor.network.decoder.deep_supervision is False
    for name, restored in predictor.network.state_dict().items():
        torch.testing.assert_close(restored, checkpoint['network_weights'][name])
    assert predictor.checkpoint_metadata['paired_tam_training'] == {'split_sha256': 'example'}


def test_reload_rejects_nonstandard_mirroring_even_when_metadata_agrees(tmp_path):
    _write_checkpoint(tmp_path, mirror_axes=(0,))

    with pytest.raises(ValueError, match='mirroring axes'):
        PairedTAMPredictor.from_model_folder(tmp_path, 0, 'checkpoint_final.pth', 'cpu')
