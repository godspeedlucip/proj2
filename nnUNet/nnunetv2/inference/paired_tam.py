"""Sliding-window inference for annotated anchors and image-only target phases."""

from contextlib import nullcontext
from itertools import combinations, product
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import load_json, save_pickle

from nnunetv2.inference.sliding_window_prediction import compute_gaussian, compute_steps_for_sliding_window
from nnunetv2.utilities.paired_tam_contract import (
    CONFIGURATION,
    MIRROR_AXES,
    PHASES,
    TILE_STEP_SIZE,
    TRAINER_PRESETS,
    make_checkpoint_contract,
    planned_context_phases,
    validate_plans,
)
from nnunetv2.utilities.paired_tam_io import write_image
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


def check_prediction_destinations(case_ids, source_records, output_directory, file_ending,
                                  save_probabilities=False, *, phases=PHASES):
    sources = {
        Path(path).resolve() for record in source_records for phase in phases
        for path in record[phase].values()
    }
    source_files = set()
    for path in sources:
        if path.is_file():
            stat = path.stat()
            source_files.add((stat.st_dev, stat.st_ino))
    suffixes = (file_ending, ".npz", ".pkl") if save_probabilities else (file_ending,)
    output_directory = Path(output_directory)
    for case_id in case_ids:
        for phase in phases:
            for suffix in suffixes:
                destination = output_directory / f"{case_id}_{phase}{suffix}"
                if destination.resolve() in sources or destination.is_symlink():
                    raise ValueError(f"Prediction cannot overwrite an input file or symbolic link: {destination}")
                if destination.exists():
                    stat = destination.stat()
                    if (stat.st_dev, stat.st_ino) in source_files:
                        raise ValueError(f"Prediction cannot overwrite an input file: {destination}")


def _set_geometry(image, geometry):
    image.SetSpacing(geometry['spacing'])
    image.SetOrigin(geometry['origin'])
    image.SetDirection(geometry['direction'])


def _native_logits(cropped_logits, common_geometry, crop_start_xyz, native_geometry, outside_value):
    """Place the uncropped axes on the common grid and sample the native voxel centers."""
    direction = np.asarray(common_geometry['direction']).reshape(3, 3)
    offset = direction @ np.multiply(common_geometry['spacing'], crop_start_xyz)
    cropped_geometry = {
        **common_geometry,
        'size': tuple(reversed(cropped_logits.shape[1:])),
        'origin': tuple(np.asarray(common_geometry['origin']) + offset),
    }
    if all(tuple(cropped_geometry[key]) == tuple(native_geometry[key]) for key in cropped_geometry):
        return cropped_logits

    # Giving the cropped image its physical origin restores the crop without allocating
    # a separate full common-grid volume for each class.
    scores = sitk.GetImageFromArray(np.moveaxis(cropped_logits, 0, -1), isVector=True)
    _set_geometry(scores, cropped_geometry)
    native_scores = sitk.Resample(
        scores, native_geometry['size'], sitk.Transform(3, sitk.sitkIdentity), sitk.sitkLinear,
        native_geometry['origin'], native_geometry['spacing'], native_geometry['direction'],
        outside_value, scores.GetPixelID(),
    )
    return np.moveaxis(sitk.GetArrayFromImage(native_scores), -1, 0)


class PairedTAMPredictor:
    def __init__(self, network: torch.nn.Module, plans_manager: PlansManager,
                 configuration_manager: ConfigurationManager, dataset_json: dict,
                 device: str | torch.device, mirror_axes: tuple[int, ...] | None,
                 tile_step_size: float = TILE_STEP_SIZE):
        self.network = network
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.dataset_json = dataset_json
        self.checkpoint_metadata = None
        self.device = torch.device(device)
        self.patch_size = tuple(configuration_manager.patch_size)
        self.context_phases = planned_context_phases(configuration_manager)
        self.phase_order = PHASES + self.context_phases
        self.input_phase_count = len(PHASES) + bool(self.context_phases)
        self.tile_step_size = tile_step_size
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        if self.label_manager.has_regions:
            raise ValueError('Paired TAM prediction requires mutually exclusive labels.')
        if len(self.patch_size) != 3 or min(self.patch_size) < 1:
            raise ValueError('Paired TAM prediction requires a positive 3D patch_size.')
        if not 0 < self.tile_step_size <= 1:
            raise ValueError('tile_step_size must be in (0, 1].')

        mirror_axes = () if mirror_axes is None else tuple(mirror_axes)
        if not set(mirror_axes) <= {0, 1, 2} or len(set(mirror_axes)) != len(mirror_axes):
            raise ValueError('mirror_axes must contain distinct spatial axes from 0, 1, 2.')
        # Network inputs and outputs both have three leading dimensions before the spatial axes.
        spatial_dims = tuple(axis + 3 for axis in mirror_axes)
        self._mirror_combinations = [axes for count in range(1, len(spatial_dims) + 1)
                                     for axes in combinations(spatial_dims, count)]

    @classmethod
    def from_model_folder(cls, model_folder: str | Path, fold: int, checkpoint_name: str,
                          device: str | torch.device):
        from nnunetv2.network_architecture.paired_tam import build_paired_network

        folder = Path(model_folder)
        fold = int(fold)
        dataset_json = load_json(folder / 'dataset.json')
        plans_manager = PlansManager(folder / 'plans.json')
        checkpoint = torch.load(folder / f'fold_{fold}' / checkpoint_name, map_location='cpu', weights_only=False)
        trainer_name = checkpoint['trainer_name']
        if trainer_name not in TRAINER_PRESETS or checkpoint['init_args']['configuration'] != CONFIGURATION:
            raise ValueError('A paired TAM trainer checkpoint for 3d_fullres is required.')
        mirror_axes = tuple(checkpoint.get('inference_allowed_mirroring_axes') or ())
        if mirror_axes != MIRROR_AXES:
            raise ValueError(f'Paired TAM checkpoint mirroring axes must be {MIRROR_AXES}.')

        configuration_manager = plans_manager.get_configuration(CONFIGURATION)
        validate_plans(plans_manager, configuration_manager, dataset_json)
        label_manager = plans_manager.get_label_manager(dataset_json)
        network = build_paired_network(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            1,
            label_manager.num_segmentation_heads,
            enable_deep_supervision=False,
            preset=TRAINER_PRESETS[trainer_name],
        )
        expected = make_checkpoint_contract(plans_manager, configuration_manager, dataset_json, network,
                                            fold, mirror_axes)
        if checkpoint.get('paired_tam') != expected:
            raise ValueError('Paired TAM checkpoint metadata does not match the model settings and saved plans.')

        network.load_state_dict(checkpoint['network_weights'])
        predictor = cls(network, plans_manager, configuration_manager, dataset_json, device, mirror_axes)
        predictor.checkpoint_metadata = {
            key: checkpoint.get(key) for key in ('paired_tam', 'paired_tam_training', 'current_epoch')
        }
        return predictor

    def _predict_tile(self, tile):
        prediction = self.network(tile)
        expected_shape = (1, self.input_phase_count, self.label_manager.num_segmentation_heads, *self.patch_size)
        if not isinstance(prediction, torch.Tensor) or prediction.shape != expected_shape:
            raise ValueError(f'Paired TAM prediction requires a single output tensor with shape {expected_shape}. '
                             'Disable deep supervision before inference.')
        prediction = prediction.float()
        for axes in self._mirror_combinations:
            mirrored = self.network(tile.flip(axes)).flip(axes).float()
            prediction = prediction + mirrored
        return prediction / (len(self._mirror_combinations) + 1)

    @torch.inference_mode()
    def predict_logits(self, data: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Return CPU float32 logits of shape [T, K, D, H, W] in preprocessed space."""
        if data.ndim != 4 or data.shape[0] != self.input_phase_count or min(data.shape[1:]) < 1:
            raise ValueError(f'Expected data with {self.input_phase_count} phases and shape [T, D, H, W].')
        data = torch.as_tensor(data, device='cpu', dtype=torch.float32)
        data, undo_padding = pad_nd_image(data, self.patch_size, 'constant', {'value': 0}, True)
        spatial_shape = data.shape[1:]
        steps = compute_steps_for_sliding_window(spatial_shape, self.patch_size, self.tile_step_size)

        # Full-volume buffers stay on the CPU so GPU memory scales with the plans tile size.
        gaussian = compute_gaussian(self.patch_size, dtype=torch.float32, device=torch.device('cpu'))
        logits = torch.zeros((self.input_phase_count, self.label_manager.num_segmentation_heads, *spatial_shape),
                             dtype=torch.float32, device='cpu')
        weights = torch.zeros(spatial_shape, dtype=torch.float32, device='cpu')
        self.network.to(self.device).eval()
        precision = torch.autocast('cuda') if self.device.type == 'cuda' else nullcontext()
        with precision:
            for position in product(*steps):
                window = tuple(slice(start, start + size) for start, size in zip(position, self.patch_size))
                tile = data[(slice(None), *window)][None, :, None].contiguous().to(self.device)
                prediction = self._predict_tile(tile)[0].cpu()
                logits[(slice(None), slice(None), *window)] += prediction * gaussian
                weights[window] += gaussian
        logits.div_(weights)
        if not torch.isfinite(logits).all():
            raise RuntimeError('Paired TAM sliding-window prediction produced non-finite logits.')
        return logits[(slice(None), slice(None), *undo_padding[1:])]

    def predict_case(self, record: dict, output_directory: str | Path,
                     save_probabilities: bool = False) -> dict[str, str]:
        from nnunetv2.preprocessing.paired_tam import preprocess_case

        check_prediction_destinations([record["case_id"]], [record], output_directory,
                                      self.dataset_json["file_ending"], save_probabilities, phases=self.phase_order)
        data, _, properties = preprocess_case(record, self.plans_manager, self.configuration_manager,
                                              self.dataset_json, with_labels=False)
        if not self.context_phases:
            return self.export_prediction(self.predict_logits(data), properties, record['case_id'],
                                          output_directory, save_probabilities)

        paths = {}
        for index, phase in enumerate(self.context_phases):
            # Keep the anchor predictions from the first triplet, and each target from its own triplet.
            group = PHASES + (phase,)
            first_output = 0 if index == 0 else len(PHASES)
            logits = self.predict_logits(data[[*range(len(PHASES)), len(PHASES) + index]])
            paths.update(self.export_prediction(logits[first_output:], properties, record['case_id'],
                                                output_directory, save_probabilities, phases=group[first_output:]))
        return paths

    def export_prediction(self, logits: np.ndarray | torch.Tensor, properties: dict, case_id: str,
                          output_directory: str | Path, save_probabilities: bool = False, *,
                          phases=None) -> dict[str, str]:
        """Restore physical grids before converting scores to labels or probabilities."""
        if tuple(properties['phase_order']) != self.phase_order:
            raise ValueError(f'Case {case_id}: expected prediction phase order {self.phase_order}.')
        phases = self.phase_order if phases is None else phases
        if logits.ndim != 5 or logits.shape[:2] != (len(phases), self.label_manager.num_segmentation_heads):
            raise ValueError(f'Case {case_id}: expected logits with shape [T, K, D, H, W].')
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().float().numpy()
        common_geometry = properties['common_geometry']
        native_geometries = properties['original_geometries']
        transpose_forward = self.plans_manager.transpose_forward
        transpose_backward = self.plans_manager.transpose_backward
        spacing = [properties['spacing'][axis] for axis in transpose_forward]
        crop_shape = properties['shape_after_cropping_and_before_resampling']
        bbox = properties['bbox_used_for_cropping']
        crop_start_xyz = tuple(bbox[axis][0] for axis in reversed(transpose_backward))

        destination = Path(output_directory)
        destination.mkdir(parents=True, exist_ok=True)
        paths = {}
        for phase, phase_logits in zip(phases, logits):
            cropped = self.configuration_manager.resampling_fn_probabilities(
                phase_logits, crop_shape, self.configuration_manager.spacing, spacing,
            )
            cropped = np.asarray(cropped, dtype=np.float32).transpose((0, *(axis + 1 for axis in transpose_backward)))
            native_geometry = native_geometries[phase]
            # A zero-logit tie is background. Mark out-of-crop voxels when exporting
            # probabilities so they can instead have an exact background probability of one.
            scores = _native_logits(cropped, common_geometry, crop_start_xyz, native_geometry,
                                    float('nan') if save_probabilities else 0.0)
            segmentation = self.label_manager.convert_logits_to_segmentation(scores)
            if save_probabilities:
                outside = np.isnan(scores[0])
                segmentation[outside] = 0
            label_dtype = np.uint8 if self.label_manager.num_segmentation_heads <= 256 else np.uint16
            image = sitk.GetImageFromArray(segmentation.astype(label_dtype, copy=False))
            _set_geometry(image, native_geometry)

            stem = destination / f'{case_id}_{phase}'
            output = f'{stem}{self.dataset_json["file_ending"]}'
            write_image(image, output)
            paths[phase] = output
            if save_probabilities:
                probabilities = self.label_manager.apply_inference_nonlin(scores).cpu().numpy()
                probabilities[:, outside] = 0
                probabilities[0, outside] = 1
                np.savez_compressed(f'{stem}.npz', probabilities=probabilities)
                save_pickle({'spacing': native_geometry['spacing'][::-1], 'sitk_stuff': native_geometry},
                            f'{stem}.pkl')
        return paths
