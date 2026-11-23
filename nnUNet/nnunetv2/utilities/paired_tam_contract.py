import hashlib
import json
import math

import numpy as np

PHASES = ("phase1", "phase5")
CONFIGURATION = "3d_fullres"
TILE_STEP_SIZE = 0.5
MIRROR_AXES = (0, 1, 2)
TRAINER_PRESETS = {f"nnUNetTrainerTAM{preset}": preset for preset in ("C1", "C3", "C4", "C5")}


def json_digest(value):
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def validate_plans(plans_manager, configuration_manager, dataset_json):
    config = configuration_manager
    if config.previous_stage_name is not None or config.next_stage_names:
        raise ValueError("Paired TAM requires a standalone 3d_fullres configuration.")
    if plans_manager.plans.get("image_reader_writer", "SimpleITKIO") not in ("SimpleITKIO", "NibabelIO"):
        raise ValueError("Paired TAM plans use native axes and require SimpleITKIO or NibabelIO.")
    forward, backward = plans_manager.transpose_forward, plans_manager.transpose_backward
    if sorted(forward) != [0, 1, 2] or list(backward) != list(np.argsort(forward)):
        raise ValueError("Paired TAM requires inverse permutations of the three native spatial axes.")
    label_manager = plans_manager.get_label_manager(dataset_json)
    if label_manager.has_regions or not label_manager.foreground_labels:
        raise ValueError("Paired TAM requires mutually exclusive labels and at least one foreground class.")
    channels = dataset_json.get("channel_names", dataset_json.get("modality"))
    if channels is None or list(channels) != ["0"] or channels["0"].upper() != "CT":
        raise ValueError("Paired TAM requires a single CT channel.")
    if config.normalization_schemes != ["CTNormalization"]:
        raise ValueError("Paired TAM requires CTNormalization in 3d_fullres plans.")
    patch, strides = config.patch_size, config.pool_op_kernel_sizes
    if len(patch) != 3 or any(type(size) is not int or size < 1 for size in patch):
        raise ValueError("Paired TAM requires a positive integer patch_size of length three.")
    if len(config.spacing) != 3 or any(not math.isfinite(s) or s <= 0 for s in config.spacing):
        raise ValueError("Paired TAM requires three positive finite spacings.")
    if type(config.batch_size) is not int or config.batch_size < 1:
        raise ValueError("Paired TAM requires a positive integer batch_size.")
    if (len(strides) < 3 or any(len(stage) != 3 or any(type(s) is not int or s < 1 for s in stage)
                           for stage in strides) or list(strides[0]) != [1, 1, 1]):
        raise ValueError("Paired TAM ablations require at least three 3D encoder stages and a unit first stride.")
    scale = np.prod(strides, axis=0)
    if np.any(patch % scale):
        raise ValueError(f"patch_size {patch} must be divisible by the encoder stride {scale.tolist()}.")
    if np.prod(patch // scale) <= 1:
        raise ValueError("The deepest feature map needs more than one spatial voxel for instance normalization.")


def make_checkpoint_contract(plans_manager, configuration_manager, dataset_json, network, fold, mirror_axes):
    validate_plans(plans_manager, configuration_manager, dataset_json)
    return {
        "configuration": CONFIGURATION,
        "fold": int(fold),
        "plans_name": plans_manager.plans_name,
        "dataset_name": plans_manager.dataset_name,
        "plans_sha256": json_digest(plans_manager.plans),
        "dataset_sha256": json_digest(dataset_json),
        "patch_size": list(configuration_manager.patch_size),
        "tile_size": list(configuration_manager.patch_size),
        "batch_size": configuration_manager.batch_size,
        "spacing": list(configuration_manager.spacing),
        "encoder_strides": [list(stage) for stage in configuration_manager.pool_op_kernel_sizes],
        "tile_step_size": TILE_STEP_SIZE,
        "mirror_axes": None if mirror_axes is None else list(mirror_axes),
        "network": network.tam_metadata,
    }
