import json
from pathlib import Path
from shutil import copyfile

import numpy as np
import pytest
import SimpleITK as sitk
import torch

from nnunetv2.run.paired_tam import main
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerTAM import (
    PairedDataLoader,
    PhasewiseLoss,
)
from nnunetv2.evaluation.paired_tam import segmentation_metrics
from nnunetv2.utilities.paired_tam_io import write_image


def test_loss_averages_each_phase_with_deep_supervision():
    logits = torch.tensor([[[0.0, 2.0], [1.0, 0.0]], [[2.0, 0.0], [0.0, 1.0]]])
    target = torch.tensor([[1, 1], [0, 0]])
    loss = PhasewiseLoss(lambda outputs, targets: sum(
        torch.nn.functional.cross_entropy(output, target)
        for output, target in zip(outputs, targets)
    ))
    expected = (torch.nn.functional.cross_entropy(logits[:, 0], target[:, 0])
                + torch.nn.functional.cross_entropy(logits[:, 1], target[:, 1]))
    torch.testing.assert_close(loss([logits, logits], [target, target]), expected)


def test_hd95_uses_millimetres_and_reports_empty_masks():
    first = np.zeros((5, 6, 7), dtype=bool)
    second = first.copy()
    first[1, 3, 4] = True
    second[2, 3, 4] = True
    assert segmentation_metrics(first, second, (3, 2, 1)) == (0.0, 3.0)
    assert segmentation_metrics(first, np.zeros_like(first), (3, 2, 1)) == (0.0, None)


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return str(path)


@pytest.fixture
def paired_experiment(tmp_path, monkeypatch):
    monkeypatch.setenv("nnUNet_n_proc_DA", "0")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield create_experiment(tmp_path)
    finally:
        torch.set_num_threads(previous_threads)


def create_experiment(root):
    dataset = {"channel_names": {"0": "CT"}, "labels": {"background": 0, "heart": 1},
               "numTraining": 3, "file_ending": ".nii.gz"}
    config = {
        "data_identifier": "Reference_3d_fullres", "preprocessor_name": "DefaultPreprocessor",
        "batch_size": 1, "patch_size": [16, 8, 8], "spacing": [2.0, 1.0, 1.0],
        "median_image_size_in_voxels": [18, 10, 10], "batch_dice": True,
        "normalization_schemes": ["CTNormalization"], "use_mask_for_norm": [False],
        "resampling_fn_data": "resample_data_or_seg_to_shape",
        "resampling_fn_seg": "resample_data_or_seg_to_shape",
        "resampling_fn_probabilities": "resample_data_or_seg_to_shape",
        "resampling_fn_data_kwargs": {"is_seg": False, "order": 3, "order_z": 0},
        "resampling_fn_seg_kwargs": {"is_seg": True, "order": 1, "order_z": 0},
        "resampling_fn_probabilities_kwargs": {"is_seg": False, "order": 1, "order_z": 0},
        "architecture": {
            "network_class_name": "dynamic_network_architectures.architectures.unet.PlainConvUNet",
            "arch_kwargs": {
                "n_stages": 4, "features_per_stage": [8, 8, 16, 16],
                "conv_op": "torch.nn.modules.conv.Conv3d", "kernel_sizes": [[3]*3]*4,
                "strides": [[1]*3, [2]*3, [2]*3, [2]*3],
                "n_conv_per_stage": [1]*4, "n_conv_per_stage_decoder": [1]*3,
                "conv_bias": True, "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
                "norm_op_kwargs": {"eps": 1e-5, "affine": True},
                "dropout_op": None, "dropout_op_kwargs": None,
                "nonlin": "torch.nn.LeakyReLU", "nonlin_kwargs": {"inplace": True},
            },
            "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"],
        },
    }
    plans = {
        "dataset_name": "Dataset999_PairedSmoke", "plans_name": "Reference",
        "transpose_forward": [0, 1, 2], "transpose_backward": [0, 1, 2],
        "original_median_spacing_after_transp": [2.0, 1.0, 1.0],
        "original_median_shape_after_transp": [18, 10, 10],
        "image_reader_writer": "SimpleITKIO", "configurations": {"3d_fullres": config},
    }
    records = []
    for index in range(3):
        record = {"case_id": f"case{index}", "patient_id": f"patient{index}"}
        for phase_index, phase in enumerate(("phase1", "phase5")):
            data = np.full((18, 10, 10), -20.0, dtype=np.float32)
            label = np.zeros(data.shape, dtype=np.uint8)
            label[4:12, 2:7, 2 + phase_index:7 + phase_index] = 1
            data[label == 1] = 100 + index * 10 + phase_index * 30
            paths = {}
            for kind, array in (("image", data), ("label", label)):
                path = root / f"case{index}_{phase}_{kind}.nii.gz"
                image = sitk.GetImageFromArray(array)
                image.SetSpacing((1, 1, 2))
                write_image(image, path)
                paths[kind] = path.name
            record[phase] = paths
        records.append(record)
    return (
        write_json(root / "manifest.json", records),
        write_json(root / "splits.json", [{"train": ["case0"], "val": ["case1"], "test": ["case2"]}]),
        write_json(root / "reference.json", plans),
        write_json(root / "dataset_input.json", dataset),
    )


def test_paired_augmentation_keeps_both_labels_on_the_same_grid(paired_experiment, tmp_path):
    manifest, splits, plans, dataset = paired_experiment
    prepared = tmp_path / "prepared"
    main(["prepare", manifest, splits, plans, dataset, str(prepared)])
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainerTAM import nnUNetTrainerTAMC5
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetNumpy

    properties = {"class_locations": {1: np.array([[0, 7, 4, 4]])}}
    marker = np.zeros((2, 18, 10, 10), dtype=np.int8)
    marker[:, 3:13, 1:9, 1:9] = 1
    folder = tmp_path / "shared"
    folder.mkdir()
    nnUNetDatasetNumpy.save_case(marker.astype(np.float32), marker, properties, str(folder / "case"))
    trainer = nnUNetTrainerTAMC5.from_prepared(prepared, tmp_path / "model", 0, torch.device("cpu"))
    rotation, dummy2d, initial, mirror = trainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
    patch = trainer.configuration_manager.patch_size
    transforms = trainer.get_training_transforms(
        patch, rotation, trainer._get_deep_supervision_scales(), mirror, dummy2d,
    )
    loader = PairedDataLoader(nnUNetDatasetNumpy(str(folder)), 1, initial, patch,
                              trainer.label_manager, transforms=transforms)
    for _ in range(3):
        batch = next(loader)
        assert batch["data"].shape == (1, 2, 1, *patch)
        assert len(batch["target"]) == 3
        assert torch.count_nonzero(batch["target"][0]) > 0
        for target in batch["target"]:
            assert set(target.unique().tolist()) <= {0, 1}
            torch.testing.assert_close(target[:, 0], target[:, 1])


def test_real_paired_entrypoints_train_resume_validate_and_predict(paired_experiment, tmp_path):
    manifest, splits, plans, dataset = paired_experiment
    prepared = tmp_path / "prepared"
    main(["prepare", manifest, splits, plans, dataset, str(prepared)])
    contracts = []
    for preset in ("C1", "C3", "C4", "C5"):
        model = tmp_path / preset
        main(["train", str(prepared), str(model), "--preset", preset, "--device", "cpu",
              "--epochs", "1", "--train-iterations", "1", "--val-iterations", "1"])
        saved = torch.load(model / "fold_0" / "checkpoint_final.pth", weights_only=False)
        contracts.append(saved["paired_tam"])
        assert (model / "fold_0" / "validation" / "case1_phase5.nii.gz").is_file()
    assert len({row["plans_sha256"] for row in contracts}) == 1

    model = tmp_path / "C5"
    main(["train", str(prepared), str(model), "--preset", "C5", "--device", "cpu",
          "--epochs", "2", "--train-iterations", "1", "--val-iterations", "1", "--resume"])
    checkpoint = torch.load(model / "fold_0" / "checkpoint_final.pth", weights_only=False)
    assert checkpoint["current_epoch"] == 2
    main(["evaluate", str(prepared), str(model), "--device", "cpu"])
    metrics = json.loads((model / "fold_0" / "test" / "summary.json").read_text(encoding="utf-8"))
    assert {row["patient_id"] for row in metrics["per_case"]} == {"patient2"}
    assert {row["phase"] for row in metrics["per_case"]} == {"phase1", "phase5"}
    output = tmp_path / "predictions"
    main(["predict", str(model), manifest, str(output), "--device", "cpu"])
    assert sorted(path.name for path in output.glob("*.nii.gz")) == [
        f"case{index}_{phase}.nii.gz" for index in range(3) for phase in ("phase1", "phase5")
    ]


def test_sparse_tam_trains_with_two_masks_and_exports_all_ten_phases(paired_experiment, tmp_path):
    manifest, splits, plans, dataset = paired_experiment
    context_phases = [f"phase{index}" for index in (2, 3, 4, 6, 7, 8, 9, 10)]
    entries = json.loads(Path(manifest).read_text(encoding="utf-8"))
    for record in entries:
        for phase in context_phases:
            filename = f"{record['case_id']}_{phase}_image.nii.gz"
            anchor = tmp_path / record["phase1"]["image"]
            if phase == "phase7":
                image = sitk.ReadImage(str(anchor))
                image.SetDirection((-1, 0, 0, 0, 1, 0, 0, 0, 1))
                image.SetOrigin((9, 0, 0))
                write_image(image, tmp_path / filename)
            else:
                copyfile(anchor, tmp_path / filename)
            record[phase] = {"image": filename}
    write_json(Path(manifest), entries)
    prepared = tmp_path / "sparse"
    main(["prepare", manifest, splits, plans, dataset, str(prepared), "--context-phases", *context_phases])
    with np.load(prepared / "Reference_3d_fullres" / "case0.npz") as cached:
        assert cached["data"].shape[0] == 10
        assert cached["seg"].shape[0] == 2

    from nnunetv2.training.nnUNetTrainer.nnUNetTrainerTAM import nnUNetTrainerTAMC5
    model = tmp_path / "sparse_model"
    trainer = nnUNetTrainerTAMC5.from_prepared(prepared, model, 0, torch.device("cpu"))
    trainer.initialize()
    train_loader, _ = trainer.get_dataloaders()
    batch = next(train_loader)
    assert batch["data"].shape[1:3] == (3, 1)
    assert all(target.shape[1] == 2 for target in batch["target"])
    inputs = batch["data"].requires_grad_()
    logits = trainer.network(inputs)
    assert logits[0].shape[1] == 3
    logits[0].retain_grad()
    trainer.loss(logits, batch["target"]).backward()
    # The context image affects the supervised predictions without its own segmentation loss.
    assert torch.count_nonzero(inputs.grad[:, 2]) > 0
    assert torch.count_nonzero(logits[0].grad[:, 2]) == 0
    main(["train", str(prepared), str(model), "--preset", "C5", "--device", "cpu",
          "--epochs", "1", "--train-iterations", "1", "--val-iterations", "1"])
    validation = model / "fold_0" / "validation"
    summary = json.loads((validation / "summary.json").read_text(encoding="utf-8"))
    assert {row["phase"] for row in summary["per_case"]} == {"phase1", "phase5"}

    test_case = entries[2]
    for phase in ("phase1", "phase5"):
        test_case[phase].pop("label")
    prediction_manifest = write_json(tmp_path / "image_only.json", [test_case])
    output = tmp_path / "sparse_predictions"
    main(["predict", str(model), prediction_manifest, str(output), "--device", "cpu"])
    assert {path.name for path in output.glob("*.nii.gz")} == {
        f"case2_phase{index}.nii.gz" for index in range(1, 11)
    }
    predicted = sitk.ReadImage(str(output / "case2_phase7.nii.gz"))
    source = sitk.ReadImage(str(tmp_path / test_case["phase7"]["image"]))
    assert predicted.IsSameImageGeometryAs(source)


def test_ablation_uses_one_recipe_and_reports_paired_patient_differences(paired_experiment, tmp_path):
    manifest, splits, plans, dataset = paired_experiment
    prepared = tmp_path / "prepared"
    main(["prepare", manifest, splits, plans, dataset, str(prepared)])
    output = tmp_path / "ablation"
    command = ["ablate", str(prepared), str(output), "--device", "cpu", "--seed", "23",
               "--epochs", "1", "--train-iterations", "1", "--val-iterations", "1"]
    main(command)
    report = json.loads((output / "fold_0" / "ablation_summary.json").read_text(encoding="utf-8"))
    assert report["recipe"]["seed"] == 23
    assert list(report["arms"]) == ["C1", "C3", "C4", "C5"]
    assert len({arm["settings"]["plans_sha256"] for arm in report["arms"].values()}) == 1
    assert {row["patient_id"] for row in report["paired_deltas_vs_C1"]["C5"]["per_patient"]} == {"patient2"}
    assert {row["phase"] for row in report["paired_deltas_vs_C1"]["C5"]["per_patient"]} == {"phase1", "phase5"}
    assert all(row["dice_delta"] is None or -1 <= row["dice_delta"] <= 1
               for row in report["paired_deltas_vs_C1"]["C5"]["per_patient"])
    with pytest.raises(ValueError, match="recipe"):
        main([*command, "--resume", "--seed", "24"])


def test_ablation_presets_share_initial_spatial_parameters(paired_experiment, tmp_path):
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainerTAM import nnUNetTrainerTAMC1, nnUNetTrainerTAMC5

    manifest, splits, plans, dataset = paired_experiment
    prepared = tmp_path / "prepared"
    main(["prepare", manifest, splits, plans, dataset, str(prepared)])
    trainers = [trainer.from_prepared(prepared, tmp_path / trainer.preset, 0, torch.device("cpu"))
                for trainer in (nnUNetTrainerTAMC1, nnUNetTrainerTAMC5)]
    for trainer in trainers:
        torch.manual_seed(17)
        trainer.initialize()
    for name, tensor in trainers[0].network.state_dict().items():
        torch.testing.assert_close(tensor, trainers[1].network.state_dict()[name])


def test_reference_staging_supports_mixed_windows_nifti_paths(paired_experiment, tmp_path):
    from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json

    manifest, splits, _, dataset = paired_experiment
    entries = json.loads(Path(manifest).read_text(encoding="utf-8"))
    original = tmp_path / entries[0]["phase5"]["image"]
    renamed = tmp_path / "心脏.nii.gz"
    original.rename(renamed)
    entries[0]["phase5"]["image"] = renamed.name
    write_json(tmp_path / "manifest.json", entries)
    output = tmp_path / "Dataset999_PairedReference"
    main(["reference-dataset", manifest, splits, dataset, str(output)])
    staged = json.loads((output / "dataset.json").read_text(encoding="utf-8"))
    assert set(staged["dataset"]) == {"case0_phase1", "case0_phase5"}
    example_image = staged["dataset"]["case0_phase1"]["images"][0]
    reader = determine_reader_writer_from_dataset_json(staged, example_image)()
    data, _ = reader.read_images([str(renamed)])
    assert data.shape == (1, 18, 10, 10)


def test_existing_ct_input_cannot_be_overwritten_by_prediction(paired_experiment, tmp_path):
    from nnunetv2.inference.paired_tam import check_prediction_destinations
    from nnunetv2.preprocessing.paired_tam import read_manifest

    manifest, _, _, _ = paired_experiment
    records = json.loads(Path(manifest).read_text(encoding="utf-8"))
    source = tmp_path / records[0]["phase1"]["image"]
    collision = tmp_path / "case2_phase5.nii.gz"
    collision.hardlink_to(source)
    with pytest.raises(ValueError, match="overwrite"):
        check_prediction_destinations([records[2]["case_id"]], read_manifest(manifest).values(),
                                      tmp_path, ".nii.gz")


def test_profile_checks_all_presets_with_one_patch_and_batch(paired_experiment, tmp_path):
    manifest, splits, plans, dataset = paired_experiment
    prepared = tmp_path / "prepared"
    main(["prepare", manifest, splits, plans, dataset, str(prepared)])
    output = tmp_path / "profile.json"
    main(["profile", str(prepared), str(output), "--device", "cpu", "--preset", "all", "--steps", "1"])
    report = json.loads(output.read_text(encoding="utf-8"))
    assert len(report["presets"]) == 4
    for result in report["presets"].values():
        assert result["settings"]["patch_size"] == result["settings"]["tile_size"] == [16, 8, 8]
        assert result["settings"]["batch_size"] == 1
        assert result["seconds_per_step"][0] > 0
        assert result["peak_cuda_allocated_bytes"] is None


def test_normal_nnunet_training_entrypoint_keeps_single_volume_behavior(paired_experiment, tmp_path, monkeypatch):
    from nnunetv2.run import run_training
    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetNumpy
    from nnunetv2.training.nnUNetTrainer import nnUNetTrainer as base

    manifest, splits, plans, dataset = paired_experiment
    prepared = tmp_path / "prepared"
    main(["prepare", manifest, splits, plans, dataset, str(prepared)])
    plans = json.loads((prepared / "plans.json").read_text(encoding="utf-8"))
    config = plans["configurations"]["3d_fullres"]
    ordinary_root = tmp_path / "standard_preprocessed"
    ordinary = ordinary_root / plans["dataset_name"]
    case_folder = ordinary / config["data_identifier"]
    case_folder.mkdir(parents=True)
    write_json(ordinary / "nnUNetPlans.json", plans)
    (ordinary / "dataset.json").write_bytes((prepared / "dataset.json").read_bytes())
    (ordinary / "splits_final.json").write_bytes((prepared / "splits_final.json").read_bytes())
    source = nnUNetDatasetNumpy(str(prepared / config["data_identifier"]))
    for case_id in ("case0", "case1"):
        data, seg, _, props = source.load_case(case_id)
        nnUNetDatasetNumpy.save_case(data[:1], seg[:1], props, str(case_folder / case_id))
    monkeypatch.setattr(run_training, "nnUNet_preprocessed", str(ordinary_root))
    monkeypatch.setattr(base, "nnUNet_preprocessed", str(ordinary_root))
    monkeypatch.setattr(base, "nnUNet_results", str(tmp_path / "standard_results"))
    monkeypatch.setenv("nnUNet_compile", "false")
    trainer = run_training.get_trainer_from_args(plans["dataset_name"], "3d_fullres", 0,
                                                device=torch.device("cpu"))
    trainer.initialize()
    train_loader, _ = trainer.get_dataloaders()
    batch = next(train_loader)
    assert batch["data"].shape == (1, 1, 16, 8, 8)
    assert np.isfinite(trainer.train_step(batch)["loss"])
    checkpoint_path = tmp_path / "standard.pth"
    trainer.save_checkpoint(str(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert "paired_tam" not in checkpoint
    trainer.load_checkpoint(checkpoint)
