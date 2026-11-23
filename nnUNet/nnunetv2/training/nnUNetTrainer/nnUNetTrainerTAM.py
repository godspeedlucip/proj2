import json
import os
import random
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np
import torch
from torch import nn

from nnunetv2.evaluation.paired_tam import evaluate_pairs
from nnunetv2.inference.paired_tam import PairedTAMPredictor
from nnunetv2.network_architecture.paired_tam import build_paired_network
from nnunetv2.preprocessing.paired_tam import read_manifest, validate_sources, validate_split
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetNumpy
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.paired_tam_contract import (
    CONFIGURATION,
    MIRROR_AXES,
    PHASES,
    json_digest,
    make_checkpoint_contract,
    validate_plans,
)
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as file:
        return json.load(file)


class _PreparedPairDataset(nnUNetDatasetNumpy):
    @staticmethod
    def unpack_dataset(folder, overwrite_existing=False, num_processes=1, verify=True):
        # The paired preprocessor already publishes the shared mmap files atomically.
        pass


class PairedDataLoader(nnUNetDataLoader):
    def generate_train_batch(self):
        batch = super().generate_train_batch()
        # The base loader crops and transforms both phases together as channels.
        batch["data"] = torch.as_tensor(batch["data"]).unsqueeze(2)
        target = batch["target"]
        if isinstance(target, list):
            batch["target"] = [level.unsqueeze(2) for level in target]
        else:
            batch["target"] = torch.as_tensor(target).unsqueeze(2)
        return batch


class PhasewiseLoss(nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    def forward(self, output, target):
        losses = []
        for phase_idx in range(len(PHASES)):
            if isinstance(output, (list, tuple)):
                losses.append(self.loss([level[:, phase_idx] for level in output],
                                        [level[:, phase_idx] for level in target]))
            else:
                losses.append(self.loss(output[:, phase_idx], target[:, phase_idx]))
        return torch.stack(losses).mean()


class _PairedTAMTrainer(nnUNetTrainer):
    data_loader_class = PairedDataLoader
    preset = None

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda"), *,
                 preprocessed_dataset_folder_base=None, output_folder_base=None):
        if configuration != CONFIGURATION or type(fold) is not int or fold < 0:
            raise ValueError("Paired TAM requires 3d_fullres and a non-negative prepared fold index.")
        if torch.distributed.is_initialized() or device.type not in ("cuda", "cpu"):
            raise ValueError("Paired TAM supports single-GPU CUDA training and CPU smoke tests.")
        manager = PlansManager(plans)
        validate_plans(manager, manager.get_configuration(configuration), dataset_json)
        super().__init__(
            plans, configuration, fold, dataset_json, device,
            preprocessed_dataset_folder_base=preprocessed_dataset_folder_base,
            output_folder_base=output_folder_base,
        )
        self.inference_allowed_mirroring_axes = MIRROR_AXES
        self.probabilistic_oversampling = True
        self.training_seed = None
        self.prepared = Path(self.preprocessed_dataset_folder_base)
        fingerprint = read_json(self.prepared / "dataset_fingerprint.json")
        if (fingerprint["fold"] != fold or fingerprint["phase_order"] != list(PHASES)
                or fingerprint["foreground_intensity_properties_per_channel"]
                != manager.foreground_intensity_properties_per_channel):
            raise ValueError("The prepared fold fingerprint does not match the supplied fold and plans.")
        if (json_digest(read_json(self.prepared / "plans.json")) != json_digest(plans)
                or json_digest(read_json(self.prepared / "dataset.json")) != json_digest(dataset_json)):
            raise ValueError("Use the plans.json and dataset.json from the prepared fold.")
        self.records = read_manifest(self.prepared / "paired_manifest.json", require_labels=True)
        self.fingerprint = fingerprint
        validate_sources(self.records, fingerprint)
        splits = read_json(self.prepared / "splits_final.json")
        self.split = validate_split(splits, self.records, fold)
        self.split_sha256 = json_digest(self.split)
        self.preparation_sha256 = json_digest({
            "manifest": self.records, "split": self.split, "fingerprint": fingerprint,
        })
        case_folder = Path(self.preprocessed_dataset_folder)
        if (set(nnUNetDatasetNumpy.get_identifiers(case_folder)) != self.records.keys()
                or any(not (case_folder / f"{case_id}.pkl").is_file() for case_id in self.records)):
            raise ValueError(f"Prepared case files do not match the manifest in {case_folder}.")

    @classmethod
    def from_prepared(cls, prepared_directory, model_directory, fold, device):
        prepared = Path(prepared_directory).resolve()
        return cls(read_json(prepared / "plans.json"), CONFIGURATION, fold, read_json(prepared / "dataset.json"),
                   device, preprocessed_dataset_folder_base=str(prepared),
                   output_folder_base=str(Path(model_directory).resolve()))

    @classmethod
    def build_network_architecture(cls, architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
                                   num_input_channels, num_output_channels, enable_deep_supervision=True):
        return build_paired_network(
            architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
            num_input_channels, num_output_channels, enable_deep_supervision, cls.preset,
        )

    def _do_i_compile(self):
        return False

    def _seed(self, offset):
        if self.training_seed is not None:
            seed = (self.training_seed + offset) % (2**32)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

    def initialize(self):
        self._seed(0)
        super().initialize()
        self.dataset_class = _PreparedPairDataset

    def on_train_epoch_start(self):
        self._seed(self.current_epoch + 1)
        super().on_train_epoch_start()

    def _build_loss(self):
        return PhasewiseLoss(super()._build_loss())

    def do_split(self):
        return list(self.split["train"]), list(self.split["val"])

    def plot_network_architecture(self):
        # nnU-Net's plotter passes a single (unpaired) volume.
        pass

    def checkpoint_contract(self):
        return make_checkpoint_contract(self.plans_manager, self.configuration_manager, self.dataset_json,
                                        self.network, self.fold, self.inference_allowed_mirroring_axes)

    def _checkpoint_extra_fields(self):
        return {
            "paired_tam": self.checkpoint_contract(),
            "paired_tam_training": {
                "split_sha256": self.split_sha256,
                "preparation_sha256": self.preparation_sha256,
                "seed": self.training_seed,
                "train_iterations": self.num_iterations_per_epoch,
                "val_iterations": self.num_val_iterations_per_epoch,
            },
        }

    def save_checkpoint(self, filename):
        if self.disable_checkpointing:
            return super().save_checkpoint(filename)
        with NamedTemporaryFile(dir=Path(filename).parent, suffix=".tmp", delete=False) as file:
            pending = Path(file.name)
        try:
            super().save_checkpoint(str(pending))
            os.replace(pending, filename)
        finally:
            pending.unlink(missing_ok=True)

    def load_checkpoint(self, filename_or_checkpoint):
        if not self.was_initialized:
            self.initialize()
        checkpoint = (filename_or_checkpoint if isinstance(filename_or_checkpoint, dict)
                      else torch.load(filename_or_checkpoint, map_location="cpu", weights_only=False))
        expected = self._checkpoint_extra_fields()
        if checkpoint.get("trainer_name") != type(self).__name__ or any(
                checkpoint.get(key) != value for key, value in expected.items()):
            raise ValueError("Checkpoint does not match the preset, plans, paired data, or training settings.")
        if tuple(checkpoint.get("inference_allowed_mirroring_axes") or ()) != MIRROR_AXES:
            raise ValueError("Checkpoint mirroring axes do not match the paired training configuration.")
        super().load_checkpoint(checkpoint)

    def on_train_start(self):
        if not self.was_initialized:
            self.initialize()
        self._seed(0)
        output = Path(self.output_folder_base)
        for name, expected in (("plans.json", self.plans_manager.plans), ("dataset.json", self.dataset_json),
                               ("splits_final.json", read_json(self.prepared / "splits_final.json"))):
            path = output / name
            if path.exists() and json_digest(read_json(path)) != json_digest(expected):
                raise ValueError(f"Model directory contains a different {name}: {path}")
        super().on_train_start()
        shutil.copyfile(self.prepared / "splits_final.json", output / "splits_final.json")
        settings = {
            **self._checkpoint_extra_fields(),
            "parameter_count": sum(parameter.numel() for parameter in self.network.parameters()),
            "num_epochs": self.num_epochs,
        }
        self.print_to_log_file("Paired TAM run settings", settings)
        self.print_to_log_file("Prepared data", str(self.prepared), "splits", self.split)
        with (Path(self.output_folder) / "paired_tam_run.json").open("w", encoding="utf-8") as file:
            json.dump(settings, file, indent=2)

    def validation_step(self, batch):
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [level.to(self.device, non_blocking=True) for level in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
            output = self.network(data)
            loss = self.loss(output, target)
        if isinstance(output, (tuple, list)):
            output, target = output[0], target[0]
        predictions = output.argmax(2)
        truth = target.squeeze(2)
        valid = truth != self.label_manager.ignore_label if self.label_manager.has_ignore_label else True
        phase_stats = []
        for label in self.label_manager.foreground_labels:
            pred_fg = (predictions == label) & valid
            true_fg = truth == label
            dims = (0, 2, 3, 4)
            phase_stats.append(torch.stack(((pred_fg & true_fg).sum(dims),
                                            (pred_fg & ~true_fg).sum(dims),
                                            (~pred_fg & true_fg).sum(dims))))
        counts = torch.stack(phase_stats, dim=-1).detach().cpu().numpy()  # [3, T, K_fg]
        tp, fp, fn = counts.sum(axis=1)
        return {"loss": loss.detach().cpu().numpy(), "tp_hard": tp, "fp_hard": fp, "fn_hard": fn,
                "phase_counts": counts}

    def on_validation_epoch_end(self, val_outputs):
        super().on_validation_epoch_end(val_outputs)
        tp, fp, fn = np.sum([output["phase_counts"] for output in val_outputs], axis=0)
        denominator = 2 * tp + fp + fn
        for idx, phase in enumerate(PHASES):
            dice = np.divide(2 * tp[idx], denominator[idx], where=denominator[idx] != 0,
                             out=np.full(tp.shape[1], np.nan))
            self.print_to_log_file(f"Online Dice {phase} per class", dice.tolist())

    def evaluate(self, split, save_probabilities=False):
        if not self.was_initialized:
            self.initialize()
        validate_sources(self.records, self.fingerprint)
        was_training = self.network.training
        deep_supervision = self.network.decoder.deep_supervision
        self.set_deep_supervision_enabled(False)
        try:
            predictor = self.make_predictor()
            predictor.checkpoint_metadata = {**self._checkpoint_extra_fields(), "current_epoch": self.current_epoch}
            summary = evaluate_pairs(predictor, [self.records[key] for key in self.split[split]], split,
                                     Path(self.output_folder) / ("validation" if split == "val" else split),
                                     save_probabilities, source_records=self.records.values())
            self.print_to_log_file("Native-geometry Dice and HD95 (mm)", summary["by_phase_and_structure"])
            return summary
        finally:
            self.set_deep_supervision_enabled(deep_supervision)
            self.network.train(was_training)

    def make_predictor(self):
        return PairedTAMPredictor(
            self.network, self.plans_manager, self.configuration_manager, self.dataset_json,
            self.device, self.inference_allowed_mirroring_axes,
        )

    def perform_actual_validation(self, save_probabilities=False):
        return self.evaluate("val", save_probabilities)

    def profile_training_step(self, batch, steps):
        self.network.train()
        timings = []
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        for _ in range(steps):
            start = perf_counter()
            result = self.train_step(batch)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            timings.append(perf_counter() - start)
            if not np.isfinite(result["loss"]):
                raise RuntimeError("A full training step produced a non-finite loss during the paired profile.")
        if not self.optimizer.state:
            raise RuntimeError("The paired profile did not complete an optimizer update. Try more steps.")
        return {
            "seconds_per_step": timings,
            "peak_cuda_allocated_bytes": (torch.cuda.max_memory_allocated(self.device)
                                          if self.device.type == "cuda" else None),
            "peak_cuda_reserved_bytes": (torch.cuda.max_memory_reserved(self.device)
                                         if self.device.type == "cuda" else None),
        }


class nnUNetTrainerTAMC1(_PairedTAMTrainer):
    preset = "C1"


class nnUNetTrainerTAMC3(_PairedTAMTrainer):
    preset = "C3"


class nnUNetTrainerTAMC4(_PairedTAMTrainer):
    preset = "C4"


class nnUNetTrainerTAMC5(_PairedTAMTrainer):
    preset = "C5"
