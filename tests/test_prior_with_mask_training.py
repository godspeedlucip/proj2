import torch
import pytest
import numpy as np
import json
import hashlib

import train_prior_with_mask as training
from utils.sophia import SophiaG

from train_prior_with_mask import (
    choose_selection_metric,
    load_model_weights,
    save_checkpoint,
    stage_indices,
    train_epoch,
    validate_epoch,
)


class TinyTwoInput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, image, prior):
        return None, image * 0 + prior * 0 + self.bias


def sample(*, masked=False, quality=1, target=None):
    image = torch.zeros(1, 1, 7, 7)
    if target is None:
        target = torch.ones_like(image)
    batch = {"input": image, "prior": image.clone(), "label": target}
    if masked:
        mask = torch.zeros_like(image)
        mask[:, :, 2:5, 2:5] = 1
        batch["mask"] = mask
        batch["mask_quality"] = torch.tensor([quality])
    return batch


def config(mode="boundary"):
    return {
        "mode": mode,
        "radius": 1,
        "lambda_exact": 1.0,
        "lambda_rough": 0.5,
        "lambda_boundary": 1.0,
        "lambda_gradient": 0.1,
        "rough_sigma": 1.0,
    }


def test_base_stage_ignores_mask_and_mask_loss():
    model = TinyTwoInput()
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    result = train_epoch(model, [sample(masked=True)], optimizer,
                         stage="base", loss_config=config(), device=torch.device("cpu"))
    assert result["loss_full"] == pytest.approx(1.0)
    assert result["loss_total"] == pytest.approx(1.0)


def test_finetune_stage_rejects_unmasked_sample():
    model = TinyTwoInput()
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    with pytest.raises(ValueError, match="mask"):
        train_epoch(model, [sample()], optimizer,
                    stage="finetune", loss_config=config(), device=torch.device("cpu"))


def test_sophia_curvature_state_is_updated_during_training():
    model = TinyTwoInput()
    optimizer = SophiaG(model.parameters(), lr=1e-4)
    train_epoch(model, [sample()], optimizer,
                stage="base", loss_config=config(), device=torch.device("cpu"))
    assert optimizer.state[model.bias]["hessian"].abs().sum().item() > 0


def test_finetune_boundary_stage_adds_only_exact_components():
    model = TinyTwoInput()
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    result = train_epoch(model, [sample(masked=True)], optimizer,
                         stage="finetune", loss_config=config(), device=torch.device("cpu"))
    assert result["loss_full"] == pytest.approx(1.0)
    assert result["loss_roi_exact"] == pytest.approx(1.0)
    assert result["loss_boundary"] == pytest.approx(1.0)
    assert result["loss_gradient"] == 0
    assert result["loss_total"] == pytest.approx(3.0)


def test_training_metrics_expose_weighted_terms_and_quality_counts():
    batch = sample(masked=True)
    for key in ("input", "prior", "label", "mask"):
        batch[key] = torch.cat([batch[key], batch[key]], dim=0)
    batch["mask_quality"] = torch.tensor([1, -1])
    model = TinyTwoInput()
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    result = train_epoch(model, [batch], optimizer, stage="finetune",
                         loss_config=config(), device=torch.device("cpu"))
    assert (result["exact_count"], result["rough_count"], result["unknown_count"]) == (1, 0, 1)
    assert result["loss_weighted_roi_exact"] == pytest.approx(result["loss_roi_exact"])
    assert result["loss_weighted_roi_rough"] == pytest.approx(0.5 * result["loss_roi_rough"])
    assert result["loss_weighted_boundary"] == pytest.approx(result["loss_boundary"])


def test_validation_reports_full_population_and_exact_boundary_metrics():
    model = TinyTwoInput()
    full = validate_epoch(model, [sample()], stage="base", loss_config=config(),
                          device=torch.device("cpu"))
    target = torch.zeros(1, 1, 7, 7)
    target[:, :, 2:5, 2:5] = 1
    masked = validate_epoch(model, [sample(masked=True, target=target)],
                            stage="finetune", loss_config=config(),
                            device=torch.device("cpu"))
    assert full["full_mse"] == pytest.approx(1.0)
    assert full["full_mae"] == pytest.approx(1.0)
    assert masked["full_mse"] == pytest.approx(9 / 49)
    assert masked["roi_exact_mse"] == pytest.approx(1.0)
    assert masked["roi_exact_mae"] == pytest.approx(1.0)
    assert masked["boundary_mse"] == pytest.approx(8 / 24)
    assert masked["boundary_mae"] == pytest.approx(8 / 24)
    assert masked["boundary_gradient_mae"] > 0
    assert masked["exact_count"] == 1


def test_validation_averages_each_case_before_population(tmp_path):
    model = TinyTwoInput()
    batch = sample()
    for key in ("input", "prior", "label"):
        batch[key] = torch.cat([batch[key], batch[key], batch[key]], dim=0)
    batch["label"][2].zero_()
    batch["case_id"] = ["patient_a", "patient_a", "patient_b"]
    metrics = validate_epoch(model, [batch], stage="base", loss_config=config(),
                             device=torch.device("cpu"))
    assert metrics["num_samples"] == 3
    assert metrics["num_cases"] == 2
    assert metrics["full_mse"] == pytest.approx(0.5)


def test_base_subsample_is_reproducible_and_finetune_uses_every_sample():
    first = stage_indices(100, "base", seed=7, epoch=3)
    second = stage_indices(100, "base", seed=7, epoch=3)
    assert first == second
    assert len(first) == 10
    assert len(set(first)) == 10
    assert stage_indices(100, "finetune", seed=7, epoch=3) == list(range(100))


def test_old_checkpoint_loads_weights_and_new_checkpoint_records_stage(tmp_path):
    source = TinyTwoInput()
    source.bias.data.fill_(2.0)
    old_path = tmp_path / "old.dat"
    torch.save({"state_dict": source.state_dict()}, old_path)

    restored = TinyTwoInput()
    restored.bias.data.fill_(-1.0)
    load_model_weights(restored, old_path, torch.device("cpu"))
    assert restored.bias.item() == 2.0

    optimizer = torch.optim.SGD(restored.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
    new_path = tmp_path / "finetune.pt"
    save_checkpoint(new_path, restored, optimizer, scheduler, epoch=4,
                    stage="finetune", loss_mode="boundary",
                    config={"seed": 7}, validation_metrics={"boundary_mse": 0.5})
    saved = torch.load(new_path, map_location="cpu", weights_only=False)
    assert saved["stage"] == "finetune"
    assert saved["loss_mode"] == "boundary"
    assert saved["epoch"] == 4
    assert saved["state_dict"]["bias"].item() == 2.0


def test_finetune_checkpoint_cannot_seed_another_ablation(tmp_path):
    model = TinyTwoInput()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
    path = tmp_path / "finetune.pt"
    save_checkpoint(path, model, optimizer, scheduler, epoch=1,
                    stage="finetune", loss_mode="roi", config={}, validation_metrics={})
    with pytest.raises(ValueError, match="base"):
        load_model_weights(TinyTwoInput(), path, torch.device("cpu"))


def test_selection_prefers_exact_boundary_and_marks_rough_only_unassessed():
    score, name = choose_selection_metric(
        {"exact_count": 2, "boundary_mse": 0.3, "roi_rough_mse": 0.1}
    )
    assert (score, name) == (0.3, "boundary_mse")
    score, name = choose_selection_metric(
        {"exact_count": 0, "rough_count": 3, "roi_rough_mse": 0.6}
    )
    assert (score, name) == (0.6, "roi_rough_mse")
    score, name = choose_selection_metric(
        {"exact_count": 0, "rough_count": 0, "unknown_count": 2, "roi_rough_mse": 0.7}
    )
    assert (score, name) == (0.7, "roi_rough_mse")


def test_boundary_mode_rejects_training_set_without_exact_masks(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "Dense2D", TinyTwoInput)
    sample_path = tmp_path / "case01_p0_slice0.npz"
    mask = np.zeros((7, 7), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    np.savez(sample_path, input=np.zeros((7, 7), dtype=np.float32),
             prior=np.zeros((7, 7), dtype=np.float32),
             label=np.ones((7, 7), dtype=np.float32),
             mask=mask, mask_quality=np.array(0, dtype=np.int8))
    sample_list = tmp_path / "samples.txt"
    sample_list.write_text(str(sample_path) + "\n", encoding="utf-8")
    checkpoint = tmp_path / "base.dat"
    torch.save({"state_dict": TinyTwoInput().state_dict()}, checkpoint)
    with pytest.raises(ValueError, match="exact"):
        training.main([
            "--stage", "finetune",
            "--base-valid-list", str(sample_list),
            "--masked-train-list", str(sample_list),
            "--masked-valid-list", str(sample_list),
            "--base-checkpoint", str(checkpoint),
            "--output-dir", str(tmp_path / "output"),
            "--num-workers", "0",
            "--batch-size", "1",
            "--finetune-epochs", "1",
            "--loss-mode", "boundary",
        ])


def test_both_and_standalone_finetune_reproduce_and_record_source(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "Dense2D", TinyTwoInput)
    sample_path = tmp_path / "case01_p0_slice0.npz"
    mask = np.zeros((7, 7), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    np.savez(sample_path, input=np.zeros((7, 7), dtype=np.float32),
             prior=np.zeros((7, 7), dtype=np.float32),
             label=np.ones((7, 7), dtype=np.float32), mask=mask,
             mask_quality=np.array(1, dtype=np.int8))
    full_list = tmp_path / "full.txt"
    masked_list = tmp_path / "masked.txt"
    full_list.write_text((str(sample_path) + "\n") * 10, encoding="utf-8")
    masked_list.write_text(str(sample_path) + "\n", encoding="utf-8")
    common = ["--base-valid-list", str(masked_list),
              "--masked-train-list", str(masked_list),
              "--masked-valid-list", str(masked_list),
              "--device", "cpu", "--num-workers", "0", "--batch-size", "1",
              "--base-epochs", "2", "--finetune-epochs", "1",
              "--loss-mode", "boundary", "--boundary-radius", "1", "--seed", "7"]
    both = tmp_path / "both"
    training.main(["--stage", "both", "--base-train-list", str(full_list),
                   "--output-dir", str(both), *common])
    base_logs = [json.loads(line) for line in
                 (both / "base/metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["cumulative_samples_seen"] for row in base_logs] == [1, 2]
    base_path = both / "base/latest.pt"
    standalone = tmp_path / "standalone"
    training.main(["--stage", "finetune", "--base-checkpoint", str(base_path),
                   "--output-dir", str(standalone), *common])
    both_log = json.loads((both / "finetune/boundary/metrics.jsonl").read_text().splitlines()[0])
    alone_log = json.loads((standalone / "finetune/boundary/metrics.jsonl").read_text().splitlines()[0])
    assert both_log["train"] == alone_log["train"]
    saved = torch.load(both / "finetune/boundary/latest.pt", weights_only=False)
    assert saved["config"]["base_checkpoint_sha256"] == hashlib.sha256(base_path.read_bytes()).hexdigest()
    assert saved["config"]["base_checkpoint_path"] == str(base_path.resolve())
