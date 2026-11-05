import pytest
import torch

from utils.heart_mask_loss import heart_reconstruction_loss


def centered_mask(batch=1):
    mask = torch.zeros(batch, 1, 7, 7)
    mask[:, :, 2:5, 2:5] = 1
    return mask


def loss(pred, target, mask, quality, mode="gradient", **weights):
    return heart_reconstruction_loss(
        pred,
        target,
        mask,
        quality,
        mode=mode,
        radius=1,
        lambda_exact=weights.get("lambda_exact", 0.0),
        lambda_rough=weights.get("lambda_rough", 0.0),
        lambda_boundary=weights.get("lambda_boundary", 1.0),
        lambda_gradient=weights.get("lambda_gradient", 1.0),
        rough_sigma=1.0,
    )


def test_equal_images_have_zero_loss_in_every_mode():
    image = torch.arange(49, dtype=torch.float32).reshape(1, 1, 7, 7)
    for mode in ("full", "roi", "boundary", "gradient"):
        result = loss(image, image, centered_mask(), torch.tensor([1]), mode)
        assert result["total"].item() == 0


def test_boundary_band_mse_is_normalized_by_band_pixels():
    pred = torch.zeros(1, 1, 7, 7)
    pred[0, 0, 2, 3] = 1
    result = loss(pred, torch.zeros_like(pred), centered_mask(), torch.tensor([1]),
                  mode="boundary", lambda_gradient=0)
    assert result["full"].item() == pytest.approx(1 / 49)
    assert result["boundary"].item() == pytest.approx(1 / 24)
    assert result["total"].item() == pytest.approx(1 / 49 + 1 / 24)


def test_error_outside_exact_band_only_affects_full_mse():
    pred = torch.zeros(1, 1, 7, 7)
    pred[0, 0, 0, 0] = 1
    result = loss(pred, torch.zeros_like(pred), centered_mask(), torch.tensor([1]),
                  mode="boundary", lambda_gradient=0)
    assert result["boundary"].item() == 0
    assert result["total"].item() == pytest.approx(1 / 49)


@pytest.mark.parametrize("quality", [0, -1])
def test_rough_and_unknown_never_use_mask_contour(quality):
    pred = torch.zeros(1, 1, 7, 7)
    pred[0, 0, 2, 3] = 1
    result = loss(pred, torch.zeros_like(pred), centered_mask(), torch.tensor([quality]),
                  lambda_rough=0.5)
    assert result["roi_rough"].item() > 0
    assert result["boundary"].item() == 0
    assert result["gradient"].item() == 0
    assert result["total"].item() > result["full"].item()


def test_no_exact_sample_has_finite_zero_boundary_components():
    pred = torch.ones(2, 1, 7, 7)
    result = loss(pred, torch.zeros_like(pred), centered_mask(2),
                  torch.tensor([0, -1]), lambda_rough=0.5)
    assert torch.isfinite(result["total"])
    assert result["boundary"].item() == 0
    assert result["gradient"].item() == 0


def test_sobel_boundary_term_backpropagates_into_image():
    pred = torch.zeros(1, 1, 7, 7, requires_grad=True)
    target = torch.zeros_like(pred)
    target[0, 0, 2:5, 2:5] = 1
    result = loss(pred, target, centered_mask(), torch.tensor([1]),
                  mode="gradient", lambda_gradient=0.1)
    assert result["gradient"].item() > 0
    result["total"].backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum().item() > 0


def test_bad_shape_and_mode_are_rejected():
    pred = torch.zeros(1, 1, 7, 7)
    with pytest.raises(ValueError):
        loss(pred, pred, torch.zeros(1, 1, 6, 7), torch.tensor([1]))
    with pytest.raises(ValueError):
        loss(pred, pred, centered_mask(), torch.tensor([1]), mode="dice")
