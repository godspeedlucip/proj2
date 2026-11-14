"""Actual TIGRE/CUDA known-motion phantom for the full MC update loop."""

import hashlib

import numpy as np
import pytest
import torch

from pnmc.config import ReconstructionConfig
from pnmc.data.schema import ArrayRef, CaseManifest, GridSpec
from pnmc.motion.warp import warp_pull
from pnmc.reconstruction.mc_ossart import reconstruct_target
from pnmc.reconstruction.projector import TigreProjector


def test_true_motion_reduces_64_cubed_phantom_error_vs_identity(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    grid = GridSpec((64, 64, 64), (1., 1., 1.), (-31.5, -31.5, -31.5),
                    ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)), "scanner")
    geometry = {"DSO_mm": 500., "DSD_mm": 900.,
                "nDetector_vu": [128, 128], "dDetector_vu_mm": [1., 1.],
                "offDetector_vu_mm": [0., 0.],
                "rotDetector_xyz_rad": [0., 0., 0.], "COR_mm": 0.,
                "offOrigin_zyx_mm": [0., 0., 0.]}
    angles = np.linspace(0, 2*np.pi, 20, endpoint=False, dtype=np.float32)
    phase_ids = ["a", "b", "c", "d"]
    phases = [phase_ids[i % 4] for i in range(len(angles))]
    projector = TigreProjector(geometry, grid, angles, device="cuda:0")
    z, y, x = np.mgrid[:64, :64, :64]
    phantom = (.035*np.exp(-((x-21)**2+(y-30)**2+(z-25)**2)/95.) +
               .045*np.exp(-((x-43)**2+(y-37)**2+(z-40)**2)/72.) +
               .02*np.exp(-((x-34)**2+(y-20)**2+(z-32)**2)/40.)).astype(np.float32)
    support = torch.zeros(grid.shape_zyx, dtype=torch.bool)
    support[6:58, 6:58, 6:58] = True
    truth = torch.from_numpy(phantom).to("cuda:0") * support.to("cuda:0")
    # Smoothly taper translations at Q0; the same fields generate and reconstruct.
    from scipy.ndimage import distance_transform_edt
    taper = torch.from_numpy(np.clip(distance_transform_edt(support.numpy()) / 8,
                                     0, 1).astype(np.float32)).to("cuda:0")
    true_fields = {}
    for phase, dx in zip(phase_ids, (0., -4., 4., 7.)):
        field = torch.zeros((3, *grid.shape_zyx), device="cuda:0")
        field[0] = dx * taper
        true_fields[phase] = field
    projections = np.empty((len(angles), 128, 128), dtype=np.float32)
    for phase in phase_ids:
        indices = np.asarray([i for i, p in enumerate(phases) if p == phase],
                             dtype=np.int64)
        moved = warp_pull(truth, true_fields[phase], grid, grid)
        projections[indices] = projector.project(moved, indices).cpu().numpy()
    path = tmp_path / "projection.npy"
    np.save(path, projections)
    ref = ArrayRef(path, None, "view,v,u", "line_integral",
                   hashlib.sha256(path.read_bytes()).hexdigest())
    case = CaseManifest("1", "phantom", "known_motion", "synthetic", phase_ids,
                        ref, angles.tolist(), phases, list(range(len(angles))),
                        geometry, grid, grid, {}, None,
                        {"normalization_status": "pending_train_fit"})
    initial = (.5 * truth).cpu()
    config = ReconstructionConfig(sweeps=20)
    true_result, true_history = reconstruct_target(
        case, "a", {key: value.cpu() for key, value in true_fields.items()},
        config, projector=projector, initial_mu=initial, support_mask=support,
        output=tmp_path / "true")
    identity_result, identity_history = reconstruct_target(
        case, "a", {key: torch.zeros_like(value).cpu() for key, value in true_fields.items()},
        config, projector=projector, initial_mu=initial, support_mask=support,
        output=tmp_path / "identity")
    truth_cpu = truth.cpu()
    roi = truth_cpu > .005
    true_rmse = torch.sqrt(torch.mean((true_result[roi]-truth_cpu[roi])**2)).item()
    identity_rmse = torch.sqrt(torch.mean((identity_result[roi]-truth_cpu[roi])**2)).item()
    print("64^3 TIGRE phantom ROI RMSE true={:.8g} identity={:.8g}; "
          "measurement L2 true={:.8g} identity={:.8g}".format(
              true_rmse, identity_rmse,
              true_history[-1]["measurement_residual_l2"],
              identity_history[-1]["measurement_residual_l2"]))
    assert len(true_history) == len(identity_history) == 20
    assert true_rmse < identity_rmse, (true_rmse, identity_rmse)
    assert np.isfinite(true_history[-1]["measurement_residual_l2"])
