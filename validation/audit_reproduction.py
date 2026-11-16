"""Read-only audit of official notebooks; writes only local audit artifacts.

This is a validation harness, not the training system proposed in the Spec.
Run with the user's existing CUDA/TIGRE Python environment. No pig images
or external scripts are changed. Synthetic HDF5 fixtures use a documented
scale adapter solely to exercise the original /1000 dataset loaders.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback
import warnings

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "RSTAR" / "githubcode"
EXTERNAL = Path("D:/desktop/课题2")
sys.path.insert(0, str(EXTERNAL))

import h5py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from pytorch_msssim import ssim


def cells(stage):
    p = SOURCE / f"Training_Tetris{stage}.ipynb"
    return ["".join(c["source"]) for c in json.loads(p.read_text(encoding="utf-8"))["cells"]]


def namespace(stage):
    ns = dict(torch=torch, nn=nn, F=F, Dataset=Dataset, DataLoader=DataLoader,
              np=np, os=os, h5py=h5py, time=time, ssim=ssim,
              autocast=autocast, GradScaler=GradScaler)
    source_cells = cells(stage)
    for i in [3, 4, 8, 9, 10, 11, 13, 14]:
        exec(compile(source_cells[i], f"Training_Tetris{stage}.ipynb:cell{i}", "exec"), ns)
    return ns


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def z_name(name):
    return ".conv2." in name or ".conv2_2." in name


def check(result, name, fn):
    start = time.perf_counter()
    try:
        value = fn()
        item = {"status": "passed", "details": value}
    except Exception as exc:
        item = {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()}
    item["seconds"] = time.perf_counter() - start
    result["checks"][name] = item
    print(json.dumps({"check": name, **item}, ensure_ascii=False), flush=True)
    return item


def preflight():
    result = {"ready_to_run_unmodified": False}
    for stage in [1, 2]:
        ns = {}
        try:
            exec(compile(cells(stage)[1], f"stage{stage}:imports", "exec"), ns)
            result[f"stage{stage}_imports"] = "passed"
        except Exception as exc:
            result[f"stage{stage}_imports"] = f"{type(exc).__name__}: {exc}"
        result[f"stage{stage}_data_directories"] = {
            n: (SOURCE / "example_data" / f"{n}_data_stage{stage}").is_dir()
            for n in ["train", "eval"]}
    try:
        exec(cells(2)[18].splitlines()[-1], {})
    except Exception as exc:
        result["stage2_final_statement"] = f"{type(exc).__name__}: {exc}"
    result["active_checkpoint_save_calls"] = sum(
        isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and isinstance(n.func.value,ast.Name) and n.func.value.id == "torch" and n.func.attr == "save"
        for stage in [1,2]
        for source in cells(stage)[1:] if source.strip()
        for n in ast.walk(ast.parse(source)))
    result["production_python_modules_excluding_validation"] = [
        str(p.relative_to(ROOT)) for p in ROOT.rglob("*.py")
        if p.relative_to(ROOT).parts[0] != "validation"]
    return result


def weights_and_shapes():
    out = []
    for stage in [1, 2]:
        ns = namespace(stage)
        model = ns["myUNet"]().cuda().eval()
        state = torch.load(SOURCE / "checkpoints" / f"checkpoints_Tetris{stage}.pth",
                           map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        for t in [10, 11]:
            x = torch.rand(1, 2, t, 1 if stage == 1 else 16, 32, 32, device="cuda")
            with torch.no_grad(), autocast():
                y = model(x)
            assert y.shape == (1, 1, t, x.shape[3], 32, 32)
            assert torch.isfinite(y).all()
            out.append(dict(stage=stage, T=t, shape=list(y.shape),
                            parameters=sum(p.numel() for p in model.parameters()),
                            state_entries=len(state)))
        del model, state, x, y
        gc.collect()
        torch.cuda.empty_cache()
    return out


def simulate(out):
    from utils.tool import TigreTool
    from utils.phase_mapping import build_projection_phase_mapping
    from generate_pig_first_phase_prior import reconstruct_prior
    tool = TigreTool()
    geo = tool.set_geometry((32, 32, 32), (1, 1, 1), n_detector=(64, 64),
                            d_detector=(1, 1), dso=750, dsd=1250)
    angles = tool.set_angles(0, 300, 600)
    mapping = build_projection_phase_mapping(EXTERNAL / "data/cycle.csv", 11,
              min_phase=1, max_phase=11, n_projections=600, angle_span_deg=300)
    labels = np.array([m.phase_label for m in mapping])
    assert len(labels) == 600
    assert np.array_equal(np.bincount(labels)[1:], [28,62,60,60,61,58,61,60,60,62,28])
    z, y, x = np.indices((32,32,32), dtype=np.float32) - 15.5
    body = (z/10)**2 + (y/9)**2 + (x/9)**2 < 1
    mixed = np.zeros((600,64,64), dtype=np.float32)
    target = []
    def recon(p, a):
        return reconstruct_prior(p, geo, a, tool, padding_row_pixels=4,
            padding_col_pixels=4, fov_output_transverse_pixels=32,
            fov_source_pixels=32, fov_mask_diameter_shrink_pixels=2)[0]
    for phase in range(1,12):
        hu = np.full((32,32,32), -1000, dtype=np.float32)
        hu[body] = 0
        hu[((z-(phase-6)*.35)**2 + (y+1)**2 + (x-2)**2) < 3**2] = 500
        mu = tool.hu_to_mu(hu)
        p = tool.project(mu, geo, angles)
        selected = labels == phase
        mixed[selected] = p[selected]
        target.append(recon(p, angles))
    prior = recon(mixed, angles)
    cbct = np.stack([recon(mixed[labels == phase], angles[labels == phase])
                     for phase in range(1,12)])
    target = np.stack(target)
    assert cbct.shape == target.shape == (11,32,32,32)
    assert prior.shape == (32,32,32)
    assert all(np.isfinite(a).all() for a in [cbct,target,prior,mixed])
    for stage in [1,2]:
        directory = out / f"stage{stage}_data"
        directory.mkdir(parents=True, exist_ok=True)
        with h5py.File(directory / "phantom.h5", "w") as f:
            f.attrs["audit_scale_adapter"] = "stored=mu/0.02*1000; notebook divides by 1000"
            f.attrs["synthetic_only"] = True
            f.create_dataset("cbct", data=(cbct[:,16] if stage == 1 else cbct[:,8:24])/.02*1000)
            f.create_dataset("ct", data=(target[:,16] if stage == 1 else target[:,8:24])/.02*1000)
            f.create_dataset("prior", data=(prior[16] if stage == 1 else prior[8:24])/.02*1000)
    with h5py.File(out / "tigre_phantom_mu.h5", "w") as f:
        for name,a in dict(cbct=cbct,target=target,prior=prior,mixed_line_integral=mixed,
                           angles_rad=angles,state_label=labels).items():
            f.create_dataset(name,data=a,compression="gzip")
    return dict(T=11, views=600, angle_span=300, image_shape=list(cbct.shape),
                projections_shape=list(mixed.shape), all_finite=True,
                cbct_target_rmse_mu=float(np.sqrt(np.mean((cbct-target)**2))),
                real_pig_data=False, calibration="none", noise="none")


def training_pipeline(out):
    previous = None
    records = []
    for stage in [1,2]:
        ns = namespace(stage)
        model = ns["myUNet"]().cuda()
        if stage == 1:
            initial = torch.load(SOURCE / "checkpoints/checkpoints_Tetris1.pth",
                                 map_location="cpu", weights_only=True)
        else:
            initial = previous
        model.load_state_dict(initial, strict=True)
        ns["model"] = model
        if stage == 2:
            before = {k:v.clone() for k,v in model.state_dict().items() if not z_name(k)}
            exec(compile(cells(2)[16], "Training_Tetris2.ipynb:cell16", "exec"), ns)
            assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
            for k,v in model.state_dict().items():
                if z_name(k) and not (k in [f"conv{i}.conv2.weight" for i in range(1,5)]):
                    assert torch.count_nonzero(v).item() == 0, k
        data = ns["simudata"](str(out / f"stage{stage}_data"))
        loader = DataLoader(data,batch_size=1,num_workers=0)
        ns["loss_fn"] = nn.L1Loss()
        ns["scaler"] = GradScaler()
        opt = torch.optim.Adam(model.parameters(),lr=1.5e-4,betas=(.9,.999))
        old = {k:p.detach().clone() for k,p in model.named_parameters()}
        captured = io.StringIO()
        with warnings.catch_warnings(record=True) as ws, contextlib.redirect_stdout(captured):
            warnings.simplefilter("always")
            for epoch in [1,2]:
                ns["train1"](model,torch.device("cuda"),loader,opt,epoch)
            ns["eval1"](model,torch.device("cuda"),loader,2)
        changed = [k for k,p in model.named_parameters() if not torch.equal(old[k],p)]
        z_grad = [k for k,p in model.named_parameters() if z_name(k) and p.grad is not None]
        assert changed, "optimizer made no update"
        assert all(torch.isfinite(p).all() for p in model.parameters())
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        if stage == 1:
            assert not z_grad and not any(z_name(k) for k in changed)
        else:
            assert z_grad and any(z_name(k) for k in changed)
        model.eval()
        batch = next(iter(loader))
        cbct,target,prior = [a.cuda() for a in batch]
        inp = torch.cat([cbct,prior.unsqueeze(2).expand_as(cbct)],dim=1)
        with torch.no_grad(),autocast():
            predicted = model(inp)
        path = out / f"audit_stage{stage}_weights.pth"
        torch.save(model.state_dict(),path)
        reloaded = ns["myUNet"]().cuda().eval()
        state = torch.load(path,map_location="cpu",weights_only=True)
        reloaded.load_state_dict(state,strict=True)
        with torch.no_grad(),autocast():
            repeated = reloaded(inp)
        assert torch.equal(predicted,repeated)
        previous = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        records.append(dict(stage=stage, training_steps=2, input_shape=list(inp.shape),
            parameters_updated=len(changed), z_parameters_with_gradient=len(z_grad),
            save_reload_max_abs=float((predicted-repeated).abs().max()),
            audit_output_rmse_mu=float(((predicted.float()-target)*.02).square().mean().sqrt()),
            warnings=sorted(set(str(w.message) for w in ws)), log=captured.getvalue(),
            checkpoint_written_by="audit harness; original notebooks do not save by default"))
        del model,reloaded,opt,ns,inp,predicted,repeated,old,initial
        if stage == 2:
            del before
        gc.collect()
        torch.cuda.empty_cache()
    return records


def memory(shape):
    ns = namespace(2)
    model = ns["myUNet"]().cuda()
    state = torch.load(SOURCE / "checkpoints/checkpoints_Tetris2.pth",
                       map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    scaler = GradScaler()
    x = torch.rand(1,2,11,*shape,device="cuda")
    target = torch.rand(1,1,11,*shape,device="cuda")
    torch.cuda.reset_peak_memory_stats()
    losses=[]
    scales=[]
    initial = model.conv1.conv1.weight.detach().clone()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            y=model(x)
            loss=F.l1_loss(y,target)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        scaler.step(optimizer)
        scaler.update()
        assert torch.isfinite(loss)
        losses.append(float(loss))
        scales.append(scaler.get_scale())
    assert not torch.equal(initial,model.conv1.conv1.weight), "optimizer made no update"
    torch.cuda.synchronize()
    return dict(input_shape=list(x.shape), steps=2, loss=losses, scaler_scales=scales,
        initialization="official Stage II checkpoint", finite_gradients=True, parameter_update=True,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=["smoke","memory"],default="smoke")
    parser.add_argument("--shape",type=int,nargs=3,default=[16,128,128])
    args=parser.parse_args()
    out=ROOT/"validation"/"artifacts"/"2026-11-16"
    out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(2026)
    np.random.seed(2026)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    result={"python":sys.version,"torch":torch.__version__,
        "gpu":torch.cuda.get_device_name(0),"checks":{},
        "scope":"component/synthetic integration audit, not full paper or real-pig training",
        "sources":{str(p.relative_to(ROOT)):digest(p) for p in SOURCE.glob("*.ipynb")},
        "harness_sha256":digest(Path(__file__)),
        "external_sources":{str(EXTERNAL/p):digest(EXTERNAL/p) for p in
            ["utils/tool.py","utils/phase_mapping.py","generate_pig_first_phase_prior.py","data/cycle.csv"]}}
    if args.mode=="smoke":
        check(result,"preflight_observations_not_a_readiness_pass",preflight)
        check(result,"strict_weights_and_T10_T11_forward",weights_and_shapes)
        sim=check(result,"tigre_dynamic_phantom",lambda:simulate(out))
        if sim["status"]=="passed":
            check(result,"original_training_functions_and_transition",lambda:training_pipeline(out))
        filename="smoke_results.json"
    else:
        check(result,"stage2_fp16_training_memory",lambda:memory(args.shape))
        filename="memory_"+"x".join(map(str,args.shape))+".json"
    (out/filename).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("RESULT",out/filename,flush=True)
    return int(any(c["status"]=="failed" for c in result["checks"].values()))


if __name__=="__main__":
    raise SystemExit(main())
