import importlib.util
import numpy as np
import pytest


def test_pipeline_exists():
    assert importlib.util.find_spec('bird.train') is not None, 'training entrypoint must exist'


def test_data_geometry_and_validation(tmp_path):
    from bird.data import ProjectionDataset
    base=dict(projections=np.ones((2,1,1)),sources_mm=np.array([[-2.,0,0],[-2.,0,0]]),
              detector_centers_mm=np.array([[2.,0,0],[2.,0,0]]),detector_u_mm=np.array([[0.,1,0]]*2),
              detector_v_mm=np.array([[0.,0,1]]*2),times_s=np.array([0.,1.]),
              bounds_mm=np.array([[-1.,-1,-1],[1.,1,1]]),time_bounds_s=np.array([0.,1.]))
    path=tmp_path/'data.npz'
    np.savez(path,**base)
    dataset=ProjectionDataset(path)
    batch=dataset.batch(np.array([0,1]),'cpu')
    assert np.allclose(batch['directions'].numpy(),[[1,0,0],[1,0,0]])
    assert np.allclose(batch['times'].numpy().flatten(),[0,1])
    base['times_s']=np.array([0.,2.])
    np.savez(path,**base)
    with pytest.raises(ValueError,match='time'):
        ProjectionDataset(path)
    base['times_s']=np.array([0.,1.]);base['projections'][0,0,0]=np.nan
    np.savez(path,**base)
    with pytest.raises(ValueError,match='finite'):
        ProjectionDataset(path)


def test_general_rays_and_errors(tmp_path):
    from bird.data import ProjectionDataset
    base=dict(origins_mm=np.array([[-2.,0,0]]*2),directions=np.array([[2.,0,0]]*2),
              times_s=np.array([0.,1.]),projections=np.array([.03,.04]),ray_ids=np.array([2,5]),
              bounds_mm=np.array([[-1.,-1,-1],[1.,1,1]]),time_bounds_s=np.array([0.,1.]))
    path=tmp_path/'rays.npz';np.savez(path,**base)
    d=ProjectionDataset(path)
    assert np.allclose(d.batch(np.array([1]),'cpu')['directions'].numpy(),[[1,0,0]])
    base['directions'][0]=0;np.savez(path,**base)
    with pytest.raises(ValueError,match='direction'):
        ProjectionDataset(path)
    base['directions'][0]=[1,0,0];base['ray_ids']=np.array([2,2]);np.savez(path,**base)
    with pytest.raises(ValueError,match='ray_ids'):
        ProjectionDataset(path)


def test_training_resume_and_inference(tmp_path):
    import torch
    from bird.config import small_config
    from bird.demo import create_demo
    from bird.train import train
    from bird.infer import reconstruct
    torch.set_num_threads(1)
    data=tmp_path/'demo.npz';create_demo(data,views=6,detector=4,samples=16)
    cfg=small_config();cfg.iterations=4;cfg.rays_per_batch=8;cfg.samples_per_ray=8
    cfg.regularization_points=4;cfg.reg_chunk=4;cfg.log_every=1
    full=train(data,cfg,tmp_path/'full',device='cpu')
    train(data,cfg,tmp_path/'resume',device='cpu',stop_after=2)
    resumed=train(data,cfg,tmp_path/'resume',device='cpu',resume=True)
    a=torch.load(full,weights_only=True);b=torch.load(resumed,weights_only=True)
    assert a['step']==b['step']==4
    for key in a['model']:
        assert torch.equal(a['model'][key],b['model'][key]),key
    output=reconstruct(full,tmp_path/'volume',[0.,.5,1.],resolution=[4,5,6],chunk=17,device='cpu')
    with np.load(output) as v:
        assert v['mu'].shape==(3,6,5,4)
        assert v['backward_dvf_mm'].shape==(3,6,5,4,3)
        assert np.isfinite(v['mu']).all()
    with pytest.raises(FileExistsError):
        reconstruct(full,tmp_path/'volume',[0.],resolution=[4,4,4],device='cpu')
    with pytest.raises(ValueError,match='time'):
        reconstruct(full,tmp_path/'invalid.npz',[1.1],resolution=[4,4,4],device='cpu')
    with pytest.raises(FileExistsError):
        train(data,cfg,tmp_path/'full',device='cpu')
    bad=small_config();bad.iterations=5
    with pytest.raises(ValueError,match='config'):
        train(data,bad,tmp_path/'resume',device='cpu',resume=True)


def test_extensionless_output_does_not_overwrite(tmp_path):
    from bird.demo import create_demo
    path=create_demo(tmp_path/'scan',views=2,detector=2,samples=2)
    assert path.exists(), 'returned path must identify the actual file'
    old=path.read_bytes()
    with pytest.raises(FileExistsError):
        create_demo(tmp_path/'scan',views=3,detector=2,samples=2)
    assert path.read_bytes()==old


def test_float32_geometry_rejects_overflow(tmp_path):
    from bird.data import ProjectionDataset
    path=tmp_path/'overflow.npz'
    np.savez(path,origins_mm=np.array([[1e40,0,0]]),directions=np.array([[1.,0,0]]),
             times_s=np.array([0.]),projections=np.array([.03]),ray_ids=np.array([0]),
             bounds_mm=np.array([[-1.,-1,-1],[1.,1,1]]),time_bounds_s=np.array([0.,1.]))
    with pytest.raises(ValueError,match='float32'):
        d=ProjectionDataset(path)
        d.batch(np.array([0]),'cpu')


def test_resume_truncates_uncommitted_history(tmp_path):
    import json
    import torch
    from bird.config import small_config
    from bird.demo import create_demo
    from bird.train import train
    torch.set_num_threads(1)
    data=create_demo(tmp_path/'data.npz',views=2,detector=2,samples=4)
    cfg=small_config();cfg.iterations=2;cfg.rays_per_batch=4;cfg.samples_per_ray=4
    cfg.regularization_points=2;cfg.reg_chunk=2
    folder=tmp_path/'run'
    train(data,cfg,folder,stop_after=1)
    with (folder/'history.jsonl').open('a') as f:
        f.write(json.dumps({'step':2,'total':999})+'\n')
        f.write('{partial')
    train(data,cfg,folder,resume=True)
    rows=[json.loads(line) for line in (folder/'history.jsonl').read_text().splitlines()]
    assert [row['step'] for row in rows]==[1,2]
    assert rows[-1]['total']!=999


def test_metrics():
    assert importlib.util.find_spec('bird.evaluate') is not None, 'evaluation entrypoint must exist'
    from bird.evaluate import volume_metrics
    a=np.zeros((2,8,8));b=np.full_like(a,.1)
    metrics=volume_metrics(a,b,data_range=1.,ssim=False)
    assert abs(metrics['psnr']-20)<1e-8
    assert abs(metrics['mae']-.1)<1e-8
    with pytest.raises(ValueError):
        volume_metrics(a,b,data_range=0.)
