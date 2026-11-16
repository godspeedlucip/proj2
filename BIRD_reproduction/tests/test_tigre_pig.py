import importlib.util
import numpy as np
import pytest


@pytest.fixture
def synthetic_tigre_case(tmp_path):
    import json
    from bird.data import file_sha256
    p=tmp_path/'projections.npy';np.save(p,np.ones((2,2,3),np.float32))
    m=dict(schema='bird_tigre_case_v1',intensity_processing='hu_to_mu_valid_source_only_v1',
           uses_prior_image=False,uses_gt_supervision=False,
           projections=dict(path=p.name,sha256=file_sha256(p),shape=[2,2,3],complete_scan=False),
           acquisition=dict(n_views=600,duration_s=10.,angle_span_deg=300.),
           views=[dict(view_index=0,time_s=0.,angle_deg=0.),dict(view_index=300,time_s=5.,angle_deg=150.)],
           geometry=dict(dso_mm=750.,dsd_mm=1250.,n_detector_vu=[2,3],d_detector_vu_mm=[.616,.616],
             grid=dict(shape_zyx=[4,5,6],spacing_zyx_mm=[1.,1.,1.],size_zyx_mm=[4.,5.,6.],
                       offset_zyx_mm=[0.,0.,0.],bounds_xyz_mm=[[-3.,-2.5,-2.],[3.,2.5,2.]])))
    case=tmp_path/'case.json';case.write_text(json.dumps(m),encoding='utf-8')
    return case,m


@pytest.mark.parametrize('fault',['time','angle','view_id','bounds','spacing','detector','complete'])
def test_tigre_case_rejects_inconsistent_metadata(synthetic_tigre_case,fault):
    import json
    from bird.tigre_backend import TigreCase
    path,m=synthetic_tigre_case
    assert len(TigreCase(path))==2
    if fault=='time':m['views'][1]['time_s']=9.
    elif fault=='angle':m['views'][1]['angle_deg']=float('nan')
    elif fault=='view_id':m['views'][1]['view_index']=9999
    elif fault=='bounds':m['geometry']['grid']['bounds_xyz_mm']=[[0,0,0],[1,1,1]]
    elif fault=='spacing':m['geometry']['grid']['spacing_zyx_mm'][0]=2.
    elif fault=='detector':m['geometry']['d_detector_vu_mm'][0]=-1.
    else:m['projections']['complete_scan']=True
    path.write_text(json.dumps(m),encoding='utf-8')
    with pytest.raises(ValueError):TigreCase(path)


def test_tigre_inference_excludes_scan_endpoint(tmp_path):
    import torch
    from bird.config import small_config
    from bird.infer import reconstruct
    p=tmp_path/'end.pt'
    torch.save(dict(config=small_config().to_dict(),time_bounds_s=[0.,10.],tigre_options={}),p)
    with pytest.raises(ValueError,match='time'):
        reconstruct(p,tmp_path/'bad.npz',[10.])


def test_pig_modules_exist():
    assert importlib.util.find_spec('bird.pig_cbct') is not None
    assert importlib.util.find_spec('bird.tigre_backend') is not None


def test_tigre_training_and_gt_export_exist():
    assert importlib.util.find_spec('bird.train_tigre') is not None
    assert importlib.util.find_spec('bird.infer_pig') is not None


def test_gt_postprocessing_center_crop():
    from bird.infer_pig import postprocess_gt
    volume=np.arange(3*5*6,dtype=np.float32).reshape(3,5,6)
    grid=dict(intermediate_shape_zyx=[3,5,6],spacing_zyx_mm=[1,1,1],
              output_shape_zyx=[3,3,3],object_fov_vu_mm=[100,100],mask_shrink_pixels=0,
              postprocess='center_crop')
    result=postprocess_gt(volume,grid)
    assert np.array_equal(result,volume[:,1:4,1:4])


def test_physical_grid_and_hu():
    from bird.pig_cbct import hu_to_mu, acquisition_times, grid_spec
    assert np.allclose(hu_to_mu(np.array([-2048,-1000,0,1000],np.float32)),[0,0,.02,.04])
    times=acquisition_times(600,10)
    assert np.all(np.diff(times)>0)
    assert times[-1]==pytest.approx(599/60)
    g=grid_spec([560,512,512],[.25,.614,.614],dict(voxel_spacing_multiplier=1.5,
        shift_up_voxels=32,shift_down_voxels=0,shift_left_voxels=0,shift_right_voxels=0))
    assert np.allclose(g['offset_zyx_mm'],[0,29.472,0])
    assert np.allclose(g['size_zyx_mm'],[210,471.552,471.552])


def test_streaming_vjp_matches_full():
    import torch
    from bird.config import small_config
    from bird.model import BIRD
    from bird.tigre_backend import grid_points, materialize, volume_vjp
    torch.set_num_threads(1);torch.manual_seed(6)
    m=BIRD(small_config());shape=[4,5,6];time=.3
    xyz=grid_points(0,120,shape,'cpu')
    out=m(xyz,time)
    weight=torch.randn(2,4,5,6)
    full=torch.stack([out['mu'][:,0],out['mu_tp'][:,0]]).reshape(2,*shape)
    (full*weight).sum().backward()
    expected={n:p.grad.clone() for n,p in m.named_parameters() if p.grad is not None}
    m.zero_grad(set_to_none=True)
    actual=materialize(m,time,shape,17,'cpu')
    assert torch.allclose(actual,full,atol=1e-7)
    volume_vjp(m,time,shape,weight,19,'cpu')
    for n,p in m.named_parameters():
        if n in expected:
            assert torch.allclose(p.grad,expected[n],atol=1e-6,rtol=2e-4),n


def test_pig_plan_uses_scan_time_not_phase():
    from pathlib import Path
    root=Path(r'D:\desktop\课题2')
    if not (root/'generate_round1_contrast_gt.py').exists():
        pytest.skip('local pig study unavailable')
    from bird.pig_cbct import plan_case
    p=plan_case(root,1,'Control_1_7569/Heart_with_contrast_back',10.)
    assert p['acquisition']['n_views']==600
    assert len(p['mapping'])==600
    assert [r['view_index'] for r in p['mapping']]==list(range(600))
    assert p['mapping'][-1]['time_s']==pytest.approx(9.9833333333)
    assert p['mapping'][-1]['angle_deg']==299.5
    assert p['source_grid']['shape_zyx']==[560,512,512]
    assert p['evaluation_grid']['output_shape_zyx']==[467,256,256]
    times=[r['time_s'] for r in p['mapping'] if r['phase_label']==1]
    assert len(times)>1 and len(set(times))==len(times)
