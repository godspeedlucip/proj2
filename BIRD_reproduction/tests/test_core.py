import importlib.util


def test_core_exists():
    assert importlib.util.find_spec('bird') is not None, 'BIRD package must exist'


def test_hash_interpolation_and_gradients():
    import torch
    from bird.encoding import HashGrid
    enc = HashGrid(1, levels=1, features=1, base=2, maximum=2, log2_size=4)
    with torch.no_grad():
        enc.tables[0][:, 0] = torch.tensor([0., 2., 4.])
    x = torch.tensor([[.125], [.375]], requires_grad=True)
    y = enc(x)
    assert torch.allclose(y[:, 0], torch.tensor([.5, 1.5]))
    y.sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x, 4.))
    assert enc.tables[0].grad.abs().sum() > 0


def test_model_both_paths_gradients():
    import torch
    from bird.config import small_config
    from bird.model import BIRD
    torch.manual_seed(7)
    model = BIRD(small_config())
    out = model(torch.rand(9, 3), torch.rand(9, 1))
    assert out['mu'].shape == out['mu_tp'].shape == (9, 1)
    assert out['backward'].shape == out['static'].shape == (9, 3)
    (out['mu'].sum()+out['mu_tp'].sum()).backward()
    for part in [model.backward_dvf, model.tp_encoder, model.ff_encoder, model.tp_head, model.final_head]:
        assert sum(p.grad.abs().sum().item() for p in part.parameters() if p.grad is not None) > 0
    assert all(p.grad is None for p in model.forward_dvf.parameters())


def test_fft_known_value_and_order():
    import torch
    from bird.losses import frequency_distance
    p = torch.tensor([1., 0., 0., 0.], requires_grad=True)
    y = torch.tensor([0., 1., 0., 0.])
    ids = torch.arange(4)
    expected = (2 + 2 * 2**.5) / 4
    loss = frequency_distance(p, y, ids)
    assert abs(loss.item()-expected) < 1e-6
    perm = torch.tensor([2, 0, 3, 1])
    assert torch.allclose(loss, frequency_distance(p[perm], y[perm], ids[perm]))
    loss.backward()
    assert torch.isfinite(p.grad).all()


def test_inverse_consistency_and_temporal_math():
    import torch
    from bird.losses import spatial_regularizer, temporal_regularizer
    x, t = torch.rand(16, 3, dtype=torch.float64), torch.rand(16, 1, dtype=torch.float64)
    a = 1.2
    forward = lambda x,t: (a-1)*x
    backward = lambda x,t: (1/a-1)*x
    assert spatial_regularizer(forward, backward, x, t, [.01]*3) < 1e-20
    zero = lambda x,t: torch.zeros_like(x)
    assert abs(spatial_regularizer(forward, zero, x, t, [.01]*3).item() - 3*.2**2) < 1e-10
    constant = lambda x,t: torch.ones_like(x)*.2
    assert spatial_regularizer(constant, zero, x, t, [.01]*3) < 1e-20
    quadratic = lambda x,t: t*t*torch.tensor([1.,2.,3.], dtype=x.dtype)
    assert abs(temporal_regularizer(quadratic, x, t, .01).item() - 56) < 1e-8
    linear = lambda x,t: t.expand_as(x)*3
    assert temporal_regularizer(linear, x, t, .01) < 1e-18


def test_registration_only_updates_forward():
    import torch
    from torch import nn
    from bird.losses import registration_loss
    class Field(nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = nn.Parameter(torch.tensor(value))
        def forward(self, x, t):
            return self.value.expand_as(x)
    class Analytic(nn.Module):
        def __init__(self):
            super().__init__()
            self.backward_dvf = Field([.1, .0, .0])
            self.forward_dvf = Field([.2, .0, .0])
            self.intensity = nn.Parameter(torch.tensor(2.))
        def topology(self, x, t):
            return (x+self.backward_dvf(x,t))[:, :1]*self.intensity
        def registration_parameters(self):
            return list(self.backward_dvf.parameters())+[self.intensity]
    model = Analytic()
    loss = registration_loss(model, torch.rand(7,3), torch.rand(7,1))
    assert abs(loss.item()-.6) < 1e-6
    loss.backward()
    assert model.forward_dvf.value.grad[0].abs() > 0
    assert model.backward_dvf.value.grad is None and model.intensity.grad is None
    assert all(p.requires_grad for p in model.parameters())


def test_constant_ray_integrals_and_parallel_misses():
    import torch
    from bird.geometry import sample_rays
    origins = torch.tensor([[-2.,0.,0.], [-2.,2.,0.], [0.,0.,0.]])
    directions = torch.tensor([[1.,0.,0.]]*3)
    bounds = torch.tensor([[-1.,-1.,-1.],[1.,1.,1.]])
    points, ds, valid = sample_rays(origins, directions, bounds, 32)
    assert torch.isfinite(points).all()
    assert torch.equal(valid, torch.tensor([True,False,True]))
    projections = torch.full((3,32), .02).mul(ds[:,None]).sum(-1)
    assert torch.allclose(projections, torch.tensor([.04,0.,.02]))


def test_chunk_projection_and_gradients():
    import torch
    from bird.config import small_config
    from bird.model import BIRD
    from bird.geometry import render_rays
    torch.manual_seed(5)
    model=BIRD(small_config())
    o=torch.tensor([[-2.,0.,0.],[-2.,.2,.1],[-2.,.3,.1]])
    d=torch.tensor([[1.,0.,0.]]*3)
    b=torch.tensor([[-1.,-1.,-1.],[1.,1.,1.]])
    t=torch.zeros(3,1)
    a=render_rays(model,o,d,t,b,8,ray_chunk=3,checkpoint_chunks=False)
    (a[0].sum()+a[1].sum()).backward()
    grad=[None if p.grad is None else p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    c=render_rays(model,o,d,t,b,8,ray_chunk=1,checkpoint_chunks=True)
    (c[0].sum()+c[1].sum()).backward()
    assert torch.allclose(a[0],c[0],atol=1e-7)
    for old,p in zip(grad,model.parameters()):
        if old is not None:
            assert torch.allclose(old,p.grad,atol=1e-6,rtol=1e-4)
