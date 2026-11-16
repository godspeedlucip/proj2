"""Independent mathematical oracles for the journal equations, not code mirrors."""


def test_fft_dense_dft_oracle_values_and_gradients():
    import torch
    from bird.losses import frequency_distance
    for n in (3,5,8):
        g=torch.Generator().manual_seed(350+n)
        p=torch.randn(n,generator=g,dtype=torch.float64,requires_grad=True)
        y=torch.randn(n,generator=g,dtype=torch.float64)
        ids=torch.randperm(n,generator=g);order=torch.argsort(ids)
        k=torch.arange(n,dtype=torch.float64)
        matrix=torch.exp(-2j*torch.pi*k[:,None]*k[None,:]/n)
        oracle=(matrix@(p[order]-y[order]).to(torch.complex128)).abs().sum()/n
        actual=frequency_distance(p,y,ids)
        assert torch.allclose(actual,oracle,atol=1e-12,rtol=1e-12)
        actual_grad=torch.autograd.grad(actual,p,retain_graph=True)[0]
        oracle_grad=torch.autograd.grad(oracle,p)[0]
        assert torch.allclose(actual_grad,oracle_grad,atol=1e-12,rtol=1e-12)


def test_spatial_regularizer_full_affine_jacobian_and_gradients():
    import torch
    from bird.losses import spatial_regularizer
    dtype=torch.float64
    a=torch.tensor([[.2,.3,-.1],[.1,-.15,.25],[-.12,.06,.09]],dtype=dtype,requires_grad=True)
    b=torch.tensor([[-.1,.08,.04],[.02,.12,-.03],[.07,-.09,.05]],dtype=dtype,requires_grad=True)
    forward=lambda x,t:x@a.T+x.new_tensor([.1,-.05,.08])
    backward=lambda x,t:x@b.T+x.new_tensor([-.04,.02,.03])
    g=torch.Generator().manual_seed(120)
    x=torch.rand(13,3,generator=g,dtype=dtype);t=torch.rand(13,1,generator=g,dtype=dtype)
    actual=spatial_regularizer(forward,backward,x,t,[.01,.017,.023])
    # Derivative of f(x)+b(x+f(x)): A+B(I+A), including cross-axis entries.
    jacobian=a+b@(torch.eye(3,dtype=dtype)+a)
    oracle=jacobian.square().sum()
    assert torch.allclose(actual,oracle,atol=1e-12,rtol=1e-12)
    actual_grad=torch.autograd.grad(actual,(a,b),retain_graph=True)
    oracle_grad=torch.autograd.grad(oracle,(a,b))
    assert all(torch.allclose(u,v,atol=1e-11,rtol=1e-11) for u,v in zip(actual_grad,oracle_grad))


def test_actual_bird_registration_preserves_other_accumulated_gradients():
    import torch
    from bird.config import small_config
    from bird.model import BIRD
    from bird.losses import registration_loss
    torch.manual_seed(85);model=BIRD(small_config()).double()
    # Non-flat content: near-constant initial float32 values may quantize loss to zero.
    with torch.no_grad():
        for table in model.tp_encoder.tables:table.uniform_(-.5,.5)
        model.tp_head.second[-1].weight.mul_(20.)
        model.forward_dvf.head.second[-1].bias.copy_(torch.tensor([.04,-.03,.02],dtype=torch.float64))
    x=torch.rand(31,3,dtype=torch.float64)*.8+.1
    t=torch.rand(31,1,dtype=torch.float64)*.8+.1
    for p in model.parameters():p.grad=torch.ones_like(p)
    before={name:p.grad.clone() for name,p in model.named_parameters()}
    loss=registration_loss(model,x,t)
    assert loss.item()>0
    loss.backward();forward_change=0.
    for name,p in model.named_parameters():
        if name.startswith('forward_dvf.'):
            forward_change+=(p.grad-before[name]).abs().sum().item()
        else:assert torch.equal(p.grad,before[name]),name
    assert forward_change>0
    assert all(p.requires_grad for p in model.parameters())
