"""Journal Eq.15-22. Finite differences avoid second-order grid autograd."""
from contextlib import contextmanager
import torch


def frequency_distance(predicted,measured,ray_ids):
    predicted,measured,ray_ids=predicted.reshape(-1),measured.reshape(-1),ray_ids.reshape(-1)
    if len(predicted)==0 or predicted.shape!=measured.shape or predicted.shape!=ray_ids.shape:
        raise ValueError('frequency distance needs equal nonempty vectors')
    order=torch.argsort(ray_ids,stable=True)
    return (torch.fft.fft(predicted[order],norm='backward')-torch.fft.fft(measured[order],norm='backward')).abs().mean()


def data_fidelity(predicted,tp,measured,ids,lambda_tp):
    return lambda_tp*frequency_distance(tp,measured,ids)+(1-lambda_tp)*frequency_distance(predicted,measured,ids)


def spatial_regularizer(forward,backward,x,t,steps):
    def cycle_error(q):
        f=forward(q,t)
        return f+backward(q+f,t)
    jac=[]
    for axis,h in enumerate(steps):
        if h<=0:
            raise ValueError('spatial steps must be positive')
        delta=torch.zeros_like(x)
        delta[:,axis]=h
        jac.append((cycle_error(x+delta)-cycle_error(x-delta))/(2*h))
    return torch.stack(jac,-1).square().sum((-1,-2)).mean()


def temporal_regularizer(forward,x,t,step):
    if step<=0:
        raise ValueError('temporal step must be positive')
    acceleration=(forward(x,t+step)-2*forward(x,t)+forward(x,t-step))/step**2
    return acceleration.square().sum(-1).mean()


@contextmanager
def frozen_parameters(parameters):
    parameters=list(parameters)
    flags=[p.requires_grad for p in parameters]
    try:
        for p in parameters:
            p.requires_grad_(False)
        yield
    finally:
        for p,flag in zip(parameters,flags):
            p.requires_grad_(flag)


def registration_loss(model,x,t):
    with torch.no_grad():
        xs=x+model.backward_dvf(x,t)
        target=model.topology(x,t)
    xhat=xs+model.forward_dvf(xs,t)
    # Freeze parameters, not input derivatives. no_grad here would break training.
    with frozen_parameters(model.registration_parameters()):
        prediction=model.topology(xhat,t)
    return (prediction-target).abs().mean()
