import os
import random
import torch.nn
from torch.utils.data import DataLoader
from torch.backends import cudnn
import time
import torch.optim as optim
from datetime import datetime
import tifffile
from torch.utils.tensorboard import SummaryWriter
import torch.autograd as autograd
import numpy as np
from tqdm import tqdm

def compute_gradient_penalty(D, real_samples, fake_samples):
    # print(real_samples.size())
    Tensor = torch.cuda.FloatTensor
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    if real_samples.ndim == 5:
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1, 1)))
    else:
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.tensor(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty



def extract_patches_online(tensor, num=2):

    if tensor.ndim == 5:

        split_w = torch.chunk(tensor, chunks=num, dim=3)
        stack_w = torch.reshape(torch.stack(split_w, dim=0),
                                [num*tensor.shape[0], tensor.shape[1],
                                 tensor.shape[2], tensor.shape[3]//num, tensor.shape[4]])
        split_h = torch.chunk(stack_w, chunks=num, dim=4)
        stack_h = torch.reshape(torch.stack(split_h, dim=0),
                                [num*num*tensor.shape[0], tensor.shape[1],
                                 tensor.shape[2], tensor.shape[3]//num, tensor.shape[4]//num])

        return stack_h

    elif tensor.ndim == 4:

        split_w = torch.chunk(tensor, chunks=num, dim=2)
        # print('split_w', split_w.size())
        stack_w = torch.reshape(torch.stack(split_w, dim=0),
                                [num*tensor.shape[0], tensor.shape[1],
                                 tensor.shape[2]//num, tensor.shape[3]])
        # print('stack_w', stack_w.size())
        split_h = torch.chunk(stack_w, chunks=num, dim=3)
        # print('split_h', split_h.size())
        stack_h = torch.reshape(torch.stack(split_h, dim=0),
                                [num*num*tensor.shape[0], tensor.shape[1],
                                 tensor.shape[2]//num, tensor.shape[3]//num])
        # print('stack_h', stack_h.size())

        return stack_h

    else:
        print('Expect for the tensor with dim==5 or 4, other cases are not yet implemented.')
