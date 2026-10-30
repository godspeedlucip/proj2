import torch
import os
import scipy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
import time
import torch.optim as optim
import argparse
from utils.utils import load_model
from torchvision import models
from datetime import datetime
from math import exp
from torch.autograd import Variable
from utils import *
import sys
sys.path.append('..')
# from autoencoder_model import AutoEncoderPCCT

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class feature_extractor(nn.Module):

#     Sequential(
#     (0): Conv2d(5, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): BatchNorm2d(32, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
#     (2): LeakyReLU(negative_slope=0.01, inplace=True)
#     (3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (4): BatchNorm2d(32, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
#     (5): LeakyReLU(negative_slope=0.01, inplace=True)
#     (6): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (7): BatchNorm2d(32, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
#     (8): LeakyReLU(negative_slope=0.01, inplace=True)
#     (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
#     (10): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (11): BatchNorm2d(64, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
#     (12): LeakyReLU(negative_slope=0.01, inplace=True)
#     (13): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (14): BatchNorm2d(64, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
#     (15): LeakyReLU(negative_slope=0.01, inplace=True)
#     (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
#     (17): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (18): BatchNorm2d(128, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
#     (19): LeakyReLU(negative_slope=0.01, inplace=True)
#     (20): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (21): BatchNorm2d(128, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
#     (22): LeakyReLU(negative_slope=0.01, inplace=True)
#     (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
#     (24): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (25): BatchNorm2d(256, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
#     (26): LeakyReLU(negative_slope=0.01, inplace=True)
#     (27): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (28): BatchNorm2d(256, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
#     (29): LeakyReLU(negative_slope=0.01, inplace=True)
#     (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
#     (31): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (32): BatchNorm2d(512, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
#     (33): LeakyReLU(negative_slope=0.01, inplace=True)
#     (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (35): BatchNorm2d(512, eps=0.0001, momentum=0.1, affine=True, track_running_stats=True)
#     (36): LeakyReLU(negative_slope=0.01, inplace=True)
# )

    def __init__(self, gpu_id='3'):
        super(feature_extractor, self).__init__()

        model = AutoEncoderPCCT(in_chl=5, out_chl=5, model_chl=64)
        checkpoint = torch.load('/home/yikun/Workspace/TransferLearning/Denoising/runs/AutoEncoderPCCT/checkpoints/model_at_epoch_268.dat', map_location='cuda:'+gpu_id)
        model = load_model(model, checkpoint)
        # model.eval()

        self.layers = nn.Sequential(*list(model.encoder.children())[:38])

    def forward(self, img):
        
        out = [None] * (len(self.layers)+1)

        # print(len(self.layers)+1)

        # if img.shape[1] != 3:
        #     img = img.repeat(1, 3, 1, 1)

        out[0] = img

        for i in range(len(self.layers)):

            out[i+1] = self.layers[i](out[i])

        # return out[7], out[14], out[21], out[28], out[35]
        return out[9], out[16], out[23], out[30], out[37]


def contrastive_loss_calc(x, y, z, feature_extractor, eps=1e-7):

    # x: output, y: label, z: input

    xf1, xf2, xf3, xf4, xf5 = feature_extractor(x)
    yf1, yf2, yf3, yf4, yf5 = feature_extractor(y)
    zf1, zf2, zf3, zf4, zf5 = feature_extractor(z)

    # perceptual_loss_xy = F.mse_loss(xf1, yf1) + \
    #                      F.mse_loss(xf2, yf2) + \
    #                      F.mse_loss(xf3, yf3) + \
    #                      F.mse_loss(xf4, yf4) + \
    #                      F.mse_loss(xf5, yf5)

    # perceptual_loss_xz = F.mse_loss(xf1, zf1) + \
    #                      F.mse_loss(xf2, zf2) + \
    #                      F.mse_loss(xf3, zf3) + \
    #                      F.mse_loss(xf4, zf4) + \
    #                      F.mse_loss(xf5, zf5)

    perceptual_loss_xy = F.l1_loss(xf1, yf1) + \
                         F.l1_loss(xf2, yf2) + \
                         F.l1_loss(xf3, yf3) + \
                         F.l1_loss(xf4, yf4) + \
                         F.l1_loss(xf5, yf5)

    perceptual_loss_xz = F.l1_loss(xf1, zf1) + \
                         F.l1_loss(xf2, zf2) + \
                         F.l1_loss(xf3, zf3) + \
                         F.l1_loss(xf4, zf4) + \
                         F.l1_loss(xf5, zf5)
    
    return perceptual_loss_xy/(perceptual_loss_xz + eps)
    # return perceptual_loss_xy


def perceptual_loss_calc(x, y, feature_extractor):

    xf1, xf2, xf3, xf4, xf5 = feature_extractor(x)
    yf1, yf2, yf3, yf4, yf5 = feature_extractor(y)

    perceptual_loss = F.mse_loss(xf1, yf1) + \
                      F.mse_loss(xf2, yf2) + \
                      F.mse_loss(xf3, yf3) + \
                      F.mse_loss(xf4, yf4) + \
                      F.mse_loss(xf5, yf5)

    # perceptual_loss = F.l1_loss(xf1, yf1) + \
    #                   F.l1_loss(xf2, yf2) + \
    #                   F.l1_loss(xf3, yf3) + \
    #                   F.l1_loss(xf4, yf4) + \
    #                   F.l1_loss(xf5, yf5)

    return perceptual_loss

# x = torch.randn(1, 5, 512, 512)
# x = torch.FloatTensor(x).cuda()

# y = torch.randn(1, 5, 512, 512)
# y = torch.FloatTensor(y).cuda()

# FE =  feature_extractor()

# loss = perceptual_loss_calc(x, y, FE)
# print(loss)

# C1, C2, C3, C4, C5 = FE(x)
# D1, D2, D3, D4, D5 = FE(y)
# print(C1.size(),
#       C2.size(),
#       C3.size(),
#       C4.size(),
#       C5.size())

# img = torch.tensor(, dtype= torch.float)