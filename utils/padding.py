import numpy as np
import os

def proj_padding(proj, padding_size):

    proj_width = np.size(proj, 2)
    proj_padded_width = padding_size * 2 + proj_width
    proj_padded = np.zeros([np.size(proj, 0), np.size(proj, 1), proj_padded_width], dtype=np.float32)
    proj_padded[:, :, padding_size:padding_size+proj_width] = proj

    for i in range(0, padding_size):
        proj_padded[:, :, i] = proj[:, :, 0] / padding_size * i
    for i in range(proj_width+padding_size, proj_padded_width):
        proj_padded[:, :, i] = proj[:, :, proj_width-1] / padding_size * (proj_padded_width-1-i)

    return proj_padded
