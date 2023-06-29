################################################################################
# Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
# LEAP project developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# LivermorE AI Projector for Computed Tomography (LEAP)
# demo: test example of FBP reconstruction using projector class
################################################################################

import sys
sys.stdout.flush()
import argparse
import numpy as np
import imageio

import torch
import torch.nn as nn
from leaptorch import Projector


# program arguments
parser = argparse.ArgumentParser()
parser.add_argument("--proj-fn", default="sample_data/FORBILD_head_512_sino.npy", help="path to input projection data file")
parser.add_argument("--param-fn", default="sample_data/param_parallel512.cfg", help="path to projection geometry configuration file")
parser.add_argument("--out-fn", default="sample_data/FORBILD_head_512_fbp.npy", help="path to the output reconstructed image file")
parser.add_argument("--use-fov", action='store_true', default=False, help="whether fov is used or not")
args = parser.parse_args()



class FBP_Parallel(nn.Module):
    # modified from https://github.com/drgHannah/Radon-Transformation/blob/main/radon_transformation/radon.py
    def __init__(self, param_fn, use_gpu, gpu_device, use_mask, batch_size):
        super().__init__()
        # initialize projector and load parameters
        self.projector = Projector(forward_project=False, use_static=False, use_gpu=use_gpu, gpu_device=gpu_device, batch_size=batch_size)
        self.projector.load_param(param_fn)
        #views, rows, cols = projector.get_projection_dim()
        _, _, dimx = self.projector.get_volume_dim()
        self.filter_size_total = max(64, int(2 ** (2 * torch.tensor(dimx)).float().log2().ceil()))
        self.filter_size_pad = (self.filter_size_total - dimx)
        self.filter = self.ramp_filter(self.filter_size_total).to(gpu_device)
        self.use_mask = use_mask
        self.recon_mask = self.create_circular_mask(dimx, batch_size).to(gpu_device)

    def create_circular_mask(self, size, batch_size):
        c = int(size / 2) - 0.5
        r = min(c, size - c)
        grid_y, grid_x = np.ogrid[0:size, 0:size]
        dist = np.sqrt((grid_y - c)**2 + (grid_x - c)**2)
        mask = (dist <= r)
        mask = np.repeat(mask[np.newaxis, np.newaxis, :, :], batch_size, axis=0)
        return torch.from_numpy(mask)

    def ramp_filter(self, size):
        n = np.concatenate((np.arange(1,size/2+1,2,dtype=int), np.arange(size/2-1,0,-2,dtype=int)))
        f = np.zeros(size)
        f[0] = 0.25
        f[1::2] = -1 / (np.pi * n) ** 2
        return torch.tensor(2 * np.real(np.fft.fft(f)))

    def filter_project(self, sino): 
        # input sinogram: projection_angles x detector_row x detector_column/size
        sino_padded = torch.nn.functional.pad(sino, [0,self.filter_size_pad], mode='constant', value=0)
        projection = torch.fft.fft(sino_padded, dim=1) * self.filter[None,:].double()
        sino_filtered = torch.real(torch.fft.ifft(projection, dim=1))
        return sino_filtered[:,:sino.shape[1]]

    def forward(self, sino): 
        '''
        input sino: batch_size x projection_angles x detector_row(1) x detector_column
        ''' 
        batch, nviews, nrows, ncols = sino.shape
        sino_filtered = torch.zeros_like(sino)
        for n in range(batch):
            sino_filtered[n,:,0,:] = self.filter_project(sino[n,:,0,:])
        img = self.projector(sino_filtered.contiguous())
        #img = img_.clone() ########### need to fix it
        if self.use_mask:
            img = self.recon_mask * img
        img[img < 0] = 0
        return img



# read arguments
proj_fn = args.proj_fn
param_fn = args.param_fn
out_fn = args.out_fn
use_fov = args.use_fov

# if CUDA is available, use the first GPU
use_cuda = torch.cuda.is_available()
#use_cuda = False
if use_cuda:
    print("##### GPU CUDA mode #####")
    device_name = "cuda:0"
    device = torch.device(device_name)
    torch.cuda.set_device(0)    
else:
    print("##### CPU mode #####")
    device_name = "cpu"
    device = torch.device(device_name)

# load projectiond data
g = np.load(proj_fn)
print("projection loaded: ", g.shape)
g = g.reshape((1, g.shape[0], 1, g.shape[1]))
g_th = torch.from_numpy(g).to(device)

# perform FBP
fbp = FBP_Parallel(param_fn, use_cuda, device, use_fov, 1)
f_th = fbp(g_th)
f = f_th[0,0,:,:].cpu().detach().numpy()

# save output image
np.save(out_fn, f)
f_img = (f/np.max(f)*255).astype(np.uint8)
imageio.imwrite(out_fn[:-4]+".png", f_img)




'''
fbp = FBP_Parallel(param_fn, use_cuda, device, use_fov, 1)

fn_list = ["/p/vast1/mlct/CT_COE_Imatron/parallel512/S_007/S_007_0200_sino.npy", \
           "/p/vast1/mlct/CT_COE_Imatron/parallel512/S_010/S_010_0200_sino.npy", \
           "/p/vast1/mlct/CT_COE_Imatron/parallel512/S_020/S_020_0200_sino.npy"]

for ind, fn in enumerate(fn_list):
    g = np.load(fn)
    g = g.reshape((1, g.shape[0], 1, g.shape[1]))
    g_th = torch.from_numpy(g).to(device)
    f_th = fbp(g_th)
    f = f_th[0,0,:,:].cpu().detach().numpy()
    f_img = (f/np.max(f)*255).astype(np.uint8)
    imageio.imwrite("a_%d.png" % ind, f_img)
'''

