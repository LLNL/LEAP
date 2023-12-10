################################################################################
# Copyright 2022-2022 Lawrence Livermore National Security, LLC and other 
# LEAP project developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# LivermorE AI Projector for tomographic reconstruction (LEAP)
# demo: test example with projector class
# this example shows how to use forward projection and filtered back projection in LEAP
################################################################################

# param_fn=/p/vast1/mlct/CT_COE_Imatron/param_parallel512.cfg
# data_dir=/usr/workspace/kim63/src/ctnetplus_techmat/results/20230222_la512/test/
# python test_project.py --param-fn ${param_fn}  --img-fn ${data_dir}/S_193_0100_pred.npy  --out-fn ${data_dir}/S_193_0100_sino2.npy

import sys
sys.stdout.flush()
import argparse
import numpy as np
import imageio

import torch
import torch.nn as nn
import leapct
from leaptorch import Projector


# program arguments
parser = argparse.ArgumentParser()
parser.add_argument("--img-fn", default="sample_data/FORBILD_head_64.npy", help="path to input image data file")
parser.add_argument("--param-fn", default="sample_data/param_parallel64.cfg", help="path to projection geometry configuration file")
parser.add_argument("--out-fn", default="sample_data/FORBILD_head_64_sino.npy", help="path to output projection data file")
parser.add_argument("--use-API", action="store_true", help="use API or PyTorch Projector class (default)")
args = parser.parse_args()



def simulate_img(M, N):
    midP = int(N/2)
    quarterP = int(midP/2)
    f = np.zeros((M, N, N), dtype=np.float32, order='C')
    for n in range(0,M):
        #f[quarterP:midP+quarterP,quarterP:midP+quarterP,n] = 1.0
        f[n, quarterP:midP+quarterP,quarterP:midP+quarterP] = 0.02
        f[n, midP-int(quarterP/2):midP+int(quarterP/2),midP-int(quarterP/2):midP+int(quarterP/2)] = 0.04
    return f.reshape((1, f.shape[0], f.shape[1], f.shape[2]))

def load_param(param_fn):
    pdic = {}
    with open(param_fn, 'r') as f:
        lines = f.readlines()
        for line in lines:
            key = line.split('=')[0].strip()
            value = line.split('=')[1].strip()
            if key == 'proj_phis' or key == 'proj_geometry':
                pdic[key] = value
            else:
                pdic[key] = float(value)
    phis_str = pdic['proj_phis']
    if len(phis_str) > 0:
        phis = torch.from_numpy(np.array([float(x.strip()) for x in phis_str.split(',')]))
    else:
        phis = np.array(range(int(pdic['proj_nangles']))).astype(np.float32)
        phis = phis*pdic['proj_arange']/float(pdic['proj_nangles'])
    return pdic, phis



# read arguments
img_fn = args.img_fn
param_fn = args.param_fn
out_fn = args.out_fn
use_API = args.use_API


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


# initialize projector and load parameters
proj = Projector(use_gpu=use_cuda, gpu_device=device, batch_size=1)
proj.load_param(param_fn)
proj.set_projector(1)
proj.print_param()
dimz, dimy, dimx = proj.get_volume_dim()
views, rows, cols = proj.get_projection_dim()
params, _ = load_param(param_fn)

# load image (f)
if len(img_fn) > 0:
    f = np.load(img_fn)
    f = f.reshape((1, 1, f.shape[0], f.shape[1])) # batch, z, y, x
else:
    f = simulate_img(dimz, dimx)


if not use_API: # use PyTorch Projector class
    print("### use leapct Projector class ###")

    # set image tensor
    f_th = torch.from_numpy(f).to(device)
    imageio.imsave(img_fn[:-4] + ".png", f[0,0,:,:]/np.max(f)*255)

    # forward project
    g_th = proj(f_th)
    g = g_th.cpu().detach().numpy()[0,:,0,:]  # batch, angles, detector row, detector col -> which gives "sinogram"

    # save projection data
    np.save(out_fn, g)
    imageio.imsave(out_fn[:-4] + ".png", g/np.max(g)*255)

    # run filtered back projection
    f_recon = proj.fbp(g_th)
    f_recon_slice = f_recon.cpu().detach().numpy()[0,0,:,:]

    # save FBP reconstructed image
    np.save(out_fn[:-4] + "_FBP.npy", f_recon_slice)
    imageio.imsave(out_fn[:-4] + "_FBP.png", f_recon_slice/np.max(f_recon_slice)*255)

else: # use LEAP API functions
    print("### use leapct API ###")
    M = int(params["img_dimz"])
    N = int(params["img_dimx"])
    pixelSize = params["img_pwidth"]
    if pixelSize == 0:
        pixelSize = 0.2*2048.0/float(N)

    N_phis = int(params["proj_nangles"])
    sod = params["proj_sod"]
    sdd = params["proj_sdd"]
    print(M,N,N_phis, "aaa")

    f = torch.from_numpy(f[0,:,:,:])

    # set up CT parameters
    whichProjector = 1
    midP = int(N/2)
    quarterP = int(midP/2)
    arange = 180
    phis = np.array(range(N_phis)).astype(np.float32)
    phis = phis*arange/float(N_phis)
    phis = torch.from_numpy(phis)

    # set up CT projector
    if params["proj_geometry"] == "parallel":
        leapct.set_parallel_beam(-1, N_phis, M, N, pixelSize, pixelSize, 0.5*float(M-1), 0.5*float(N-1), phis)
        leapct.set_volume(-1, N, N, M, pixelSize, pixelSize, 0.0, 0.0, 0.0)
    elif params["proj_geometry"] == "cone":
        leapct.set_cone_beam(-1, N_phis, M, N, pixelSize, pixelSize, 0.5*float(M-1), 0.5*float(N-1), phis, sod, sdd)
        leapct.set_volume(-1, N, N, M, pixelSize*sod/sdd, pixelSize*sod/sdd, 0.0, 0.0, 0.0)

    leapct.print_param(-1)
    leapct.set_projector(-1, whichProjector)
    leapct.set_gpu(-1, 0)

    # create projection data (g)
    g = np.zeros((N_phis,M,N), dtype=np.float32, order='C')
    g = torch.from_numpy(g)

    # simulate projection data
    if use_cuda:
        f = f.float().to(device)
        g = g.float().to(device)
        leapct.project_gpu(-1, g, f)
    else:
        leapct.project_cpu(-1, g, f)

    # save projection data to npy file and image file
    g_np = g.cpu().detach().numpy()
    np.save(out_fn, g_np)
    imageio.imsave(out_fn[:-4] + ".png", g_np[:,0,:]/np.max(g_np))

    # run filtered back projection
    if use_cuda:
        leapct.fbp_gpu(-1, g, f)
    else:
        leapct.fbp_cpu(-1, g, f)
    
    # save FBP reconstructed image
    f_np = f.cpu().detach().numpy()
    f_np = f_np[0,:,:]
    np.save(out_fn[:-4] + "_FBP.npy", f_np)
    imageio.imsave(out_fn[:-4] + "_FBP.png", f_np[:,0,:]/np.max(f_np))
