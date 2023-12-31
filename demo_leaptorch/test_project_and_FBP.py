################################################################################
# Copyright 2022-2022 Lawrence Livermore National Security, LLC and other 
# LEAP project developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# LivermorE AI Projector for tomographic reconstruction (LEAP)
# demo: test example with projector class
# this example shows how to use forward projection and filtered back projection in LEAP
################################################################################

import sys
from sys import platform as _platform
sys.stdout.flush()
import argparse
import numpy as np
import imageio
import torch

# program arguments
# All of these arguments are optional.  If no input file names are given, then the data will be generated
# by forward projecting the FORBILD head phantom using a parallel-beam geometry
parser = argparse.ArgumentParser()
parser.add_argument("--img-fn", default="", help="path to input image data file")
parser.add_argument("--param-fn", default="", help="path to projection geometry configuration file")
parser.add_argument("--out-fn", default="sample_data/example.npy", help="path to output projection data file")
parser.add_argument("--use_leapctype", action="store_true", help="use leapctype tomographicModels class or leaptorch Projector class (default)")
args = parser.parse_args()


# read arguments
if _platform == "win32":
    args.img_fn = args.img_fn.replace("/","\\")
    args.param_fn = args.param_fn.replace("/","\\")
    args.out_fn = args.out_fn.replace("/","\\")

img_fn = args.img_fn
param_fn = args.param_fn
out_fn = args.out_fn
use_leapctype = args.use_leapctype

if (len(img_fn) > 0 and len(param_fn) == 0) or (len(img_fn) == 0 and len(param_fn) > 0):
    print('Error: must specify both img-fn and param-fn or neither')
    quit()

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


if not use_leapctype: # use leaptorch Projector class
    print("### using leaptorch Projector class ###")
    
    import torch.nn as nn
    from leaptorch import Projector

    proj = Projector(forward_project=True, use_static=True, use_gpu=use_cuda, gpu_device=device, batch_size=1)

    # Set the CT geometry and CT volume parameters and set the voxelized phantom
    if len(img_fn) > 0 and len(param_fn) > 0:
        proj.load_param(param_fn)
        f = proj.leapct.load_volume(img_fn)
        f = f.reshape((1, f.shape[0], f.shape[1], f.shape[2])) # batch, z, y, x
        f = np.ascontiguousarray(f, dtype=np.float32)
        
        # Convert to torch tensor and copy to GPU if necessary
        f_th = torch.from_numpy(f).to(device)
    else:
        # Set the scanner geometry
        numCols = 256
        numAngles = 2*int(360*numCols/1024)
        pixelSize = 0.5*512/numCols
        numRows = 1
        proj.leapct.set_parallelbeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), proj.leapct.setAngleArray(numAngles, 180.0))
        proj.leapct.set_default_volume()
        proj.allocate_batch_data()
        f = proj.leapct.allocate_volume()
        proj.leapct.set_FORBILD(f,True,3)
        
        # Convert to torch tensor and copy to GPU if necessary
        f_th = torch.from_numpy(f).to(device)
        f_th = f_th.unsqueeze(0)
            
    # forward project
    g_th = proj(f_th)
    g = g_th.cpu().detach().numpy()[0,:,:,:] # just get first batch
    g_sino = np.squeeze(g[:,g.shape[1]//2,:]) # just get one sinogram

    # save projection data
    proj.leapct.save_projections(out_fn, g)
    imageio.imsave(out_fn[:-4] + ".png", np.uint8(g_sino/np.max(g_sino)*255))

    # run filtered back projection
    f_recon = proj.fbp(g_th)
    f_recon = f_recon.cpu().detach().numpy()[0,:,:,:]
    f_recon[f_recon<0.0] = 0.0
    f_recon_slice = f_recon[f_recon.shape[0]//2,:,:]

    # save FBP reconstructed image
    proj.leapct.save_volume(out_fn[:-4] + "_FBP.npy", f_recon)
    imageio.imsave(out_fn[:-4] + "_FBP.png", np.uint8(f_recon_slice/np.max(f_recon_slice)*255))
    
else: # use leapctype tomographicModels class
    print("### using leapctype tomographicModels class ###")
    
    from leapctype import *
    leapct = tomographicModels()
    
    # Set the CT geometry and CT volume parameters and set the voxelized phantom
    if len(img_fn) > 0 and len(param_fn) > 0:
        leapct.load_param(param_fn)
        f = leapct.load_volume(img_fn)
        f = f.reshape((1, f.shape[0], f.shape[1], f.shape[2])) # batch, z, y, x
        f = np.ascontiguousarray(f, dtype=np.float32)
    else:
        # Set the scanner geometry
        numCols = 256
        numAngles = 2*int(360*numCols/1024)
        pixelSize = 0.5*512/numCols
        numRows = 1
        leapct.set_parallelbeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 180.0))
        leapct.set_default_volume()
        f = leapct.allocate_volume()
        leapct.set_FORBILD(f,True,3)
        
    # Convert to torch tensor and copy to GPU if necessary
    f_th = torch.from_numpy(f).to(device)
    
    # Allocate projection data
    g = leapct.allocate_projections()
    g_th = torch.from_numpy(g).to(device)
    
    # forward project
    leapct.project(g_th, f_th)
    g = g_th.cpu().detach().numpy()
    g_sino = np.squeeze(g[:,g.shape[1]//2,:]) # just get one sinogram

    # save projection data
    leapct.save_projections(out_fn, g)
    imageio.imsave(out_fn[:-4] + ".png", np.uint8(g_sino/np.max(g_sino)*255))

    # run filtered back projection
    f_recon = leapct.fbp(g_th)
    f_recon = f_recon.cpu().detach().numpy()
    f_recon[f_recon<0.0] = 0.0
    f_recon_slice = f_recon[f_recon.shape[0]//2,:,:]

    # save FBP reconstructed image
    leapct.save_volume(out_fn[:-4] + "_FBP.npy", f_recon)
    imageio.imsave(out_fn[:-4] + "_FBP.png", np.uint8(f_recon_slice/np.max(f_recon_slice)*255))
