################################################################################
# Copyright 2022-2022 Lawrence Livermore National Security, LLC and other 
# LEAP project developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# LivermorE AI Projector for tomographic reconstruction (LEAP)
# demo: test example of neural network-based optimization (gradient decent) using projector class
################################################################################

import os
import sys
from sys import platform as _platform
sys.stdout.flush()
import argparse
import numpy as np
import imageio

import torch
import torch.nn as nn
from torch.optim import Adam, SGD, Adagrad, lr_scheduler

from leaptorch import Projector


# program arguments
parser = argparse.ArgumentParser()
parser.add_argument("--proj-fn", default="FORBILD_head_64_sino.npy", help="path to input projection data file")
parser.add_argument("--param-fn", default="FORBILD_head_64_param.cfg", help="path to projection geometry configuration file")
parser.add_argument("--output-dir", default="./sample_data", help="directory storing intermediate/output files")
parser.add_argument("--network-mode", type=int, default=0, help="0: no network used, 1: fully connected only, 2: convolutional with fully connected")
parser.add_argument("--use-fov", action='store_true', default=True, help="whether fov is used or not")
args = parser.parse_args()



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


# Neural Network (weight and bias)
class NN(nn.Module):
    def __init__(self, dimz, dimy, dimx, use_conv=True, device_name="cpu", verbose=0):
        super(NN, self).__init__()
        self.grid_dim = [dimz, dimy, dimx]
        self.use_conv = use_conv
        self.device_name = device_name
        self.verbose = verbose
        if self.use_conv:
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(64)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(1)
            )
            self.fc = nn.Sequential(
                nn.Linear(dimz*dimy*dimx, dimz*dimy*dimx),
                nn.LeakyReLU(0.01)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(dimz*dimy*dimx, dimz*dimy*dimx),
                #nn.Sigmoid(),
                #nn.ReLU(),
                #nn.LeakyReLU(0.001)
                nn.LeakyReLU(0.01)
            )
        #torch.nn.init.xavier_uniform(self.fc.weight)

    def forward(self, x):
        if self.verbose != 0:
            print(x.shape)
        if self.use_conv:
            x = x.unsqueeze(0).unsqueeze(0).squeeze(-1)
            x = self.conv1(x)
            x = self.conv2(x)
            #print(x.shape)
            flatten = torch.reshape(x, [-1, self.grid_dim[0]*self.grid_dim[1]*self.grid_dim[2]])
            out = self.fc(flatten)
        else:
            flatten = torch.reshape(x.unsqueeze(-1), [-1, self.grid_dim[0]*self.grid_dim[1]*self.grid_dim[2]])
            if self.verbose != 0:
                print(flatten.shape)
            out = self.fc(flatten)
        if self.verbose != 0:
            print(out.shape)
        y = torch.reshape(out, [self.grid_dim[0], self.grid_dim[1], self.grid_dim[2]])
        if self.verbose != 0:
            print(y.shape)
        return y


# CT reconstruction solver
class Reconstructor:
    def __init__(self, nn_model, projector, theWeights, device_name, learning_rate=0.01, use_decay=False,
                 iter_count=1000, stop_criterion=1e-1, save_dir='.', save_freq=100, verbose=1):

        # set nn_model and projector
        self.nn = nn_model
        self.projector = projector
        self.weights = theWeights
        self.device_name = device_name

        # set up hyperparameters
        self.learning_rate = learning_rate
        self.use_decay = use_decay
        self.iter_count = iter_count
        self.stop_criterion = stop_criterion
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.verbose = verbose

    def loss_func(self, input, target):
        if self.weights is None:
            return ((input - target) ** 2).mean()
        else:
            return (self.weights * (input - target) ** 2).mean()
        
    def reconstruct(self, g, f_init, fn_prefix="output"):
        # if no neural network is used, make f trainable
        if self.nn == None:
            f_estimated = f_init.clone()
            f_estimated.requires_grad = True

        # set up loss, optimizer
        #loss_func = nn.MSELoss().float() # MSE for loss/cost function
        #loss_func = self.weighted_mse_loss().float()
        #if self.nn == None:
            #optimizer = Adam([f_estimated], lr=float(self.learning_rate))
        #optimizer = Adam(self.nn.parameters(), lr=float(self.learning_rate))
        if self.nn == None:
            optimizer = Adagrad([f_estimated], lr=float(self.learning_rate))
        else:
            optimizer = Adagrad(self.nn.parameters(), lr=float(self.learning_rate))
        if self.use_decay:
            scheduler = lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

        # set to train mode
        if self.nn != None:
            self.nn.train()
        self.projector.train()
        
        # main optimization iteration
        for i in range(self.iter_count):
            optimizer.zero_grad()

            # forward pass
            if self.nn != None:
                f_estimated = self.nn(f_init)
            g_estimated = self.projector(f_estimated)
        
            # compute loss
            loss = self.loss_func(g_estimated.cpu().float(), g.cpu().float())
            
            # L1 regularization?
            #l1_lambda = 0.001
            #l1_norm = sum(torch.linalg.norm(p, 1) for p in self.nn.parameters())
            #loss = loss1 + l1_lambda * l1_norm
        
            # back-propagate and update
            loss.backward()
            optimizer.step()
            if self.use_decay:
                scheduler.step()
        
            if i == 0:
                self.firstLoss = loss.cpu().data.item()

            # status display and save images
            loss_val = loss.cpu().data.item()
            if self.verbose > 0:
                print("[%d/%d] %s training loss %.9f  learning rate: %f" % (i, self.iter_count, self.device_name, loss_val/self.firstLoss, optimizer.param_groups[0]['lr']))
            if loss_val/self.firstLoss < self.stop_criterion:
                break
            if i == 0 or i % self.save_freq == 0:
                f_img = f_estimated.cpu().detach().numpy()[0,0,:,:]
                if np.max(f_img) == 0:
                    scaleVal = 1
                else:
                    scaleVal = 255.0/np.max(f_img)
                f_img[f_img<0.0] = 0.0
                imageio.imsave(os.path.join(self.save_dir, "%s_LEAP_%s_%07d.png" % (fn_prefix, self.device_name.replace(':','_'), i)), np.uint8(scaleVal*f_img))

        # eval mode to get final f
        if self.nn != None:
            self.nn.eval()
            f_estimated = self.nn(f_init)
        f_img = f_estimated.cpu().detach().numpy()[0,0,:,:]
        f_img[f_img<0.0] = 0.0
        if np.max(f_img) == 0:
            scaleVal = 1
        else:
            scaleVal = 255.0/np.max(f_img)
        imageio.imsave(os.path.join(self.save_dir, "%s_LEAP_%s_final.png" % (fn_prefix, self.device_name.replace(':','_'))), np.uint8(scaleVal*f_img))
        return f_estimated


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


################################################################################
# 1. Read or simulate F and G using LEAP
################################################################################

# read arguments
output_dir = args.output_dir
proj_fn = args.proj_fn
param_fn = args.param_fn
network_mode = args.network_mode
use_fov = args.use_fov


# initialize projector and load CT geometry and CT volume parameters
proj = Projector(forward_project=True, use_static=True, use_gpu=use_cuda, gpu_device=device)
proj.load_param(param_fn)
proj.allocate_batch_data()
proj.print_param()

# Get volume and projection data sizes
dimz, dimy, dimx = proj.lct.get_volume_dim()
views, rows, cols = proj.lct.get_projection_dim()


# initialize model (NN + projector)
if network_mode == 0:
    model = None
elif network_mode == 1:
    model = NN(dimz, dimy, dimx, use_conv=False, device_name=device_name, verbose=0)
    model.to(device)
elif network_mode == 2:
    model = NN(dimz, dimy, dimx, use_conv=True, device_name=device_name, verbose=0)
    model.to(device)


# load g
g = np.load(proj_fn)
if use_cuda:
    g = torch.from_numpy(g).to(device)
else:
    g = torch.from_numpy(g)
g = g.unsqueeze(0)
g = g.unsqueeze(2)
print("projection loaded: ", g.shape)


# initialize f to be solved, given g above
f_init = np.zeros((1, dimz, dimy, dimx),dtype=np.float32)
if use_cuda:
    f_init = torch.from_numpy(f_init).to(device)
else:
    f_init = torch.from_numpy(f_init)

    
# remove field of view mask if necessary
if args.use_fov == False:
    x_max = np.max(np.abs(proj.lct.x_samples()))
    y_max = np.max(np.abs(proj.lct.y_samples()))
    proj.lct.set_diameterFOV(2.0*np.sqrt(x_max**2+y_max**2))


# initialize and run reconstructor (solver)
if _platform == "win32":
    output_dir = output_dir.replace("/","\\")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
solver = Reconstructor(model, proj, None, device_name, learning_rate=0.01, use_decay=False, stop_criterion=1e-7, save_dir=output_dir)
f_final = solver.reconstruct(g, f_init, "f")


# Display Result
import matplotlib.pyplot as plt
plt.imshow(np.squeeze(f_final.cpu().detach().numpy()[0,0,:,:]))
plt.show()
