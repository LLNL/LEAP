################################################################################
# Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
# LEAP project developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# LivermorE AI Projector for Computed Tomography (LEAP)
# PyTorch projector class
################################################################################

import numpy as np
import torch
#import leapct
from leapctype import *
lct = tomographicModels()

# Note:
# Image tensor format: [Batch, ImageZ, ImageY, ImageX]
# Projection tensor format: [Batch, Views, Detector_Row, Detector_Col]

# CPU Projector for forward and backward propagation
class ProjectorFunctionCPU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, proj, vol, param_id): # input: image, output: projection (sinogram)
        for batch in range(input.shape[0]):
            f = input[batch]
            g = proj[batch]
            lct.project_cpu(g, f, param_id.item()) # compute proj (g) from input (f)
        ctx.save_for_backward(input, vol, param_id)
        return proj

    @staticmethod
    def backward(ctx, grad_output): # grad_output: projection (sinogram) grad_input: image
        input, vol, param_id = ctx.saved_tensors
        for batch in range(input.shape[0]):
            f = vol[batch]
            g = grad_output[batch]
            lct.backproject_cpu(g, f, param_id.item()) # compute input (f) from proj (g)
        return vol, None, None, None

# GPU Projector for forward and backward propagation
class ProjectorFunctionGPU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, proj, vol, param_id): # input: image, output: projection (sinogram)
        for batch in range(input.shape[0]):
            f = input[batch]
            g = proj[batch]
            lct.project_gpu(g, f, param_id.item()) # compute proj (g) from input (f)
        ctx.save_for_backward(input, vol, param_id)
        return proj

    @staticmethod
    def backward(ctx, grad_output): # grad_output: projection (sinogram) grad_input: image
        input, vol, param_id = ctx.saved_tensors
        for batch in range(input.shape[0]):
            f = vol[batch]
            g = grad_output[batch]
            lct.backproject_gpu(g, f, param_id.item()) # compute input (f) from proj (g)
        return vol, None, None, None


# CPU BackProjector for forward and backward propagation
class BackProjectorFunctionCPU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, proj, vol, param_id): # input: projection (sinogram), output: image
        for batch in range(input.shape[0]):
            f = vol[batch]
            g = input[batch]
            lct.backproject_cpu(g, f, param_id.item()) # compute input (f) from proj (g)
            #vol[batch] = f
        ctx.save_for_backward(input, proj, param_id)
        return vol

    @staticmethod
    def backward(ctx, grad_output): # grad_output: image, grad_input: projection (sinogram)
        input, proj, param_id = ctx.saved_tensors
        for batch in range(input.shape[0]):
            f = grad_output[batch]
            g = proj[batch]
            lct.project_cpu(g, f, param_id.item()) # compute proj (g) from input (f)
            #proj[batch] = g
        return None, proj, None, None

# GPU BackProjector for forward and backward propagation
class BackProjectorFunctionGPU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, proj, vol, param_id): # input: projection (sinogram), output: image
        for batch in range(input.shape[0]):
            f = vol[batch]
            g = input[batch]
            lct.backproject_gpu(g, f, param_id.item()) # compute input (f) from proj (g)
            #vol[batch] = f
        ctx.save_for_backward(input, proj, param_id)
        return vol
        
    @staticmethod
    def backward(ctx, grad_output): # grad_output: image, grad_input: projection (sinogram)
        input, proj, param_id = ctx.saved_tensors
        for batch in range(input.shape[0]):
            f = grad_output[batch]
            g = proj[batch]
            lct.project_gpu(g, f, param_id.item()) # compute proj (g) from input (f)
            #proj[batch] = g
        return None, proj, None, None


# Pytorch Projector class
class Projector(torch.nn.Module):
    def __init__(self, forward_project=True, use_static=False, use_gpu=False, gpu_device=None, batch_size=1):
        super(Projector, self).__init__()
        self.forward_project = forward_project

        if use_static:
            self.param_id = 0
            self.lct = tomographicModels(self.param_id)
        else:
            self.lct = tomographicModels()
            self.param_id = self.lct.whichModel
        lct.whichModel = self.param_id
        self.param_id_t = torch.tensor(self.param_id).to(gpu_device) if use_gpu else torch.tensor(self.param_id)
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device
        if self.use_gpu:
            self.set_gpu(self.gpu_device)
        self.batch_size = batch_size
        self.vol_data = None
        self.proj_data = None
        
    def set_volume(self, numX, numY, numZ, voxelWidth, voxelHeight, offsetX=0.0, offsetY=0.0, offsetZ=0.0):
        self.lct.set_volume(numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ)
        vol_np = np.ascontiguousarray(np.zeros((self.batch_size, numZ, numY, numX),dtype=np.float32), dtype=np.float32)
        self.vol_data = torch.from_numpy(vol_np)
        if self.use_gpu:
            self.vol_data = self.vol_data.float().to(self.gpu_device)
            
    def set_default_volume(self, scale=1.0):
        self.lct.set_defaultVolume(scale)
        dim1, dim2, dim3 = self.get_volume_dim()
        vol_np = np.ascontiguousarray(np.zeros((self.batch_size, dim1, dim2, dim3),dtype=np.float32), dtype=np.float32)
        self.vol_data = torch.from_numpy(vol_np)
        if self.use_gpu:
            self.vol_data = self.vol_data.float().to(self.gpu_device)

    def set_parallelbeam(self, numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis):
        if type(phis) is torch.Tensor:
            phis = phis.numpy()
        self.lct.set_parallelbeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis)
        proj_np = np.ascontiguousarray(np.zeros((self.batch_size, numAngles, numRows, numCols),dtype=np.float32), dtype=np.float32)
        self.proj_data = torch.from_numpy(proj_np)
        if self.use_gpu:
            self.proj_data = self.proj_data.float().to(self.gpu_device)
            
    def set_fanbeam(self, numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau=0.0):
        if type(phis) is torch.Tensor:
            phis = phis.numpy()
        self.lct.set_fanbeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau)
        proj_np = np.ascontiguousarray(np.zeros((self.batch_size, numAngles, numRows, numCols),dtype=np.float32), dtype=np.float32)
        self.proj_data = torch.from_numpy(proj_np)
        if self.use_gpu:
            self.proj_data = self.proj_data.float().to(self.gpu_device)

    def set_conebeam(self, numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau=0.0, helicalPitch=0.0):
        if type(phis) is torch.Tensor:
            phis = phis.numpy()
        self.lct.set_conebeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau, helicalPitch)
        proj_np = np.ascontiguousarray(np.zeros((self.batch_size, numAngles, numRows, numCols),dtype=np.float32), dtype=np.float32)
        self.proj_data = torch.from_numpy(proj_np)
        if self.use_gpu:
            self.proj_data = self.proj_data.float().to(self.gpu_device)
    
    def set_modularbeam(self, numAngles, numRows, numCols, pixelHeight, pixelWidth, sourcePositions, detectorCenters, rowVec, colVec):
        if type(sourcePositions) is torch.Tensor:
            sourcePositions = sourcePositions.numpy()
        if type(detectorCenters) is torch.Tensor:
            detectorCenters = detectorCenters.numpy()
        if type(rowVec) is torch.Tensor:
            rowVec = rowVec.numpy()
        if type(colVec) is torch.Tensor:
            colVec = colVec.numpy()
        self.lct.set_modularbeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, sourcePositions, detectorCenters, rowVec, colVec)
        proj_np = np.ascontiguousarray(np.zeros((self.batch_size, numAngles, numRows, numCols),dtype=np.float32), dtype=np.float32)
        self.proj_data = torch.from_numpy(proj_np)
        if self.use_gpu:
            self.proj_data = self.proj_data.float().to(self.gpu_device)
        
    def get_volume_dim(self):
        return self.lct.get_volume_dim()
        
    def get_projection_dim(self):
        return self.lct.get_projection_dim()
        
    def allocate_batch_data(self):
        vol_dim1, vol_dim2, vol_dim3 = self.get_volume_dim()
        proj_dim1, proj_dim2, proj_dim3 = self.get_projection_dim()
        
        if vol_dim1 > 0 and vol_dim2 > 0 and vol_dim3 > 0 and proj_dim1 > 0 and proj_dim2 > 0 and proj_dim3 > 0:
            self.vol_data = torch.from_numpy(np.ascontiguousarray(np.zeros((self.batch_size, vol_dim1, vol_dim2, vol_dim3),dtype=np.float32), dtype=np.float32))
            self.proj_data = torch.from_numpy(np.ascontiguousarray(np.zeros((self.batch_size, proj_dim1, proj_dim2, proj_dim3),dtype=np.float32), dtype=np.float32))
            if self.use_gpu:
                self.vol_data = self.vol_data.float().to(self.gpu_device)
                self.proj_data = self.proj_data.float().to(self.gpu_device)
                
    def parse_param_dic(self, param_fn):
        pdic = {}
        with open(param_fn, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if len(line.strip()) == 0:
                    continue
                key = line.split('=')[0].strip()
                value = line.split('=')[1].strip()
                if key == 'proj_phis' or key == 'proj_geometry':
                    pdic[key] = value
                else:
                    pdic[key] = float(value)
        return pdic

    def load_param(self, param_fn, param_type=0): # param_type 0: cfg, 1: dict
        pdic = {}
        if param_type == 0:
            pdic = self.parse_param_dic(param_fn)
        elif param_type == 1:
            pdic = param_fn

        phis_str = pdic['proj_phis']
        if len(phis_str) > 0:
            #phis = torch.from_numpy(np.array([float(x.strip()) for x in phis_str.split(',')]).astype(np.float32))
            phis = np.array([float(x.strip()) for x in phis_str.split(',')]).astype(np.float32)
        else:
            phis = np.array(range(int(pdic['proj_nangles']))).astype(np.float32)
            phis = phis*pdic['proj_arange']/float(pdic['proj_nangles'])
            #phis = torch.from_numpy(phis)
        
        self.set_volume(int(pdic['img_dimx']), int(pdic['img_dimy']), int(pdic['img_dimz']),
                        pdic['img_pwidth'], pdic['img_pheight'], 
                        pdic['img_offsetx'], pdic['img_offsety'], pdic['img_offsetz'])
        if pdic['proj_geometry'] == 'parallel':
            self.set_parallelbeam(int(pdic['proj_nangles']), int(pdic['proj_nrows']), int(pdic['proj_ncols']), 
                                   pdic['proj_pheight'], pdic['proj_pwidth'], 
                                   pdic['proj_crow'], pdic['proj_ccol'], phis)
        elif pdic['proj_geometry'] == 'fan':
            self.set_fanbeam(int(pdic['proj_nangles']), int(pdic['proj_nrows']), int(pdic['proj_ncols']), 
                               pdic['proj_pheight'], pdic['proj_pwidth'], 
                               pdic['proj_crow'], pdic['proj_ccol'], 
                               phis, pdic['proj_sod'], pdic['proj_sdd'])
        elif pdic['proj_geometry'] == 'cone':
            self.set_conebeam(int(pdic['proj_nangles']), int(pdic['proj_nrows']), int(pdic['proj_ncols']), 
                               pdic['proj_pheight'], pdic['proj_pwidth'], 
                               pdic['proj_crow'], pdic['proj_ccol'], 
                               phis, pdic['proj_sod'], pdic['proj_sdd'])
        #elif pdic['proj_geometry'] == 'modular':
        #    self.set_modularbeam(int(pdic['proj_nangles']), int(pdic['proj_nrows']), int(pdic['proj_ncols']), 
        #                          pdic['proj_pheight'], pdic['proj_pwidth'], 
        #                          srcpos, modcenter, rowvec, colvec)

    def save_param(self, param_fn):
        return self.lct.save_param(param_fn)

    def set_gpu(self, which):
        self.gpu_device = which
        return self.lct.set_gpu(self.gpu_device.index)
        
    def set_gpus(self, listofgpus):
        self.gpu_device = listofgpus[0]
        return self.lct.set_gpus(listofgpus)
        
    def print_parameters(self):
        self.lct.print_parameters()
    
    def fbp(self, input): # input is projection data (g batch)
        for batch in range(input.shape[0]):
            if self.use_gpu:
                self.lct.FBP_gpu(input[batch], self.vol_data[batch])
            else:
                self.lct.FBP_cpu(input[batch], self.vol_data[batch])
        return self.vol_data

    def forward(self, input):
        if self.forward_project:
            if self.use_gpu:
                return ProjectorFunctionGPU.apply(input, self.proj_data, self.vol_data, self.param_id_t)
            else:
                return ProjectorFunctionCPU.apply(input, self.proj_data, self.vol_data, self.param_id_t)
        else:
            if self.use_gpu:
                return BackProjectorFunctionGPU.apply(input, self.proj_data, self.vol_data, self.param_id_t)
            else:
                return BackProjectorFunctionCPU.apply(input, self.proj_data, self.vol_data, self.param_id_t)

    

    ###################################################################################################################
    ###################################################################################################################
    # THIS SECTION OF FUNCTIONS ARE ALIASES OF FUNCTIONS ABOVE INCLUDED FOR BACKWARD COMPATIBILITY
    ###################################################################################################################
    ###################################################################################################################
    def print_param(self):
        self.print_parameters()

    def set_GPU(self,which):
        return self.set_gpu(which)
        
    def set_GPUs(self,listofgpus):
        return self.set_gpus(listofgpus)
    