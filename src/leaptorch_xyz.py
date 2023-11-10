################################################################################
# Copyright 2022-2022 Lawrence Livermore National Security, LLC and other 
# LEAP project developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# LivermorE AI Projector for Computed Tomography (LEAP)
# PyTorch projector class
################################################################################

import numpy as np
import torch
import leapct

# Note:
# Image tensor format: [Batch, ImageZ, ImageY, ImageX]
# Projection tensor format: [Batch, Views, Detector_Row, Detector_Col]

# CPU Projector for forward and backward propagation
class ProjectorFunctionCPU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, proj, vol, param_id): # input: image, output: projection (sinogram)
        for batch in range(input.shape[0]):
            #print(input.shape, proj.shape)
            f = input[batch]
            f_xyz = torch.Tensor.contiguous(torch.permute(f, (2, 1, 0)))
            g = proj[batch]
            leapct.project_cpu(param_id, g, f_xyz) # compute proj (g) from input (f)
            proj[batch] = g
        ctx.save_for_backward(input, proj, vol, param_id)
        return proj

    @staticmethod
    def backward(ctx, grad_output): # grad_output: projection (sinogram) grad_input: image
        input, proj, vol, param_id = ctx.saved_tensors
        for batch in range(input.shape[0]):
            f = vol[batch]
            f_xyz = torch.Tensor.contiguous(torch.permute(f, (2, 1, 0)))
            g = grad_output[batch]
            leapct.backproject_cpu(param_id, g, f_xyz) # compute input (f) from proj (g)
            f_zyx = torch.Tensor.contiguous(torch.permute(f_xyz, (2, 1, 0)))
            vol[batch] = f_zyx
        return vol, None, None, None

# GPU Projector for forward and backward propagation
class ProjectorFunctionGPU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, proj, vol, param_id): # input: image, output: projection (sinogram)
        for batch in range(input.shape[0]):
            f = input[batch]
            f_xyz = torch.Tensor.contiguous(torch.permute(f, (2, 1, 0)))
            g = proj[batch]
            leapct.project_gpu(param_id, g, f_xyz) # compute proj (g) from input (f)
            proj[batch] = g
        ctx.save_for_backward(input, proj, vol, param_id)
        return proj

    @staticmethod
    def backward(ctx, grad_output): # grad_output: projection (sinogram) grad_input: image
        input, proj, vol, param_id = ctx.saved_tensors
        for batch in range(input.shape[0]):
            f = vol[batch]
            f_xyz = torch.Tensor.contiguous(torch.permute(f, (2, 1, 0)))
            g = grad_output[batch]
            leapct.backproject_gpu(param_id, g, f_xyz) # compute input (f) from proj (g)
            f_zyx = torch.Tensor.contiguous(torch.permute(f_xyz, (2, 1, 0)))
            vol[batch] = f_zyx
        return vol, None, None, None


# CPU BackProjector for forward and backward propagation
class BackProjectorFunctionCPU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, proj, vol, param_id): # input: projection (sinogram), output: image
        for batch in range(input.shape[0]):
            f = vol[batch]
            f_xyz = torch.Tensor.contiguous(torch.permute(f, (2, 1, 0)))
            g = input[batch]
            leapct.backproject_cpu(param_id, g, f_xyz) # compute input (f) from proj (g)
            f_zyx = torch.Tensor.contiguous(torch.permute(f_xyz, (2, 1, 0)))
            vol[batch] = f_zyx
        ctx.save_for_backward(input, proj, vol, param_id)
        return vol

    @staticmethod
    def backward(ctx, grad_output): # grad_output: image, grad_input: projection (sinogram)
        input, proj, vol, param_id = ctx.saved_tensors
        for batch in range(input.shape[0]):
            #print(input.shape, proj.shape)
            f = grad_output[batch]
            f_xyz = torch.Tensor.contiguous(torch.permute(f, (2, 1, 0)))
            g = proj[batch]
            leapct.project_cpu(param_id, g, f_xyz) # compute proj (g) from input (f)
            proj[batch] = g
        return proj, None, None, None

# GPU BackProjector for forward and backward propagation
class BackProjectorFunctionGPU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, proj, vol, param_id): # input: projection (sinogram), output: image
        for batch in range(input.shape[0]):
            f = vol[batch]
            f_xyz = torch.Tensor.contiguous(torch.permute(f, (2, 1, 0)))
            g = input[batch]
            leapct.backproject_gpu(param_id, g, f_xyz) # compute input (f) from proj (g)
            f_zyx = torch.Tensor.contiguous(torch.permute(f_xyz, (2, 1, 0)))
            vol[batch] = f_zyx
        ctx.save_for_backward(input, proj, vol, param_id)
        return vol

    @staticmethod
    def backward(ctx, grad_output): # grad_output: image, grad_input: projection (sinogram)
        input, proj, vol, param_id = ctx.saved_tensors
        for batch in range(input.shape[0]):
            f = grad_output[batch]
            f_xyz = torch.Tensor.contiguous(torch.permute(f, (2, 1, 0)))
            g = proj[batch]
            leapct.project_gpu(param_id, g, f_xyz) # compute proj (g) from input (f)
            proj[batch] = g
        return proj, None, None, None


# Pytorch Projector class
class Projector(torch.nn.Module):
    def __init__(self, forward_project=True, use_static=False, use_gpu=False, gpu_device=None, batch_size=1):
        super(Projector, self).__init__()
        self.forward_project = forward_project
        self.param_id = -1 if use_static else leapct.create_param()
        self.param_id_t = torch.tensor(self.param_id).to(gpu_device) if use_gpu else torch.tensor(self.param_id)
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device
        if self.use_gpu:
            leapct.set_gpu(self.param_id, self.gpu_device.index)
        self.batch_size = batch_size
        self.vol_data = None
        self.proj_data = None
        self.vol_param = None
        self.proj_param = None
        
    def set_volume(self, dimx, dimy, dimz, width, height, offsetx, offsety, offsetz):
        leapct.set_volume(self.param_id, dimx, dimy, dimz, width, height, offsetx, offsety, offsetz)
        vol_np = np.zeros((self.batch_size, dimz, dimy, dimx)).astype(np.float32)
        self.vol_data = torch.from_numpy(vol_np)
        if self.use_gpu:
            self.vol_data = self.vol_data.float().to(self.gpu_device)
            
    def set_default_volume(self, scale=1.0):
        leapct.set_volume(self.param_id, scale)
        dim1, dim2, dim3 = self.get_volume_dim()
        vol_np = np.ascontiguousarray(np.zeros((self.batch_size, dim1, dim2, dim3)).astype(np.float32), dtype=np.float32)
        self.vol_data = torch.from_numpy(vol_np)
        if self.use_gpu:
            self.vol_data = self.vol_data.float().to(self.gpu_device)

    def set_parallel_beam(self, nangles, nrows, ncols, pheight, pwidth, crow, ccol, arange, phis):
        leapct.set_parallel_beam(self.param_id, nangles, nrows, ncols, pheight, pwidth, crow, ccol, arange, phis)
        proj_np = np.zeros((self.batch_size, nangles, nrows, ncols)).astype(np.float32)
        self.proj_data = torch.from_numpy(proj_np)
        if self.use_gpu:
            self.proj_data = self.proj_data.float().to(self.gpu_device)

    def set_fan_beam(self, nangles, nrows, ncols, pheight, pwidth, crow, ccol, phis, sod, sdd):
        leapct.set_fan_beam(self.param_id, nangles, nrows, ncols, pheight, pwidth, crow, ccol, phis, sod, sdd)
        proj_np = np.ascontiguousarray(np.zeros((self.batch_size, nangles, nrows, ncols)).astype(np.float32), dtype=np.float32)
        self.proj_data = torch.from_numpy(proj_np)
        if self.use_gpu:
            self.proj_data = self.proj_data.float().to(self.gpu_device)

    def set_cone_beam(self, nangles, nrows, ncols, pheight, pwidth, crow, ccol, arange, phis, sod, sdd):
        leapct.set_cone_beam(self.param_id, nangles, nrows, ncols, pheight, pwidth, crow, ccol, arange, phis, sod, sdd)
        proj_np = np.zeros((self.batch_size, nangles, nrows, ncols)).astype(np.float32)
        self.proj_data = torch.from_numpy(proj_np)
        if self.use_gpu:
            self.proj_data = self.proj_data.float().to(self.gpu_device)
    
    def set_modular_beam(self, nangles, nrows, ncols, pheight, pwidth, srcpos, modcenter, rowvec, colvec):
        leapct.set_modular_beam(self.param_id, nangles, nrows, ncols, pheight, pwidth, srcpos, modcenter, rowvec, colvec)
        proj_np = np.zeros((self.batch_size, nangles, nrows, ncols)).astype(np.float32)
        self.proj_data = torch.from_numpy(proj_np)
        if self.use_gpu:
            self.proj_data = self.proj_data.float().to(self.gpu_device)

    def get_volume_dim(self):
        dim_tensor = torch.from_numpy(np.array([0,0,0]).astype(np.int32))
        leapct.get_volume_dim(self.param_id, dim_tensor)
        dim = dim_tensor.cpu().detach().numpy()
        return dim[2], dim[1], dim[0]

    def get_projection_dim(self):
        dim_tensor = torch.from_numpy(np.array([0,0,0]).astype(np.int32))
        leapct.get_projection_dim(self.param_id, dim_tensor)
        dim = dim_tensor.cpu().detach().numpy()
        return dim[0], dim[1], dim[2]

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
            phis = torch.from_numpy(np.array([float(x.strip()) for x in phis_str.split(',')]).astype(np.float32))
        else:
            phis = np.array(range(int(pdic['proj_nangles']))).astype(np.float32)
            phis = phis*pdic['proj_arange']/float(pdic['proj_nangles'])
            phis = torch.from_numpy(phis)
        
        self.set_volume(int(pdic['img_dimx']), int(pdic['img_dimy']), int(pdic['img_dimz']),
                        pdic['img_pwidth'], pdic['img_pheight'], 
                        pdic['img_offsetx'], pdic['img_offsety'], pdic['img_offsetz'])
        if pdic['proj_geometry'] == 'parallel':
            self.set_parallel_beam(int(pdic['proj_nangles']), int(pdic['proj_nrows']), int(pdic['proj_ncols']), 
                                   pdic['proj_pheight'], pdic['proj_pwidth'], 
                                   pdic['proj_crow'], pdic['proj_ccol'], 
                                   pdic['proj_arange'], phis)
        elif pdic['proj_geometry'] == 'fan':
            self.set_fan_beam(int(pdic['proj_nangles']), int(pdic['proj_nrows']), int(pdic['proj_ncols']), 
                               pdic['proj_pheight'], pdic['proj_pwidth'], 
                               pdic['proj_crow'], pdic['proj_ccol'], 
                               pdic['proj_arange'], phis, 
                               pdic['proj_sod'], pdic['proj_sdd'])
        elif pdic['proj_geometry'] == 'cone':
            self.set_cone_beam(int(pdic['proj_nangles']), int(pdic['proj_nrows']), int(pdic['proj_ncols']), 
                               pdic['proj_pheight'], pdic['proj_pwidth'], 
                               pdic['proj_crow'], pdic['proj_ccol'], 
                               pdic['proj_arange'], phis, 
                               pdic['proj_sod'], pdic['proj_sdd'])
        #elif pdic['proj_geometry'] == 'modular':
        #    self.set_modular_beam(int(pdic['proj_nangles']), int(pdic['proj_nrows']), int(pdic['proj_ncols']), 
        #                          pdic['proj_pheight'], pdic['proj_pwidth'], 
        #                          srcpos, modcenter, rowvec, colvec)

    def save_param(self, param_fn):
        return leapct.save_param(self.param_id, param_fn)

    def set_symmetry_axis(self, val):
        return leapct.set_symmetry_axis(self.param_id, val)
        
    def set_projector(self, which):
        return leapct.set_projector(self.param_id, which)

    def set_gpu(self, which):
        self.gpu_device = which
        return leapct.set_gpu(self.param_id, self.gpu_device.index)
        
    def print_param(self):
        leapct.print_param(self.param_id)
        
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



