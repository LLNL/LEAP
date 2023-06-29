# LEAP: Differentiable forward and backward projectors for AI/ML-driven computed tomography applications

  
## Introduction
LivermorE AI Projector (**LEAP**) is a python package library to enable differentiable forward and back projection operations for machine learning-based computed tomography (CT) applications. This package enables the integration of the forward projection into the neural network training pipeline. LEAP currently supports three CT projection geometries: parallel beam, cone beam, and modular beam. This library is provided as both native python APIs and PyTorch classes. 


## Installation

LEAP's core forward and back projections are written in C++ and CUDA. To compile and install LEAP, you should have a proper GCC compiler (7.3 or higher). To perform projections on GPU, make sure to install a proper CUDA driver and toolkit (10.2 or higher) as well as a CUDA compiler (nvcc). LEAP also requires PyTorch (1.10 or higher). If you have a GPU and would like to use GPU-based projection, overwrite setup.py using setup_gpu.py. Otherwise, use setup_cpu.py
To install LEAP, use **pip install**:
$ cd leap
$ pip install .


## Getting Started

The main PyTorch module class is found in leaptorch. **Projector** is the main class defined in leaptorch. It defines a CT projection associated with a specific projection geometry and parameters. The following example instantiates a Projector class for an image dimension of 256^3. This code uses a parallel beam geometry with 180 projection views. 
```
import torch
from leaptorch import Projector
proj = Projector(use_gpu=False) # CPU mode
```
The constructor of the Projector class takes several arguments to specify 
the GPU device and other options. To use the GPU projector, 
```
device_name = "cuda:0"
device = torch.device(device_name)
proj = Projector(use_gpu=True, gpu_device=device)
```
The image dimension can be specified using "set_volume" function:
```
proj.set_volume(256, 256, 256, 0.8, 0.8, 0, 0, 0)
```
Three functions to specify projection parameters are provided for parallel, cone, and modular beam projections. To use the parallel beam projection, use "set_parallel_beam" function:
```
proj.set_parallel_beam(180, 256, 256, 0.8, 0.8, 127.5, 127.5, 180, phis)
```
"print_param" function summarizes all projection parameters specified in the current projector: proj.print_param()
The projector class can also read a configuration file to specify all projection parameters as well as image dimensions without calling "set_volume" and "set_parallel_beam" functions (if the projection uses a parallel beam):
```
proj.load_param('parameter.cfg')
```

### Integrating LEAP Projector into existing PyTorch Models

Since the projector class is derived from ```torch.nn.Module```, it can be integrated into any existing PyTorch-based neural network models. Below is an example of how the projector object can be used to compute the forward projection loss function. Assume that "network" is a neural network model defined as a ```torch.nn.Module```. 
```
from torch.optim import Adagrad

optimizer = Adagrad(self.nn.parameters(), lr=float(self.learning_rate)) # setup an optimizer
N_iter = 1000 # number of iterations
network.train() # a neural network model derived from torch.nn.Module
proj.train() # CT projector class instance
for i in range(N_iter):
	optimizer.zero_grad()
	# a neural network that takes an input image to predict a noiseless/artifactless image
	img_pred = network(img_init)
	# forward-project the predicted image to sinogram projection data
	sino_pred = proj(img_pred) 
	# user-specified loss function, e.g., torch.nn.MSE()
	loss = loss_func(sino_pred.float(), sino_gt.float()) 
	loss.backward()
	optimizer.step()
```

## Class References

#### leaptorch.Projector.set_volume(dimx, dimy, dimz, width, height, offsetx, offsety, offsetz)
specify image volume dimension to be reconstructed
* dimx, dimy, dimz: image dimension in x, y and z axis
* width, height: pixel width and height
* offsetx, offsety, offsetz: offset in x, y, and z-axis

#### leaptorch.Projector.set_parallel_beam(nangles, nrows, ncols, pheight, pwidth, crow, ccol, arange, phis)
set up parallel beam projection parameters
* nangles: number of projections
* nrows, ncols: detector rows and columns
* pheight, pwidth: pixel height and width
* crow, ccol: row and column centers
* arange: angular range of the projections (in degree)
* phis: all projection angles as a PyTorch tensor array (in degree)

#### leaptorch.Projector.set_cone_beam(nangles, nrows, ncols, pheight, pwidth, crow, ccol, arange, phis, sod, sdd)
set up cone beam projection parameters
* nangles: number of projections
* nrows, ncols: detector rows and columns
* pheight, pwidth: pixel height and width
* crow, ccol: row and column centers
* arange: angular range of the projections (in degree)
* phis: all projection angles as a PyTorch tensor array (in degree)
* sod, sdd: source to object distance (SOD), and source to detector distance (SDD)

#### leaptorch.Projector.set_modular_beam(nangles, nrows, ncols, pheight, pwidth, srcpos, modcenter, rowvec, colvec)
set up modular beam projection parameters

#### leaptorch.Projector.get_volume_dim()
get the dimension of image data. It return dimz, dimy, and dimx. 

#### leaptorch.Projector.get_projection_dim()
get the dimension of projection data (sinogram). It returns projection views, detector rows, and detector columns. 

#### leaptorch.Projector.load_param(param_fn)
load all image and projection parameters from a configuration file
* param_fn: path to a file storing all parameters

#### leaptorch.Projector.save_param(param_fn)
save all image and projection parameters into a file
* param_fn: path to a file storing all parameters

#### leaptorch.Projector.print_param()
print all image and projection parameter specifications

## Parameter File Format
The parameter configuration file is an ASCII text format. Below is an example of a parameter file content:
```
img_dimx = 512     # image x dimension
img_dimy = 512     # image y dimension
img_dimz = 1       # image z dimension
img_pwidth = 0.8   # image pixel width
img_pheight = 0.8  # image pixel height
img_offsetx = 0    # image x offset
img_offsety = 0    # image y offset
img_offsetz = 0    # image z offset
proj_geometry = parallel    # projection geometry type, parallel, cone, or modular
proj_arange = 180       # angular range in degree     
proj_nangles = 720      # number of projections
proj_nrows = 1          # detector row size
proj_ncols = 512        # detector column size
proj_pwidth = 0.8       # detector pixel width
proj_pheight = 0.8      # detector pixel height
proj_crow = 0           # detector row center
proj_ccol = 255.5       # detector column center
proj_phis =             # array of projection angles (size is the same as number of projections), seperated by comma (,)
proj_sod = 0            # source-to-object distance
proj_sdd = 0            # source-to-detector distance 
```



