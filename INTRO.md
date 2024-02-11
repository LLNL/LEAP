# LEAP: Differentiable forward and backward projectors for AI/ML-driven computed tomography applications

  
## Introduction
LivermorE AI Projector (**LEAP**) is a C++/CUDA library of 3D tomographic algorithms (pre-processing algorithms, projectors, and analytic (FBP) and iterative reconstruction algorithms) with a Python interface. The projectors (forward and back projection) are implemented for both multi-GPU and multi-core CPU and we provide bindings to PyTorch to achieve differentiable forward and backward projectors for AI/ML-driven Computed Tomography (CT) applications. 


## Installation

Installation details are covered on the [wiki](https://github.com/LLNL/LEAP/wiki) page.


## Getting Started

The direct Python interface to the C++/CUDA LEAP backend is implemented in [leapctype.py](https://github.com/LLNL/LEAP/blob/main/src/leapctype.py).

This file contains a class called tomographicModels which implements an API to the backend.  If one wishes to use LEAP as a collection of tomographic algorithms (not through a PyTorch neutral network), then users should use this class directly.  This API works with both numpy and torch tensors.  The torch tensors allow the LEAP algorithms to operate directly on data that is already on a GPU and thus does not rely on CPU-to-GPU data transfers.  There are a collection of demo scripts in the [demo_leapctype](https://github.com/LLNL/LEAP/tree/main/demo_leapctype) directory that demonstrate various functionality of the software packages.  To use this package start with the following:
```
from leapctype import *
leapct = tomographicModels()
```

The integration of LEAP to PyTorch is implemented in [leaptorch.py](https://github.com/LLNL/LEAP/blob/main/src/leaptorch.py).

Essentially this is a wrapper around the tomographicModels class in leapctype.py.  There are a collection of demo scripts in the [demo_leaptorch](https://github.com/LLNL/LEAP/tree/main/demo_leaptorch) directory that demonstrate how to use LEAP with PyTorch Neural Networks.

To start using the class to integrate into a PyTorch NN, start with the following:
```
from leaptorch import Projector
proj = Projector(forward_project=True, use_static=True, use_gpu=use_cuda, gpu_device=device)
```

Note that the Projector class contains the tomographicModels class as the member function named leapct.  Thus one can access the tomographicModels class functions through proj.leapct


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

