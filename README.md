# LivermorE AI Projector for Computed Tomography (LEAP)
This is a C/C++/CUDA library of 3D tomographic projectors (forward and back projection) implemented for both multi-GPU and multi-core CPU.  We provide bindings to PyTorch to achieve differentiable forward and backward projectors for AI/ML-driven Computed Tomography (CT) applications.

<p align="center">
  <img src=https://github.com/LLNL/LEAP/blob/main/documentation/LEAPoverview.png>
</p>

There are a lot of CT reconstruction packages out there, so why choose LEAP?
1) **Seamless integration with PyTorch** using torch.nn.Module and torch.autograd.Function to enable differentiable forward and backward projectors for AI/ML-driven Computed Tomography (CT) applications.
2) **Quantitatively accurate, matched (forward and back) projector pairs** that model the finite size of the voxel and detector pixel; very similar to the Separable Footprint method [Long, Fessler, and Balter, TMI, 2010].  These matched projectors ensure convergence and provide accurate, smooth results.  Unmatch projectors or those projectors that do not model the finite size of the voxel or detector pixel may produce artifacts when used over enough iterations [DeMan and Basu, PMB, 2004].
3) **Multi-GPU and multi-core CPU implementations of all algorithms** that are as fast or faster than other popular CT packages.
4) **Flexible 3D CT geometry** specification that allows users to specify arbitrary shifts of the detectors and non-uniform angular spacing.
5) **Flexible 3D CT volume** specification.
6) **Quantitatively accurate and flexible** analytic reconstruction algorithms, i.e., Filtered Backprojection (**FBP**).
7) Special-case FBP algorithms that are rarely included in other packages, such as helical, truncated projections, offset detector scan, and Attenuated Radon Transform.
8) Special-case models such as the Attenuated Radon Transform (SPECT and VAM applications) and reconstruction of cylindrically-symmetric objects (flash x-ray applications).
9) Iterative reconstruction algorithms: OSEM, OS-SART, ASD-POCS, RWLS, ML-TR.
10) Fast multi-GPU 3D densoing methods.
11) Easy-to-build executable because the only dependency is CUDA.  Python API can be run with or without PyTorch (of course the neural network stuff requires PyTorch).
12) Permissible license.

If one is looking for a more general-purpose and full functioning CT software package (it does not, however, work with PyTorch and is closed-source), see LTT (https://www.sciencedirect.com/science/article/abs/pii/S0963869521001948)


## Installation and Usage

Installation and usage information is posted on the wiki page here: https://github.com/LLNL/LEAP/wiki


## Authors
Kyle Champley (champley@gmail.com)

Hyojin Kim (hkim@llnl.gov)   


## License
LEAP is distributed under the terms of the MIT license. All new contributions must be made under this license. See LICENSE in this directory for the terms of the license.
See [LICENSE](LICENSE) for more details.  
SPDX-License-Identifier: MIT  
LLNL-CODE-848657  

Please cite our work by referencing this github page and citing our article: Hyojin Kim and Kyle Champley, "Differentiable Forward Projector for X-ray Computed Tomography”, ICML, 2023
