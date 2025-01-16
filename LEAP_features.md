# LEAP Features

1) **Seamless integration with PyTorch** using torch.nn.Module and torch.autograd.Function to enable differentiable forward and backward projectors for AI/ML-driven Computed Tomography (CT) applications.
2) **Quantitatively accurate, matched (forward and back) projector pairs** that model the finite size of the voxel and detector pixel; very similar to the Separable Footprint method [Long, Fessler, and Balter, TMI, 2010].  These matched projectors ensure convergence and provide accurate, smooth results.  Unmatch projectors or those projectors that do not model the finite size of the voxel or detector pixel may produce artifacts when used over enough iterations [DeMan and Basu, PMB, 2004].
3) **Multi-GPU and multi-core CPU implementations of all algorithms** that are as fast or faster than other popular CT reconstruction packages.
4) **Algorithms not limited by the amount of GPU memory**.
5) **Flexible 3D CT geometry** specification that allows users to specify arbitrary shifts of the source and detector positions, non-uniform angular spacing, and more.
6) **Flexible 3D CT volume** specification.
7) **Quantitatively accurate and flexible** analytic reconstruction algorithms, i.e., Filtered Backprojection (**FBP**).
8) Can **avoid** costly **CPU-to-GPU data transfers** by performing operations on data already on a GPU. 
9) Special-case FBP algorithms that are rarely included in other packages, such as helical, truncated projections, offset detector scan, and Attenuated Radon Transform.
10) Special-case models such as the Attenuated Radon Transform (SPECT and VAM applications) and reconstruction of cylindrically-symmetric objects (flash x-ray applications).
11) Iterative reconstruction algorithms: OSEM, OS-SART, ASD-POCS, RWLS, RDLS, ML-TR, IFBP (RWLS-SARR)
12) Fast multi-GPU 3D densoing methods.
13) Pre-processing algorithms: outlier correction, detector deblur, ring removal, scatter correction, metal artifact reduction (MAR), multi-material beam hardening correction (BHC), dual energy decomposition, and SIRZ
14) Easy-to-use, simple API.
15) [PyQt GUI](https://github.com/kylechampley/LEAPCT-UI-GUI)
16) Easy-to-build executable because the only dependency is CUDA.  Python API can be run with or without PyTorch (of course the neural network stuff requires PyTorch).
17) Permissible license.
