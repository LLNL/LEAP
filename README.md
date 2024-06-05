# LivermorE AI Projector for Computed Tomography (LEAP)
This is a C++/CUDA library (Linux, Windows, and Mac*) of 3D tomographic algorithms (pre-processing algorithms, projectors, and analytic (FBP) and iterative reconstruction algorithms) with a Python interface.  The projectors (forward and back projection) are implemented for both multi-GPU and multi-core CPU and we provide bindings to PyTorch to achieve differentiable forward and backward projectors for AI/ML-driven Computed Tomography (CT) applications.

<p align="center">
  <img src=https://github.com/LLNL/LEAP/blob/main/documentation/LEAPoverview.png>
</p>

There are a lot of CT reconstruction packages out there, so why choose LEAP?  In short, LEAP has more accurate projectors and FBP algorithms, more features, and most algorithms run faster than other popular CT reconstruction packages, but here is a more detailed list:
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
15) Easy-to-build executable because the only dependency is CUDA.  Python API can be run with or without PyTorch (of course the neural network stuff requires PyTorch).
16) Permissible license.

Physics-based modeling and correction algorithms (e.g., scatter correction, beam hardening correction (BHC), dual energy decomposition, and SIRZ) can be applied when used with the [XrayPhysics](https://github.com/kylechampley/XrayPhysics) package.

*Mac version does not have GPU support and some featurings are missing.

## Installation and Usage

Documentation is available [here](https://leapct.readthedocs.io/)

Installation and usage information is posted on the [wiki](https://github.com/LLNL/LEAP/wiki) page

Demo scripts for most functionality in the [demo_leapctype](https://github.com/LLNL/LEAP/tree/main/demo_leapctype) directory

Demo scripts for AI/ML/DL applications in the [demo_leaptorch](https://github.com/LLNL/LEAP/tree/main/demo_leaptorch) directory

## Example Results

As a simple demonstration of the accuracy of our projectors we show below the results of FDK reconstructions using ASTRA and LEAP of the walnut CT data.  The LEAP reconstruction has 1.7 times higher SNR and reconstructed this data 7.5 times faster than ASTRA.
<p align="center">
  <img src=https://github.com/LLNL/LEAP/blob/main/results/walnut_comparison.png>
</p>

## Future Releases

For the next releases, we are working on the following:
1) Fixes of bugs reported by our users
2) Feature requests from our users
3) More noise reduction filters
4) AMD GPU Support
5) cone-parallel geometry support
6) PyQt GUI

## Authors
Kyle Champley (champley@gmail.com)

Hyojin Kim (hkim@llnl.gov)   


## License
LEAP is distributed under the terms of the MIT license. All new contributions must be made under this license. See LICENSE in this directory for the terms of the license.
See [LICENSE](LICENSE) for more details.  
SPDX-License-Identifier: MIT  
LLNL-CODE-848657  

Please cite our work by referencing this github page and citing our [article](https://arxiv.org/abs/2307.05801):

Hyojin Kim and Kyle Champley, "Differentiable Forward Projector for X-ray Computed Tomography”, ICML, 2023
