# LivermorE AI Projector for Computed Tomography (LEAP)
This is a C/C++/CUDA library of tomographic projectors (forward and back projection) implemented for both multi-GPU and multi-core CPU.  We provide bindings to PyTorch to achieve differentiable forward and backward projectors for AI/ML-driven Computed Tomography (CT) applications.

Our projectors are implemented for the standard 3D CT geometry types: parallel-, fan-, and cone-beam.  These geometry types accomodate shifts of the detectors and non-uniform angular spacing.  For added flexibility, we also provide a flexible modular-beam format where the user may specify the location and orientation of every source and detector pair.  All projectors use the Separable Footprint [Long, Fessler, and Balter, TMI, 2010] method which provides a matched projector pair that models the finite size of the voxel and detector pixel.  These matched projectors ensure convergence and provide accurate, smooth results.  Unmatch projectors or those projectors that do not model the finite size of the voxel or detector pixel may produce artifacts when used over enough iterations [DeMan and Basu, PMB, 2004].

We also provide projectors and analytic inversion algorithms, i.e., FBP, for a few specialized x-ray/Radon transforms:
1) Cylindrically-symmetric/anitsymmetric objects (related to the Abel Transform) in parallel- and cone-beam geometries with user-specified symmetry axis [Champley and Maddox, Optica, 2021].  These are often used in flash radiography applications.
2) Attenuated Radon Transform (ART) for parallel-beam geometries.  These are used in parallel-hole collimator SPECT and Volumetric Additive Manufacturing (VAM).

In addition to the projectors, we also provide a few other algorithms for tomographic imaging, including:
1) Quantitatively-accurate analytic inversion algorithms, i.e., Filtered Backprojection (FBP) for each geometry except modular-beam.
2) A GPU implementation of 3D anisotropic Total Variation (TV) functional, gradient, and quadratic form to be used in regularized reconstruction.
3) Python implementations of some iterative reconstruction algorithms: OSEM, SART, ASD-POCS, and RWLS.

The CPU- and GPU-based projectors are nearly identical (32-bit floating point precision) and are quantitatively accurate and thus can be used in conjuction with physics-based corrections, such as, scatter and beam hardening correction.  If one is looking for a more general-purpose and full functioning CT software package (it does not, however, work with PyTorch and is closed-source), see LTT (https://www.sciencedirect.com/science/article/abs/pii/S0963869521001948)


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
