# LivermorE AI Projector for Computed Tomography (LEAP)
This is a C++/CUDA library (Linux, Windows, and Mac*) of 3D tomographic algorithms (pre-processing algorithms, projectors, and analytic (FBP) and iterative reconstruction algorithms) with a Python interface.  The projectors (forward and back projection) are implemented for both multi-GPU and multi-core CPU and we provide bindings to PyTorch to achieve differentiable forward and backward projectors for AI/ML-driven Computed Tomography (CT) applications.

<p align="center">
  <img src=https://github.com/LLNL/LEAP/blob/main/documentation/LEAPoverview.png>
</p>

There are a lot of CT reconstruction packages out there, so why choose LEAP?  In short, LEAP has more accurate projectors and FBP algorithms, more features, and most algorithms run as fast or faster than other popular CT reconstruction packages, but [here](https://github.com/LLNL/LEAP/blob/main/LEAP_features.md) is a more detailed list.

The LEAP PyQt GUI can be found [here](https://github.com/kylechampley/LEAPCT-UI-GUI).

Physics-based modeling and correction algorithms (e.g., scatter correction, beam hardening correction (BHC), dual energy decomposition, and SIRZ) can be applied when used with the [XrayPhysics](https://github.com/kylechampley/XrayPhysics) package.

*Mac version does not have GPU support and some featurings are missing.

## Installation and Usage

Documentation is available [here](https://leapct.readthedocs.io/)

Installation and usage information is posted on the [wiki](https://github.com/LLNL/LEAP/wiki) page

Demo scripts for most functionality in the [demo_leapctype](https://github.com/LLNL/LEAP/tree/main/demo_leapctype) directory

Demo scripts for AI/ML/DL applications in the [demo_leaptorch](https://github.com/LLNL/LEAP/tree/main/demo_leaptorch) directory

## Example Results

As a simple demonstration of the accuracy of our projectors we show below the results of FDK reconstructions using ASTRA and LEAP of the walnut CT data.  The LEAP reconstruction has 1.7 times higher SNR than ASTRA.  An explanation for this improvement in SNR can be found [here](https://github.com/LLNL/LEAP/blob/main/results/SF_vs_VD.md).
<p align="center">
  <img src=https://github.com/LLNL/LEAP/blob/main/results/walnut_comparison.png>
</p>

## Future Releases

For the next releases, we are working on the following:
1) Fixes of bugs reported by our users
2) AMD GPU Support
3) multi-material beam hardening correction algorithms for more than two materials and that account for variable takeoff angle and graded collimator/ bowtie filter
4) triple energy decomposition
5) spectral calibration

If you are interested in requesting a new feature in LEAP, please make a post in the [Feature Request](https://github.com/LLNL/LEAP/discussions/88) discussion.

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


If you use RDLS, azimuthalFilter, or histogramSparsity, please cite the following paper:

Champley, Kyle M., Michael B. Zellner, Joseph W. Tringe, and Harry E. Martz Jr. "Methods for Few-View CT Image Reconstruction." arXiv preprint arXiv:2410.07552 (2024).
