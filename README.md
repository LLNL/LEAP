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

### Installing a wheel

Install a pre-built wheel with `pip`:

```bash
python -m pip install leapct-<version>-<tags>.whl
```

If the wheel was built with GPU support, use it with a compatible CUDA or ROCm runtime in
that Python environment. The optional PyTorch bindings require the `torch` extra or a
separate PyTorch installation:

```bash
python -m pip install 'leapct[torch]'
```

### Editable installs for development

For local development, use an editable install. This builds the native library once and
installs it as package data while leaving the Python sources editable. If `LEAP_GPU` is not
specified, the build probes for available accelerator compilers and selects CUDA or HIP
only when exactly one is detected. If both CUDA and HIP are detected, the build fails and
asks you to choose explicitly. Use `LEAP_GPU=None` to force a CPU-only build:

```bash
python -m pip install -e . -Ccmake.define.LEAP_GPU=None
```

Choose a GPU backend explicitly by changing the CMake definition, for example:

```bash
python -m pip install -e . -Ccmake.define.LEAP_GPU=NVIDIA
python -m pip install -e . -Ccmake.define.LEAP_GPU=AMD
```

### Building wheels

Build a wheel from a checkout with Python's standard build frontend:

```bash
python -m build --wheel .
```

If `LEAP_GPU` is not specified, the build probes for HIP and CUDA compiler support. The
build selects the detected accelerator when exactly one is available, selects CPU-only when
neither is available, and fails as ambiguous when both are available. Specify the backend
explicitly when you need a particular target:

```bash
python -m build --wheel -Ccmake.define.LEAP_GPU=None .
python -m build --wheel -Ccmake.define.LEAP_GPU=NVIDIA .
python -m build --wheel -Ccmake.define.LEAP_GPU=AMD .
```

Locally distributed GPU wheels should include a PEP 440 local version tag naming the
runtime ABI they were built against. The helper script does this with
`SETUPTOOLS_SCM_PRETEND_VERSION`:

```bash
scripts/build-gpu-wheel.sh cu124 NVIDIA
scripts/build-gpu-wheel.sh rocm6.3 AMD
```

For ROCm/HIP builds, see [docs/hipify-strategy.md](docs/hipify-strategy.md). To validate a
wheel against a different compatible runtime without rebuilding, see
[scripts/cross-runtime-smoke-test.md](scripts/cross-runtime-smoke-test.md).

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
