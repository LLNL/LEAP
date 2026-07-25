################################################################################
# Copyright 2022-2023 Lawrence Livermore National Security, LLC and other
# LEAP project developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# LivermorE AI Projector for Computed Tomography (LEAP)
# CPU-only build of the pytorch-free leapct module.
#
# This mirrors setup.py but compiles ONLY the CPU library (via etc/cpu_build.sh,
# which runs `cmake -DCPUONLY=ON`). It requires a C++ toolchain + CMake + OpenMP
# but NO CUDA, so it can be built on machines / base images without an NVIDIA
# GPU. Intended to be driven by ../Dockerfile.cpu (see ../package-leap-cpu).
################################################################################
import os

from setuptools import setup, find_packages

from sys import platform as _platform

if _platform == "linux" or _platform == "linux2":
    lib_ext = "so"
elif _platform == "darwin":
    lib_ext = "dylib"
else:
    print("setup_cpu.py supports Linux and macOS only")
    quit()

# Make sure a stray GPU library from a previous build is never shipped in the
# CPU-only wheel (the wheel packages everything staged under build/lib).
for _stray in ("build/lib/libleapct.so", "build/lib/libleapct.dylib"):
    if os.path.isfile(_stray):
        os.remove(_stray)

# Build the CPU-only shared library (no CUDA required).
retVal = os.system(r"sh ./etc/cpu_build.sh")
if retVal != 0:
    print("Failed to compile!")
    quit()

# Stage the freshly built library alongside the other build artifacts so it is
# picked up by package_data, matching the layout used by setup.py.
lib_fname_cpu = "cpu_build/lib_cpu/libleapct_cpu." + lib_ext
os.makedirs("build/lib", exist_ok=True)
staged_lib = os.path.join("build/lib", "libleapct_cpu." + lib_ext)
if os.system("cp " + str(lib_fname_cpu) + " " + str(staged_lib)) != 0:
    print("Failed to stage CPU library!")
    quit()
lib_fname_cpu = staged_lib

setup(
    name="leapct",
    version="2.0",
    author="Kyle Champley, Hyojin Kim",
    author_email="champley@gmail.com, hkim@llnl.gov",
    description="LivermorE AI Projector for Computed Tomography (LEAPCT) - CPU-only build",
    keywords="Machine Learning, ML, AI, Computed Tomography, CT, Differentiable Project, Forward Project, Back Project",
    python_requires=">=3.6",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=["numpy", "imageio", "scipy"],
    py_modules=[
        "leapctype",
        "leap_filter_sequence",
        "leap_preprocessing_algorithms",
        "xrayphysics",
        "leapctserver",
    ],
    package_data={"": [lib_fname_cpu]},
)
