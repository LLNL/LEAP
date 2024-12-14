################################################################################
# Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
# LEAP project developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# LivermorE AI Projector for Computed Tomography (LEAP)
# setup.py for pytorch module
################################################################################

from setuptools import setup, find_packages
from setuptools.command.install import install
from torch.utils.cpp_extension import CUDAExtension
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import BuildExtension
import os
import pybind11
import torch
from sys import platform as _platform

## todo
# compiler options (optimization flags, cufft options)
# leapctype : libleap.so 

cpp_files=[
    'analytic_ray_tracing.cpp', 
    'cpu_utils.cpp', 
    'file_io.cpp', 
    'filtered_backprojection.cpp', 
    'find_center_cpu.cpp', 
    'list_of_tomographic_models.cpp', 
    'parameters.cpp', 
    'phantom.cpp', 
    'projectors.cpp', 
    'projectors_Joseph_cpu.cpp', 
    'projectors_SF_cpu.cpp', 
    'projectors_Siddon_cpu.cpp', 
    'projectors_symmetric_cpu.cpp', 
    'ramp_filter_cpu.cpp', 
    'ray_weighting_cpu.cpp', 
    'rebin.cpp', 
    'sensitivity_cpu.cpp', 
    'resample_cpu.cpp', 
    'sinogram_replacement.cpp', 
    'tomographic_models_c_interface.cpp', 
    'tomographic_models.cpp', 
]

cuda_files=[
    'bilateral_filter.cu', 
    'guided_filter.cu', 
    'cuda_utils.cu', 
    'matching_pursuit.cu', 
    'noise_filters.cu', 
    'projectors_attenuated.cu', 
    'projectors_extendedSF.cu', 
    'projectors_Joseph.cu', 
    'projectors_SF.cu', 
    'projectors_Siddon.cu', 
    'projectors_symmetric.cu', 
    'ramp_filter.cu', 
    'ray_weighting.cu', 
    'scatter_models.cu', 
    'sensitivity.cu', 
    'resample.cu', 
    'total_variation.cu',
    'geometric_calibration.cu',
    'analytic_ray_tracing_gpu.cu',
    'backprojectors_VD.cu',
]

cuda = torch.cuda.is_available()
if cuda:
    source_files = []
    for cpp_file in cpp_files:
        source_files.append(os.path.join('src', cpp_file))
    for cuda_file in cuda_files:
        source_files.append(os.path.join('src', cuda_file))

    # optionally we could add '-O3'
    # or extra_link_args=["-std=c++11"]
    rocm = "AMD" in torch.cuda.get_device_name(0)
    if rocm: # AMD ROCM GPU
        extra_compile_args={'cxx': ['-D__USE_GPU'], 
                            'nvcc': ['-D__USE_GPU', '-O3']}
        libraries = []
    else: # CUDA GPU
        extra_compile_args={'cxx': ['-D__USE_GPU'], 
                            'nvcc': ['-D__USE_GPU', '-O3']}
        #extra_compile_args={'cxx': ['-D__USE_GPU', '-lcufft', '-D__INCLUDE_CUFFT'], 
        #                    'nvcc': ['-D__USE_GPU', '-O3', '-lcufft', '-D__INCLUDE_CUFFT']}
        #libraries = ['cufft']
        libraries = []
    ext_mod = CUDAExtension(
        name='leapct',
        sources=source_files,
        extra_compile_args=extra_compile_args,
        libraries = libraries,
        #extra_link_args=["-lcufft"], 
        extra_cflags=['-O3'])
else:
    source_files = []
    for cpp_file in cpp_files:
        source_files.append(os.path.join('src', cpp_file))

    ext_mod = CppExtension(
        name='leapct',
        sources=source_files,
        extra_cflags=['-O3'],
        #extra_link_args=["-lcufft"], 
        extra_compile_args={'cxx': ['-D__USE_CPU']}
        #extra_compile_args=['-g', '-D__USE_CPU'],
    )

setup(
    name='leapct',
    version='1.26', 
    author='Kyle Champley, Hyojin Kim', 
    author_email='champley@gmail.com, hkim@llnl.gov', 
    description='LivermorE AI Projector for Computed Tomography (LEAPCT)', 
    keywords='Machine Learning, ML, AI, Computed Tomography, CT, Differentiable Project, Forward Project, Back Project', 
    python_requires='>=3.6', 
    packages=find_packages("src"), 
    package_dir={'': 'src'},
    install_requires=['numpy', 'torch'], 
    py_modules=['leaptorch','leapctype', 'leap_filter_sequence', 'leap_preprocessing_algorithms'], 
    ext_modules=[ext_mod], 
    cmdclass={'build_ext': BuildExtension}, 
    #package_data={'': [lib_fname]},
)

