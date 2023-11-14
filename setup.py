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


cuda = torch.cuda.is_available()
if cuda:

    source_files = ['main_projector.cpp', 
                 'tomographic_models.cpp', 'tomographic_models_c_interface.cpp', 
                 'projectors_cpu.cpp', 'projectors_SF_cpu.cpp',
                 'projectors_symmetric_cpu.cpp', 'parameters.cpp', 
                 'ramp_filter_cpu.cpp', 'cuda_utils.cu', 
                 'noise_filters.cu', 'ramp_filter.cu', 
                 'ray_weighting.cu', 'total_variation.cu', 
                 'projectors.cu', 'projectors_extendedSF.cu',
                 'projectors_symmetric.cu','projectors_SF.cu']
    for i in range(len(source_files)):
        source_files[i] = os.path.join('src', source_files[i])

    ext_mod = CUDAExtension(
        name='leapct',
        sources=source_files,
        extra_compile_args={'cxx': ['-g', '-D__USE_GPU', '-I%s' % pybind11.get_include()], 
                            'nvcc': ['-g', '-D__USE_GPU']}
    )
else:
    source_files=['main_projector.cpp', 
                 'tomographic_models.cpp', 'tomographic_models_c_interface.cpp', 
                 'projectors_cpu.cpp', 'projectors_SF_cpu.cpp',
                 'projectors_symmetric_cpu.cpp', 'parameters.cpp',
                 'ramp_filter_cpu.cpp']
    for i in range(len(source_files)):
        source_files[i] = os.path.join('src',source_files[i])

    ext_mod = CppExtension(
        name='leapct',
        sources=source_files,
        extra_compile_args={'cxx': ['-g', '-D__USE_CPU', '-I%s' % pybind11.get_include()]}
        #extra_compile_args=['-g', '-D__USE_CPU'],
    )


setup(
    name='leapct',
    version='0.91', 
    author='Hyojin Kim, Kyle Champley', 
    author_email='hkim@llnl.gov', 
    description='LivermorE AI Projector for Computed Tomography (LEAPCT)', 
    keywords='Machine Learning, ML, AI, Computed Tomography, CT, Differentiable Project, Forward Project, Back Project', 
    python_requires='>=3.6', 
    packages=find_packages("src"), 
    package_dir={'': 'src'},
    install_requires=['numpy', 'torch'], 
    py_modules=['leaptorch'], 
    ext_modules=[ext_mod], 
    cmdclass={'build_ext': BuildExtension}, 
)

