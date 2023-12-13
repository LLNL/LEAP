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


cpp_files=['main_projector.cpp', 'tomographic_models.cpp', 'tomographic_models_c_interface.cpp', 
           'parameters.cpp', 'projectors.cpp', 'filtered_backprojection.cpp', 
           'projectors_SF_cpu.cpp', 'projectors_Joseph_cpu.cpp', 'projectors_symmetric_cpu.cpp', 
           'projectors_Siddon_cpu.cpp', 'sensitivity_cpu.cpp', 'ramp_filter_cpu.cpp', 'cpu_utils.cpp', 
           'phantom.cpp', 'ray_weighting_cpu.cpp']

cuda_files=['projectors_SF.cu', 'projectors_extendedSF.cu', 
            'projectors_Joseph.cu', 'projectors_symmetric.cu', 
            'projectors_attenuated.cu', 'projectors_Siddon.cu', 
            'sensitivity.cu', 'ramp_filter.cu', 
            'noise_filters.cu', 'total_variation.cu', 
            'ray_weighting.cu', 'cuda_utils.cu']


cuda = torch.cuda.is_available()
if cuda:
    source_files = []
    for cpp_file in cpp_files:
        source_files.append(os.path.join('src', cpp_file))
    for cuda_file in cuda_files:
        source_files.append(os.path.join('src', cuda_file))

    ext_mod = CUDAExtension(
        name='leapct',
        sources=source_files,
        extra_compile_args={'cxx': ['-g', '-D__USE_GPU', '-I%s' % pybind11.get_include()], 
                            'nvcc': ['-g', '-D__USE_GPU']}
    )
else:
    source_files = []
    for cpp_file in cpp_files:
        source_files.append(os.path.join('src', cpp_file))

    ext_mod = CppExtension(
        name='leapct',
        sources=source_files,
        extra_compile_args={'cxx': ['-g', '-D__USE_CPU', '-I%s' % pybind11.get_include()]}
        #extra_compile_args=['-g', '-D__USE_CPU'],
    )


setup(
    name='leapct',
    version='0.99', 
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

