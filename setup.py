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
import pybind11
import torch


cuda = torch.cuda.is_available()
if cuda:
    ext_mod = CUDAExtension(
        name='leapct',
        sources=['src/main_projector.cpp', 
                 'src/tomographic_models.cpp', 'src/tomographic_models_c_interface.cpp', 
                 'src/projectors_cpu.cpp', 'src/projectors_SF_cpu.cpp',
                 'src/projectors_symmetric_cpu.cpp', 'src/parameters.cpp', 
                 'src/cuda_utils.cu', 
                 'src/noise_filters.cu', 'src/ramp_filter.cu', 
                 'src/ray_weighting.cu', 'src/total_variation.cu', 
                 'src/projectors.cu', 'src/projectors_SF.cu', ],
        extra_compile_args={'cxx': ['-g', '-D__USE_GPU', '-I%s' % pybind11.get_include()], 
                            'nvcc': ['-g', '-D__USE_GPU']}
    )
else:
    ext_mod = CppExtension(
        name='leapct',
        sources=['src/main_projector.cpp', 
                 'src/tomographic_models.cpp', 'src/tomographic_models_c_interface.cpp', 
                 'src/projectors_cpu.cpp', 'src/projectors_SF_cpu.cpp',
                 'src/projectors_symmetric_cpu.cpp', 'src/parameters.cpp'],
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

