################################################################################
# Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
# LEAP project developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# LivermorE AI Projector for Computed Tomography (LEAP)
# setup_ctype.py for pytorch-free leapct module
################################################################################
import os
import pathlib

from setuptools import setup, find_packages
from setuptools.command.install import install

from sys import platform as _platform
if _platform == "linux" or _platform == "linux2":
    lib_fname = 'build/lib/libleap.so'
    os.system(r'sh ./etc/build.sh')
elif _platform == "win32":
    lib_fname = r'win_build\bin\Release\libleap.dll'
    os.system(r'.\etc\win_build.bat')
    
    import site
    copy_text = 'copy ' + str(lib_fname) + ' ' + str(os.path.join(site.getsitepackages()[1], 'libleap.dll'))
    os.system(copy_text)
    
elif _platform == "darwin":
    lib_fname = 'build/lib/libleap.dylib'
    os.system(r'sh ./etc/build.sh')

setup(
    name='leapct',
    version='0.99', 
    author='Kyle Champley, Hyojin Kim', 
    author_email='champley@gmail.com', 
    description='LivermorE AI Projector for Computed Tomography (LEAPCT)', 
    keywords='Machine Learning, ML, AI, Computed Tomography, CT, Differentiable Project, Forward Project, Back Project', 
    python_requires='>=3.6', 
    packages=find_packages("src"), 
    package_dir={'': 'src'},
    install_requires=['numpy'], 
    py_modules=['leapctype'], 
    package_data={'': [lib_fname]},
)
