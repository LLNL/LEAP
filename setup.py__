################################################################################
# Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
# LEAP project developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# LivermorE AI Projector for Computed Tomography (LEAP)
# setup.py for pytorch module
################################################################################
import os
import pathlib

from setuptools import setup, find_packages
from setuptools.command.install import install

from sys import platform as _platform
if _platform == "linux" or _platform == "linux2":
    lib_fname = 'build/lib/libleapct.so'
    retVal = os.system(r'sh ./etc/build.sh')
    if retVal != 0:
        print('Failed to compile!')
        quit()
    
elif _platform == "win32":
    lib_fname = r'win_build\bin\Release\libleapct.dll'
    retVal = os.system(r'.\etc\win_build_agn.bat')
    if retVal != 0:
        print('Failed to compile!')
        quit()
    
    import site
    copy_text = 'copy ' + str(lib_fname) + ' ' + str(os.path.join(site.getsitepackages()[1], 'libleapct.dll'))
    os.system(copy_text)
    
elif _platform == "darwin":
    lib_fname = 'build/lib/libleapct.dylib'
    retVal = os.system(r'sh ./etc/build.sh')
    if retVal != 0:
        print('Failed to compile!')
        quit()

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
    py_modules=['leaptorch', 'leapctype', 'leap_filter_sequence', 'leap_preprocessing_algorithms'], 
    package_data={'': [lib_fname]},
)
