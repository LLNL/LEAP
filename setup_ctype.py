################################################################################
# Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
# LEAP project developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# LivermorE AI Projector for Computed Tomography (LEAP)
# setup.py for ctype binding
################################################################################

from setuptools import setup, Extension, Command, find_packages
import setuptools.command.develop
import setuptools.command.build_ext
import setuptools.command.install
#import distutils.command.build
#import subprocess
#import sys
#import os


class install(setuptools.command.install.install):
    def run(self):
        setuptools.command.install.install.run(self)

setup(name = 'leapct',
    version = '0.1',
    description='',
    cmdclass = {'install': install},
    packages=find_packages(), 
    package_dir={'leap': 'leap'},
    package_data={
        'leap': [
            'leap/tomographic_models_c_interface.so',
            'leap/tomographic_models_c_interface.cpp',
            'leap/tomographic_models_c_interface.h',
        ]
    },
    include_package_data=True
);
