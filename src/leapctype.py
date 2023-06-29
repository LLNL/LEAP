################################################################################
# Copyright 2022-2022 Lawrence Livermore National Security, LLC and other 
# LEAP project developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# LivermorE AI Projector for Computed Tomography (LEAP)
# ctype projector class
################################################################################

import ctypes
import os
import sys
from sys import platform as _platform
from numpy.ctypeslib import ndpointer
import numpy as np


class Projector:
    ''' Python class for Projectors bindings
    Usage Example:
    from LEAP import Projector
    proj = Projector()
    proj.project(g,f)
    '''

    def __init__(self, lib_dir=""):
    	if len(lib_dir) > 0:
            current_dir = lib_dir
        else:
            current_dir = os.path.abspath(os.path.dirname(__file__))

		'''currentDirectoryPath         = os.path.dirname(os.path.realpath(__file__))

        if os.path.isdir(currentDirectoryPath) == False:
            currentDirectoryPath, thePartIdontWant = os.path.split(currentDirectoryPath)
        currentDirectoryPath = os.path.join(currentDirectoryPath, "lib")
        
        if _platform == "linux" or _platform == "linux2":
            fullProjectorsPath = os.path.join(currentDirectoryPath, "libprojectors.so")
        elif _platform == "darwin":
            fullProjectorsPath = os.path.join(currentDirectoryPath, "libprojectors.dylib")
        elif _platform == "win32":
            fullProjectorsPath = os.path.join(currentDirectoryPath, "libprojectors.dll")
        if os.path.isfile(fullProjectorsPath) == True:
            fullProjectorsPath = currentDirectoryPath
        else:
            fullProjectorsPath, thePartIdontWant = os.path.split(currentDirectoryPath)
        '''

        if _platform == "linux" or _platform == "linux2":
            import readline
            from ctypes import cdll
            self.libprojectors = cdll.LoadLibrary(os.path.join(current_dir, "libLEAP.so"))
        elif _platform == "darwin":
            from ctypes import cdll
            self.libprojectors = cdll.LoadLibrary(os.path.join(current_dir, "libLEAP.dylib"))
        elif _platform == "win32":
            from ctypes import windll
            self.libprojectors = windll.LoadLibrary(os.path.join(current_dir, "libLEAP.dll"))

    def reset(self):
        return self.libprojectors.reset()

    def printParameters(self):
        self.libprojectors.printParameters.restype = ctypes.c_bool
        return self.libprojectors.printParameters()
    
    def set_axisOfSymmetry(self,val):
        self.libprojectors.set_axisOfSymmetry.argtypes = [ctypes.c_float]
        self.libprojectors.set_axisOfSymmetry.restype = ctypes.c_bool
        return self.libprojectors.set_axisOfSymmetry(val)
        
    def setProjector(self,which):
        self.libprojectors.setProjector.argtypes = [ctypes.c_int]
        self.libprojectors.setProjector.restype = ctypes.c_bool
        return self.libprojectors.setProjector(which)
    
    def setVolumeParams(self, numX, numY, numZ, voxelWidth, voxelHeight=None, offsetX=None, offsetY=None, offsetZ=None):
        self.libprojectors.setVolumeParams.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
        self.libprojectors.setVolumeParams.restype = ctypes.c_bool
        if voxelHeight is None:
            voxelHeight = voxelWidth
        if offsetX is None:
            offsetX = 0.0
        if offsetY is None:
            offsetY = 0.0
        if offsetZ is None:
            offsetZ = 0.0
        return self.libprojectors.setVolumeParams(numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ)
	
    def setConeBeamParams(self, numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd):
        self.libprojectors.setConeBeamParams.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_float, ctypes.c_float]
        self.libprojectors.setConeBeamParams.restype = ctypes.c_bool
        return self.libprojectors.setConeBeamParams(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd)

    def setParallelBeamParams(self, numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis):
        self.libprojectors.setParallelBeamParams.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
        self.libprojectors.setParallelBeamParams.restype = ctypes.c_bool
        return self.libprojectors.setParallelBeamParams(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis)

    def setModularBeamParams(self, numAngles, numRows, numCols, pixelHeight, pixelWidth, sourcePositions, moduleCenters, rowVectors, colVectors):
        self.libprojectors.setModularBeamParams.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
        self.libprojectors.setModularBeamParams.restype = ctypes.c_bool
        return self.libprojectors.setModularBeamParams(numAngles, numRows, numCols, pixelHeight, pixelWidth, sourcePositions, moduleCenters, rowVectors, colVectors)
    
    def project(self,g,f):
        self.libprojectors.project.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
        self.libprojectors.project.restype = ctypes.c_bool
        self.libprojectors.project(g,f,True)
        return g
    
    def backproject(self,g,f):
        self.libprojectors.backproject.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
        self.libprojectors.backproject.restype = ctypes.c_bool
        self.libprojectors.backproject(g,f,True)
        return f

	
