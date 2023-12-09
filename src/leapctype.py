################################################################################
# Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
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


class tomographicModels:
    ''' Python class for tomographicModels bindings
    Usage Example:
    from leapctype import *
    leapct = tomographicModels()
    leapct.setConeParams(...)
    leapct.setDefaultVolume(...)
    ...
    leapct.project(g,f)
    '''

    def __init__(self, lib_dir=""):
        if len(lib_dir) > 0:
            current_dir = lib_dir
        else:
            current_dir = os.path.abspath(os.path.dirname(__file__))

        if _platform == "linux" or _platform == "linux2":
            import readline
            from ctypes import cdll
            self.libprojectors = cdll.LoadLibrary(os.path.join(current_dir, "../build/lib/libleap.so"))
        elif _platform == "win32":
            from ctypes import windll
            self.libprojectors = windll.LoadLibrary(os.path.join(current_dir, r'..\win_build\bin\Release\libLEAP.dll'))

    def reset(self):
        return self.libprojectors.reset()

    def printParameters(self):
        return self.print_parameters()

    def print_parameters(self):
        self.libprojectors.print_parameters.restype = ctypes.c_bool
        return self.libprojectors.print_parameters()

    ###################################################################################################################
    ###################################################################################################################
    # THIS SECTION OF FUNCTIONS SET THE CT SCANNER GEOMETRY PARAMETERS
    ###################################################################################################################
    ###################################################################################################################
    def set_conebeam(self, numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau=0.0, helicalPitch=0.0):
        '''
        The origin of the coordinate system is always at the center of rotation
        
        The first three parameters specify the projection data array size
        numAngles = number of projection angles
        numRows = number of rows in the x-ray detector
        numCols = number of columns in the x-ray detector
        
        The next two parameters specify the detector pixel size in mm
        pixelHeight = the detector pixel pitch (i.e., pixel size) between detector rows, measured in mm
        pixelWidth = the detector pixel pitch (i.e., pixel size) between detector columns, measured in mm
        
        The next two parameters specify the placement of the detector, i.e.,
        changing these parameters causes shifts in the detector location relative to the source
        centerRow = the detector pixel row index for the ray that passes from the source, through the origin, and hits the detector
        centerCol = the detector pixel column index for the ray that passes from the source, through the origin, and hits the detector
        
        phis = a numpy array for specifying the angles of each projection, measured in degrees
        
        sod = source to object distance, measured in mm; this can also be viewed as the source to center of rotation distance
        sdd = source to detector distance, measured in mm
        '''
        self.libprojectors.set_conebeam.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
        self.libprojectors.set_conebeam.restype = ctypes.c_bool
        if type(phis) is not np.ndarray:
            angularRange = float(phis)
            phis = self.setAngleArray(numAngles, angularRange)
        return self.libprojectors.set_conebeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau, helicalPitch)
    
    def set_coneBeam(self, numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau=0.0, helicalPitch=0.0):
        return self.set_conebeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau, helicalPitch)
    
    def set_fanbeam(self, numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau=0.0):
        ''' see comments in the setConeBeamParams function '''
        self.libprojectors.set_fanbeam.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_float, ctypes.c_float, ctypes.c_float]
        self.libprojectors.set_fanbeam.restype = ctypes.c_bool
        if type(phis) is not np.ndarray:
            angularRange = float(phis)
            phis = self.setAngleArray(numAngles, angularRange)
        return self.libprojectors.set_fanbeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau)
        
    def set_fanBeam(self, numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau=0.0):
        return self.set_fanbeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau)

    def set_parallelbeam(self, numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis):
        ''' see comments in the setConeBeamParams function '''
        self.libprojectors.set_parallelbeam.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
        self.libprojectors.set_parallelbeam.restype = ctypes.c_bool
        if type(phis) is not np.ndarray:
            angularRange = float(phis)
            phis = self.setAngleArray(numAngles, angularRange)
        return self.libprojectors.set_parallelbeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis)

    def set_parallelBeam(self, numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis):
        return self.set_parallelbeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis)

    def set_modularbeam(self, numAngles, numRows, numCols, pixelHeight, pixelWidth, sourcePositions, moduleCenters, rowVectors, colVectors):
        self.libprojectors.set_modularbeam.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
        self.libprojectors.set_modularbeam.restype = ctypes.c_bool
        return self.libprojectors.set_modularbeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, sourcePositions, moduleCenters, rowVectors, colVectors)
    
    def set_modularBeam(self, numAngles, numRows, numCols, pixelHeight, pixelWidth, sourcePositions, moduleCenters, rowVectors, colVectors):
        return self.set_modularbeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, sourcePositions, moduleCenters, rowVectors, colVectors)
    
    def set_tau(self, tau):
        self.libprojectors.set_tau.argtypes = [ctypes.c_float]
        self.libprojectors.set_tau.restype = ctypes.c_bool
        return self.libprojectors.set_tau(tau)
    
    def set_helicalPitch(self, h):
        self.libprojectors.set_helicalPitch.argtypes = [ctypes.c_float]
        self.libprojectors.set_helicalPitch.restype = ctypes.c_bool
        return self.libprojectors.set_helicalPitch(h)
        
    def set_normalizedHelicalPitch(self, h_normalized):
        self.libprojectors.set_normalizedHelicalPitch.argtypes = [ctypes.c_float]
        self.libprojectors.set_normalizedHelicalPitch.restype = ctypes.c_bool
        return self.libprojectors.set_normalizedHelicalPitch(h_normalized)
    
    ###################################################################################################################
    ###################################################################################################################
    # THIS SECTION OF FUNCTIONS SET THE CT VOLUME PARAMETERS
    ###################################################################################################################
    ###################################################################################################################
    def set_volume(self, numX, numY, numZ, voxelWidth=None, voxelHeight=None, offsetX=None, offsetY=None, offsetZ=None):
        '''
        The first three parameters specify the volume array size
        For parallel- and fan-beam data, it is required that numRows=numZ
        numX = number of voxels in the x-dimension
        numY = number of voxels in the y-dimension
        numZ = number of voxels in the z-dimension
        
        The next two parameters specify the size of the voxels in mm
        For parallel- and fan-beam data, it is required that pixelHeight=voxelHeigh
        voxelWidth = voxel pitch (size) in the x and y dimensions
        voxelHeight = voxel pitch (size) in the z dimension
        
        The final three parameters specify the location of the voxel array
        The default position is that the voxels are centered around the origin
        and these parameters allow one to shift the volume around
        For parallel- and fan-beam data, it is required that offsetZ=0.0
        offsetX = shift the volume in the x-dimension, measured in mm
        offsetY = shift the volume in the y-dimension, measured in mm
        offsetZ = shift the volume in the z-dimension, measured in mm
        '''
        self.libprojectors.set_volume.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
        self.libprojectors.set_volume.restype = ctypes.c_bool
        if voxelWidth is None:
            voxelWidth = 0.0
        if voxelHeight is None:
            voxelHeight = voxelWidth
        if offsetX is None:
            offsetX = 0.0
        if offsetY is None:
            offsetY = 0.0
        if offsetZ is None:
            offsetZ = 0.0
        return self.libprojectors.set_volume(numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ)
        
    def set_default_volume(self,scale=1.0):
        self.libprojectors.set_default_volume.argtypes = [ctypes.c_float]
        self.libprojectors.set_default_volume.restype = ctypes.c_bool
        return self.libprojectors.set_default_volume(scale)
    
    def set_defaultVolume(self,scale=1.0):
        return self.set_default_volume(scale)
    
    def set_volumeDimensionOrder(self,which):
        '''
        Sets the order of the dimensions of the volume
        setVolumeDimensionOrder(0) set the order to XYZ
        setVolumeDimensionOrder(1) set the order to ZYX (this is the default value)
        '''
        self.libprojectors.set_volumeDimensionOrder.argtypes = [ctypes.c_int]
        self.libprojectors.set_volumeDimensionOrder.restype = ctypes.c_bool
        return self.libprojectors.set_volumeDimensionOrder(which)
        
    def get_volumeDimensionOrder(self):
        return self.libprojectors.get_volumeDimensionOrder()
        
    ###################################################################################################################
    ###################################################################################################################
    # THIS SECTION OF FUNCTIONS PROVIDE CONVENIENT ROUTINES TO MAKE THE PROJECTION DATA AND VOLUME DATA NUMPY ARRAYS
    ###################################################################################################################
    ###################################################################################################################
    def allocateProjections(self,val=0.0):
        N_phis = self.get_numAngles()
        N_rows = self.get_numRows()
        N_cols = self.get_numCols()
        if N_phis > 0 and N_rows > 0 and N_cols > 0:
            if val == 0.0:
                return np.ascontiguousarray(np.zeros((N_phis,N_rows,N_cols),dtype=np.float32), dtype=np.float32)
            else:
                return np.ascontiguousarray(val*np.ones((N_phis,N_rows,N_cols),dtype=np.float32), dtype=np.float32)
        else:
            return None
        
    def allocateVolume(self,val=0.0):
        N_x = self.get_numX()
        N_y = self.get_numY()
        N_z = self.get_numZ()
        if N_x > 0 and N_y > 0 and N_z > 0:
            if self.get_volumeDimensionOrder() == 0:
                if val == 0.0:
                    return np.ascontiguousarray(np.zeros((N_x,N_y,N_z),dtype=np.float32), dtype=np.float32)
                else:
                    return np.ascontiguousarray(val*np.ones((N_x,N_y,N_z),dtype=np.float32), dtype=np.float32)
            else:
                if val == 0.0:
                    return np.ascontiguousarray(np.zeros((N_z,N_y,N_x),dtype=np.float32), dtype=np.float32)
                else:
                    return np.ascontiguousarray(val*np.ones((N_z,N_y,N_x),dtype=np.float32), dtype=np.float32)
        else:
            return None
            
    def setAngleArray(self,numAngles,angularRange):
        return np.array(range(numAngles)).astype(np.float32) * angularRange/float(numAngles)
    
    ###################################################################################################################
    ###################################################################################################################
    # THIS SECTION OF FUNCTIONS EXECUTE THE MAIN CPU/GPU ROUTINES IN LEAP
    ###################################################################################################################
    ###################################################################################################################
    def project(self,g,f):
        ''' Calculate the forward projection of f and stores the result in g '''
        self.libprojectors.project.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
        self.libprojectors.project.restype = ctypes.c_bool
        self.libprojectors.project(g,f,True)
        return g
        
    def filterProjections(self, g):
        ''' Filters the projection data, g, so that its backprojection results in an FBP reconstruction '''
        self.libprojectors.filterProjections.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
        self.libprojectors.filterProjections.restype = ctypes.c_bool
        self.libprojectors.filterProjections(g, True)
        return g
        
    def rampFilterProjections(self, g):
        ''' Applies the ramp filter to the projection data which is a subset of the operations in the filterProjections function '''
        self.libprojectors.rampFilterProjections.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool, ctypes.c_float]
        self.libprojectors.rampFilterProjections.restype = ctypes.c_bool
        self.libprojectors.rampFilterProjections(g,True,1.0)
        return g
        
    def HilbertFilterProjections(self, g):
        ''' Applies the Hilbert filter to the projection data which is a subset of the operations in the filterProjections function '''
        self.libprojectors.HilbertFilterProjections.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool, ctypes.c_float]
        self.libprojectors.HilbertFilterProjections.restype = ctypes.c_bool
        self.libprojectors.HilbertFilterProjections(g,True,1.0)
        return g
    
    def backproject(self,g,f):
        ''' Calculate the backprojection (adjoint of the forward projection) of g and stores the result in f '''
        self.libprojectors.backproject.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
        self.libprojectors.backproject.restype = ctypes.c_bool
        self.libprojectors.backproject(g,f,True)
        return f
        
    def weightedBackproject(self,g,f):
        ''' Calculate the weighted backprojection (adjoint of the forward projection) of g and stores the result in f '''
        self.libprojectors.weightedBackproject.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
        self.libprojectors.weightedBackproject.restype = ctypes.c_bool
        self.libprojectors.weightedBackproject(g,f,True)
        return f
        
    def rampFilterVolume(self, f):
        ''' Applies the 2D ramp filter to the volume data, f, for each z-slice '''
        self.libprojectors.rampFilterVolume.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
        self.libprojectors.rampFilterVolume.restype = ctypes.c_bool
        self.libprojectors.rampFilterVolume(f,True)
        return f

    def get_FBPscalar(self):
        ''' Returns the scalar necessary for quantitative reconstruction when using the filterProjections and backproject functions '''
        self.libprojectors.get_FBPscalar.argtypes = []
        self.libprojectors.get_FBPscalar.restype = ctypes.c_float
        return self.libprojectors.get_FBPscalar()

    def FBP(self, g, f, inplace=False):
        ''' Performs a Filtered Backprojection (FBP) reconstruction of the projection data, g, and stores the result in f '''
        self.libprojectors.FBP.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
        self.libprojectors.FBP.restype = ctypes.c_bool
        if inplace == False and self.get_GPU() < 0:
            q  = g.copy()
            self.libprojectors.FBP(q,f,True)
        else:
            self.libprojectors.FBP(g,f,True)
        return f
        
    def BPF(self, g, f):
        ''' Performs a Backprojection Filtration (BPF) reconstruction of the projection data, g, and stores the result in f '''
        self.backproject(g,f)
        self.rampFilterVolume(f)
        f *= self.get_FBPscalar()
        return f
        
    def sensitivity(self, f=None):
        if f is None:
            f = self.allocateVolume()
        #bool sensitivity(float* f, bool cpu_to_gpu);
        self.libprojectors.sensitivity.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
        self.libprojectors.sensitivity.restype = ctypes.c_bool
        self.libprojectors.sensitivity(f,True)
        return f
    
    def rowRangeNeededForBackprojection(self):
        rowsNeeded = np.zeros((2,1),dtype=np.float32)
        rowsNeeded[1] = self.get_numRows()-1
        self.libprojectors.rowRangeNeededForBackprojection.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
        self.libprojectors.rowRangeNeededForBackprojection.restype = ctypes.c_bool
        self.libprojectors.rowRangeNeededForBackprojection(rowsNeeded)
        return rowsNeeded
    
    ###################################################################################################################
    ###################################################################################################################
    # THIS SECTION PROVIDES ITERATIVE RECONSTRUCTION ALGORITHM THAT USE THE LEAP FORWARD AND BACKPROJECTION OPERATIONS
    ###################################################################################################################
    ###################################################################################################################
    def MLEM(self, g, f, numIter):
        if not np.any(f):
            f[:] = 1.0
        else:
            f[f<0.0] = 0.0
 
        Pstar1 = self.sensitivity()
        Pstar1[Pstar1==0.0] = 1.0
        d = self.allocateVolume()
        Pd = self.allocateProjections()
        
        for n in range(numIter):
            print('MLEM iteration ' + str(n+1) + ' of ' + str(numIter))
            self.project(Pd,f)
            ind = Pd != 0.0
            Pd[ind] = g[ind]/Pd[ind]
            self.backproject(Pd,d)
            f *= d/Pstar1
        return f
    
    def OSEM(self, g, f, numIter, numSubsets=1):
        if not np.any(f):
            f[:] = 1.0
        else:
            f[f<0.0] = 0.0
 
        numSubsets = min(numSubsets, self.get_numAngles())
        if self.get_geometry() == 'MODULAR' and numSubsets > 1:
            print('WARNING: Subsets not yet implemented for modular-beam geometry, setting to 1.')
            numSubsets = 1
        if numSubsets <= 1:
            return self.MLEM(g,f,numIter)
        else:
        
            phis = self.get_angles()
            
            # divide g and phis
            phis_subsets = []
            g_subsets = []
            for m in range(numSubsets):
                phis_subset = np.ascontiguousarray(phis[m:-1:numSubsets], np.float32)
                phis_subsets.append(phis_subset)
                
                g_subset = np.ascontiguousarray(g[m:-1:numSubsets,:,:], np.float32)
                g_subsets.append(g_subset)
                
            d = self.allocateVolume()
            for n in range(numIter):
                print('MLEM iteration ' + str(n+1) + ' of ' + str(numIter))
                for m in range(numSubsets):
                
                    # set angle array
                    self.set_angles(phis_subsets[m])
                    
                    Pstar1 = self.sensitivity()
                    #Pstar1[Pstar1=0.0] = 1.0

                    Pd = self.allocateProjections()
                    self.project(Pd,f)
                    ind = Pd != 0.0
                    Pd[ind] = g_subsets[m][ind]/Pd[ind]
                    self.backproject(Pd,d)
                    f *= d/Pstar1
            return f
        
    def SART(self, g, f, numIter, numSubsets=1):
        numSubsets = min(numSubsets, self.get_numAngles())
        if self.get_geometry() == 'MODULAR' and numSubsets > 1:
            print('WARNING: Subsets not yet implemented for modular-beam geometry, setting to 1.')
            numSubsets = 1
        if numSubsets <= 1:
            P1 = self.allocateProjections()
            self.project(P1,self.allocateVolume(1.0))
            P1[P1==0.0] = 1.0
            
            Pstar1 = self.sensitivity()
            Pstar1[Pstar1==0.0] = 1.0
            
            Pd = self.allocateProjections()
            d = self.allocateVolume()

            for n in range(numIter):
                print('SART iteration ' + str(n+1) + ' of ' + str(numIter))
                self.project(Pd,f)
                Pd = (g-Pd) / P1
                self.backproject(Pd,d)
                f += 0.9*d / Pstar1
                f[f<0.0] = 0.0
            return f
        else:
            P1 = self.allocateProjections()
            self.project(P1,self.allocateVolume(1.0))
            P1[P1==0.0] = 1.0
            
            phis = self.get_angles()
            
            # divide g, P1, and phis
            phis_subsets = []
            g_subsets = []
            P1_subsets = []
            for m in range(numSubsets):
                phis_subset = np.ascontiguousarray(phis[m:-1:numSubsets], np.float32)
                phis_subsets.append(phis_subset)
                
                g_subset = np.ascontiguousarray(g[m:-1:numSubsets,:,:], np.float32)
                g_subsets.append(g_subset)
                
                P1_subset = np.ascontiguousarray(P1[m:-1:numSubsets,:,:], np.float32)
                P1_subsets.append(P1_subset)
            
            d = self.allocateVolume()
            for n in range(numIter):
                print('SART iteration ' + str(n+1) + ' of ' + str(numIter))
                for m in range(numSubsets):
                
                    # set angle array
                    self.set_angles(phis_subsets[m])
                    
                    Pstar1 = self.sensitivity()
                    #Pstar1[Pstar1==0.0] = 1.0

                    Pd = self.allocateProjections()
                    self.project(Pd,f)
                    Pd = (g_subsets[m]-Pd) / P1_subsets[m]
                    self.backproject(Pd,d)
                    f += 0.9*d / Pstar1
                    f[f<0.0] = 0.0
            return f
            
    def ASDPOCS(self, g, f, numIter, numSubsets, numTV, delta=0.0):
        if numTV <= 0:
            return self.SART(g,f,numIter,numSubsets)
        numSubsets = min(numSubsets, self.get_numAngles())
        if self.get_geometry() == 'MODULAR' and numSubsets > 1:
            print('WARNING: Subsets not yet implemented for modular-beam geometry, setting to 1.')
            numSubsets = 1
        omega = 0.8
        P1 = self.allocateProjections()
        self.project(P1,self.allocateVolume(1.0))
        P1[P1==0.0] = 1.0

        phis_subsets = []
        g_subsets = []
        P1_subsets = []
        phis = self.get_angles()
        if numSubsets > 1:
            # divide g, P1, and phis
            for m in range(numSubsets):
                phis_subset = np.ascontiguousarray(phis[m:-1:numSubsets], np.float32)
                phis_subsets.append(phis_subset)
                
                g_subset = np.ascontiguousarray(g[m:-1:numSubsets,:,:], np.float32)
                g_subsets.append(g_subset)
                
                P1_subset = np.ascontiguousarray(P1[m:-1:numSubsets,:,:], np.float32)
                P1_subsets.append(P1_subset)
        else:
            Pstar1 = self.sensitivity()
            Pstar1[Pstar1==0.0] = 1.0
        
        #Pd = self.allocateProjections()
        Pf_minus_g = self.allocateProjections()
        Pf_TV_minus_g = self.allocateProjections()
        d = self.allocateVolume()
        f_TV = self.allocateVolume()

        self.project(Pf_minus_g, f)
        Pf_minus_g -= g
        
        curCost = np.sum(Pf_minus_g**2)
        
        for n in range(numIter):
            print('ASDPOCS iteration ' + str(n+1) + ' of ' + str(numIter))
            
            # SART Update
            if numSubsets <= 1:
                Pf_minus_g = Pf_minus_g / P1
                self.backproject(Pf_minus_g,d)
                f -= 0.9*d / Pstar1
                f[f<0.0] = 0.0
            else:
                for m in range(numSubsets):
                    self.set_angles(phis_subsets[m])
                    Pstar1 = self.sensitivity()
                    #Pstar1[Pstar1==0.0] = 1.0

                    Pd = self.allocateProjections()
                    self.project(Pd,f)
                    Pd = (g_subsets[m]-Pd) / P1_subsets[m]
                    self.backproject(Pd,d)
                    f += 0.9*d / Pstar1
                    f[f<0.0] = 0.0
                self.set_angles(phis)

            # Calculate SART error sinogram and calculate cost            
            self.project(Pf_minus_g, f)
            Pf_minus_g = Pf_minus_g - g
            epsilon_SART = np.sum(Pf_minus_g**2)

            #'''            
            # TV step(s)
            f_TV[:] = f[:]
            self.diffuse(f_TV, delta, numTV)
            #self.displayVolume(f_TV)
            self.project(Pf_TV_minus_g, f_TV)
            Pf_TV_minus_g = Pf_TV_minus_g - g
            epsilon_TV = np.sum(Pf_TV_minus_g**2)
            
            # Combine SART and TV Steps
            temp = np.sum(Pf_minus_g*Pf_TV_minus_g)
            a = epsilon_SART - 2.0 * temp + epsilon_TV
            b = temp - epsilon_SART
            c = epsilon_SART - ((1.0 - omega) * epsilon_SART + omega * curCost)
            disc = b * b - a * c
            alpha = 0.0
            alpha_1 = 0.0
            alpha_2 = 0.0
            if disc < 0.0:
                print("  Unable to estimate step size, setting lambda to zero!")
                alpha = 0.0
            else:
                if a != 0.0:
                    alpha_1 = (-b + np.sqrt(disc)) / a
                    alpha_2 = (-b - np.sqrt(disc)) / a
                if a == 0.0:
                    alpha = 1.0
                elif a > 0.0:
                    alpha = alpha_1
                else:
                    alpha = alpha_2
            alpha = min(1.0, alpha)
            alpha = max(0.0, min(1.0, alpha))  # force lambda to be a valid number
            print('  lambda = ' + str(np.round(1000.0*alpha)/1000.0) + ' (' + str(np.round(1000.0*alpha_1)/1000.0) + ', ' + str(np.round(1000.0*alpha_2)/1000.0) + ')')
            if alpha < 0.0:
                print("  Stopping criteria met, stopping iterations.")
                break
                
                
            # Do Update
            f[:] = (1.0-alpha)*f[:] + alpha*f_TV[:]
            Pf_minus_g[:] = (1.0-alpha)*Pf_minus_g[:] + alpha*Pf_TV_minus_g[:]
            curCost = np.sum(Pf_minus_g**2)
            #'''
                        
        return f
        
        
    def LS(self, g, f, numIter, SQS=False):
        return self.RWLS(g, f, numIter, 0.0, 0.0, 1.0, SQS)
        
    def WLS(self, g, f, numIter, W=None, SQS=False):
        return self.RWLS(g, f, numIter, 0.0, 0.0, W, SQS)
        
    def RLS(self, g, f, numIter, delta=0.0, beta=0.0, SQS=False):
        return self.RWLS(g, f, numIter, delta, beta, 1.0, SQS)
        
    def RWLS(self, g, f, numIter, delta=0.0, beta=0.0, W=None, SQS=False):
        conjGradRestart = 50
        if W is None:
            W = g.copy()
            W = np.exp(-W)
        if f is None:
            f = self.allocateVolume()
        Pf = g.copy()
        if np.any(f):
            # fix scaling
            f[f<0.0] = 0.0
            self.project(Pf,f)
            Pf_dot_Pf = np.sum(Pf**2)
            g_dot_Pf = np.sum(g*Pf)
            if Pf_dot_Pf > 0.0 and g_dot_Pf > 0.0:
                f *= g_dot_Pf / Pf_dot_Pf
                Pf *= g_dot_Pf / Pf_dot_Pf
        else:
            Pf[:] = 0.0
        Pf_minus_g = Pf
        Pf_minus_g -= g
        
        grad = self.allocateVolume()
        u = self.allocateVolume()
        Pu = self.allocateProjections()
        
        d = self.allocateVolume()
        Pd = self.allocateProjections()
        
        grad_old_dot_grad_old = 0.0
        grad_old = self.allocateVolume()
        
        if SQS == True:
            # Calculate the SQS preconditioner
            # Reuse some of the memory allocated above
            #Q = 1.0 / P*WP1
            Q = self.allocateVolume()
            Q[:] = 1.0
            self.project(Pu,Q)
            Pu *= W
            self.backproject(Pu,Q)
            Q = 1.0 / Q
        else:
            Q = 1.0
        
        for n in range(numIter):
            print('RWLS iteration ' + str(n+1) + ' of ' + str(numIter))
            WPf_minus_g = Pf_minus_g.copy()
            if W is not None:
                WPf_minus_g *= W
            self.backproject(WPf_minus_g, grad)
            if beta > 0.0:
                Sf1 = self.TVgradient(f, delta, beta)
                grad += Sf1

                #f[:] = grad[:] # FIXME
                #return f # FIXME
                
            u[:] = grad[:]
            u = Q*u
            self.project(Pu, u)
            
            if n == 0 or (n % conjGradRestart) == 0:
                d[:] = u[:]
                Pd[:] = Pu[:]
            else:
                gamma = (np.sum(u*grad) - np.sum(u*grad_old)) / grad_old_dot_grad_old

                d = u + gamma*d
                Pd = Pu + gamma*Pd

                if np.sum(d*grad) <= 0.0:
                    print('\tRLWS-CG: CG descent condition violated, must use GD descent direction')
                    d[:] = u[:]
                    Pd[:] = Pu[:]
            
            grad_old_dot_grad_old = np.sum(u*grad)
            grad_old[:] = grad[:]
            
            stepSize = self.RWLSstepSize(f, grad, d, Pd, W, delta, beta)
            if stepSize <= 0.0:
                print('invalid step size; quitting!')
                break
            
            f -= stepSize*d
            f[f<0.0] = 0.0
            self.project(Pf,f)
            Pf_minus_g = Pf-g
        return f

    def RWLSstepSize(self, f, grad, d, Pd, W, delta, beta):
        num = np.sum(d*grad)
        if W is not None:
            denomA = np.sum(Pd*Pd*W)
        else:
            denomA = np.sum(Pd**2)
        denomB = 0.0;
        if beta > 0.0:
            denomB = self.TVquadForm(f, d, delta, beta)
            #print('denomB = ' + str(denomA))
        denom = denomA + denomB

        stepSize = 0.0
        if np.abs(denom) > 1.0e-16:
            stepSize = num / denom
        print('\tlambda = ' + str(stepSize))
        return stepSize


    ###################################################################################################################
    ###################################################################################################################
    # THIS SECTION OF FUNCTIONS EXECUTE LEAP'S GPU DENOISING FILTERS
    ###################################################################################################################
    ###################################################################################################################
    def BlurFilter(self, f, FWHM=2.0):
        #bool BlurFilter(float* f, int, int, int, float FWHM);
        self.libprojectors.BlurFilter.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_bool]
        self.libprojectors.BlurFilter.restype = ctypes.c_bool
        return self.libprojectors.BlurFilter(f, f.shape[0], f.shape[1], f.shape[2], FWHM, True)
    
    def MedianFilter(self, f, threshold=0.0):
        #bool MedianFilter(float* f, int, int, int, float threshold);
        self.libprojectors.MedianFilter.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_bool]
        self.libprojectors.MedianFilter.restype = ctypes.c_bool
        return self.libprojectors.MedianFilter(f, f.shape[0], f.shape[1], f.shape[2], threshold, True)
    
    def TVcost(self, f, delta, beta=0.0):
        #float TVcost(float* f, int N_1, int N_2, int N_3, float delta, float beta);
        self.libprojectors.TVcost.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_bool]
        self.libprojectors.TVcost.restype = ctypes.c_float
        
        return self.libprojectors.TVcost(f, f.shape[0], f.shape[1], f.shape[2], delta, beta, True)
        
    def TVgradient(self, f, delta, beta=0.0):
        #bool TVgradient(float* f, float* Df, int N_1, int N_2, int N_3, float delta, float beta);
        self.libprojectors.TVgradient.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_bool]
        self.libprojectors.TVgradient.restype = ctypes.c_bool
        
        Df = np.ascontiguousarray(np.zeros(f.shape).astype(np.float32), dtype=np.float32)
        self.libprojectors.TVgradient(f, Df, f.shape[0], f.shape[1], f.shape[2], delta, beta, True)
        return Df
    
    def TVquadForm(self, f, d, delta, beta=0.0):
        #float TVquadForm(float* f, float* d, int N_1, int N_2, int N_3, float delta, float beta);
        self.libprojectors.TVquadForm.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_bool]
        self.libprojectors.TVquadForm.restype = ctypes.c_float
        
        return self.libprojectors.TVquadForm(f, d, f.shape[0], f.shape[1], f.shape[2], delta, beta, True)
        
    def diffuse(self, f, delta, numIter):
        self.libprojectors.Diffuse.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_bool]
        self.libprojectors.Diffuse.restype = ctypes.c_bool
        self.libprojectors.Diffuse(f, f.shape[0], f.shape[1], f.shape[2], delta, numIter, True)
        return f
        '''
        for n in range(N):
            d = self.TVgradient(f, delta)
            num = np.sum(d**2)
            denom = self.TVquadForm(f, d, delta)
            if denom <= 1.0e-16:
                break
            stepSize = num / denom
            f -= stepSize * d
        return f
        '''

    ###################################################################################################################
    ###################################################################################################################
    # THIS SECTION OF FUNCTIONS SET AND GET VARIOUS PARAMETERS, INCLUDING THOSE THAT SET HOW LEAP IS TO BE RUN
    ###################################################################################################################
    ###################################################################################################################
    def set_gpu(self, which):
        return self.set_GPU(which)
    
    def set_GPU(self, which):
        self.libprojectors.set_GPU.argtypes = [ctypes.c_int]
        self.libprojectors.set_GPU.restype = ctypes.c_bool
        return self.libprojectors.set_GPU(which)
        
    def set_gpus(self, listOfGPUs):
        return self.set_GPUs(listOfGPUs)
        
    def set_GPUs(self, listOfGPUs):
        self.libprojectors.set_GPUs.argtypes = [ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_int]
        self.libprojectors.set_GPUs.restype = ctypes.c_bool
        listOfGPUs = np.ascontiguousarray(listOfGPUs, dtype=np.int32)
        return self.libprojectors.set_GPUs(listOfGPUs, int(listOfGPUs.size))
        
    def get_GPU(self):
        self.libprojectors.get_GPU.restype = ctypes.c_int
        return self.libprojectors.get_GPU()
        
    def set_diameterFOV(self, d):
        self.libprojectors.set_rFOV.argtypes = [ctypes.c_float]
        self.libprojectors.set_rFOV.restype = ctypes.c_bool
        return self.libprojectors.set_rFOV(0.5*d)
    
    def set_axisOfSymmetry(self,val):
        self.libprojectors.set_axisOfSymmetry.argtypes = [ctypes.c_float]
        self.libprojectors.set_axisOfSymmetry.restype = ctypes.c_bool
        return self.libprojectors.set_axisOfSymmetry(val)
        
    def clear_axisOfSymmetry(self):
        self.libprojectors.clear_axisOfSymmetry.argtypes = []
        self.libprojectors.clear_axisOfSymmetry.restype = ctypes.c_bool
        return self.libprojectors.clear_axisOfSymmetry()
        
    def set_projector(self,which):
        self.libprojectors.set_projector.argtypes = [ctypes.c_int]
        self.libprojectors.set_projector.restype = ctypes.c_bool
        return self.libprojectors.set_projector(which)
        
    def set_rampFilter(self,which):
        self.libprojectors.set_rampID.argtypes = [ctypes.c_int]
        self.libprojectors.set_rampID.restype = ctypes.c_bool
        return self.libprojectors.set_rampID(which)
        
    def set_attenuationMap(self, mu):
        self.libprojectors.set_attenuationMap.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
        self.libprojectors.set_attenuationMap.restype = ctypes.c_bool
        return self.libprojectors.set_attenuationMap(mu)
        
    def set_cylindircalAttenuationMap(self, c, R):
        self.libprojectors.set_cylindircalAttenuationMap.argtypes = [ctypes.c_float, ctypes.c_float]
        self.libprojectors.set_cylindircalAttenuationMap.restype = ctypes.c_bool
        return self.libprojectors.set_cylindircalAttenuationMap(c, R)
        
    def clear_attenuationMap(self):
        self.libprojectors.clear_attenuationMap.restype = ctypes.c_bool
        return self.libprojectors.clear_attenuationMap()
        
    def get_angles(self):
        phis = np.ascontiguousarray(np.zeros(self.get_numAngles()).astype(np.float32), dtype=np.float32)
        self.libprojectors.get_angles.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
        self.libprojectors.get_angles.restype = ctypes.c_bool
        self.libprojectors.get_angles(phis)
        return phis
        
    def set_angles(self,phis):
        self.libprojectors.set_angles.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int]
        self.libprojectors.set_angles.restype = ctypes.c_bool
        return self.libprojectors.set_angles(phis, int(phis.size))

    ###################################################################################################################
    ###################################################################################################################
    # THIS SECTION RETRIEVES THE VARIOUS CT GEOMETRY AND VOLUME PARAMETERS THAT HAS BEEN SET IN LEAP
    ###################################################################################################################
    ###################################################################################################################
    def get_geometry(self):
        self.libprojectors.get_geometry.restype = ctypes.c_int
        geometryType = self.libprojectors.get_geometry()
        if geometryType == 0:
            return 'CONE'
        elif geometryType == 1:
            return 'PARALLEL'
        elif geometryType == 2:
            return 'FAN'
        elif geometryType == 3:
            return 'MODULAR'
        else:
            return 'UNKNOWN'
            
    def get_sod(self):
        self.libprojectors.get_sod.restype = ctypes.c_float
        return self.libprojectors.get_sod()
        
    def get_sdd(self):
        self.libprojectors.get_sdd.restype = ctypes.c_float
        return self.libprojectors.get_sdd()
        
    def get_helicalPitch(self):
        self.libprojectors.get_helicalPitch.restype = ctypes.c_float
        return self.libprojectors.get_helicalPitch()
        
    def get_z_source_offset(self):
        self.libprojectors.get_z_source_offset.restype = ctypes.c_float
        return self.libprojectors.get_z_source_offset()
    
    def get_numAngles(self):
        return self.libprojectors.get_numAngles()
        
    def get_numRows(self):
        return self.libprojectors.get_numRows()
        
    def get_numCols(self):
        return self.libprojectors.get_numCols()
        
    def get_pixelHeight(self):
        self.libprojectors.get_pixelHeight.restype = ctypes.c_float
        return self.libprojectors.get_pixelHeight()
        
    def get_pixelWidth(self):
        self.libprojectors.get_pixelWidth.restype = ctypes.c_float
        return self.libprojectors.get_pixelWidth()
        
    def get_centerRow(self):
        self.libprojectors.get_centerRow.restype = ctypes.c_float
        return self.libprojectors.get_centerRow()
        
    def get_centerCol(self):
        self.libprojectors.get_centerCol.restype = ctypes.c_float
        return self.libprojectors.get_centerCol()
        
    def get_tau(self):
        self.libprojectors.get_tau.restype = ctypes.c_float
        return self.libprojectors.get_tau()
        
    def get_sourcePositions(self):
        #bool get_sourcePositions(float*);
        if self.get_numAngles() <= 0:
            return None
        else:
            x = np.ascontiguousarray(np.zeros((self.get_numAngles(),3)).astype(np.float32), dtype=np.float32)
            self.libprojectors.get_sourcePositions.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
            self.libprojectors.get_sourcePositions.restype = ctypes.c_bool
            self.libprojectors.get_sourcePositions(x)
            return x
        
    def get_moduleCenters(self):
	    #bool get_moduleCenters(float*);
        if self.get_numAngles() <= 0:
            return None
        else:
            x = np.ascontiguousarray(np.zeros((self.get_numAngles(),3)).astype(np.float32), dtype=np.float32)
            self.libprojectors.get_moduleCenters.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
            self.libprojectors.get_moduleCenters.restype = ctypes.c_bool
            self.libprojectors.get_moduleCenters(x)
            return x
        
    def get_rowVectors(self):
	    #bool get_rowVectors(float*);
        if self.get_numAngles() <= 0:
            return None
        else:
            x = np.ascontiguousarray(np.zeros((self.get_numAngles(),3)).astype(np.float32), dtype=np.float32)
            self.libprojectors.get_rowVectors.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
            self.libprojectors.get_rowVectors.restype = ctypes.c_bool
            self.libprojectors.get_rowVectors(x)
            return x
        
    def get_colVectors(self):
	    #bool get_colVectors(float*);
        if self.get_numAngles() <= 0:
            return None
        else:
            x = np.ascontiguousarray(np.zeros((self.get_numAngles(),3)).astype(np.float32), dtype=np.float32)
            self.libprojectors.get_colVectors.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
            self.libprojectors.get_colVectors.restype = ctypes.c_bool
            self.libprojectors.get_colVectors(x)
            return x
        
    def get_numX(self):
        return self.libprojectors.get_numX()
    
    def get_numY(self):
        return self.libprojectors.get_numY()
    
    def get_numZ(self):
        return self.libprojectors.get_numZ()
        
    def get_voxelHeight(self):
        self.libprojectors.get_voxelHeight.restype = ctypes.c_float
        return self.libprojectors.get_voxelHeight()
        
    def get_voxelWidth(self):
        self.libprojectors.get_voxelWidth.restype = ctypes.c_float
        return self.libprojectors.get_voxelWidth()
        
    def get_offsetX(self):
        self.libprojectors.get_offsetX.restype = ctypes.c_float
        return self.libprojectors.get_offsetX()
        
    def get_offsetY(self):
        self.libprojectors.get_offsetY.restype = ctypes.c_float
        return self.libprojectors.get_offsetY()
        
    def get_offsetZ(self):
        self.libprojectors.get_offsetZ.restype = ctypes.c_float
        return self.libprojectors.get_offsetZ()
        
    def get_z0(self):
        self.libprojectors.get_z0.restype = ctypes.c_float
        return self.libprojectors.get_z0()

    
    ###################################################################################################################
    ###################################################################################################################
    # UTILITY FUNCTIONS
    ###################################################################################################################
    ###################################################################################################################
    def x_samples(self,centerCoords=False):
        if centerCoords:
            x_0 = -0.5*(self.get_numX()-1)*self.get_voxelWidth()
        else:
            x_0 = self.get_offsetX() - 0.5*(self.get_numX()-1)*self.get_voxelWidth()
        return np.array(range(self.get_numX()),dtype=np.float32)*self.get_voxelWidth() + x_0
        
    def y_samples(self,centerCoords=False):
        if centerCoords:
            y_0 = -0.5*(self.get_numY()-1)*self.get_voxelWidth()
        else:
            y_0 = self.get_offsetY() - 0.5*(self.get_numY()-1)*self.get_voxelWidth()
        return np.array(range(self.get_numY()),dtype=np.float32)*self.get_voxelWidth() + y_0
        
    def z_samples(self,centerCoords=False):
        if centerCoords:
            z_0 = -0.5*(self.get_numZ()-1)*self.get_voxelHeight()
        else:
            z_0 = self.get_z0()
        return np.array(range(self.get_numZ()),dtype=np.float32)*self.get_voxelHeight() + z_0
    
    def voxelSamples(self,centerCoords=False):
        x = self.x_samples(centerCoords)
        y = self.y_samples(centerCoords)
        z = self.z_samples(centerCoords)
        if self.get_volumeDimensionOrder() == 0:
            x,y,z = np.meshgrid(x,y,z, indexing='ij')
        else:
            z,y,x = np.meshgrid(z,y,x, indexing='ij')
        return x,y,z

    def display(self,vol):
        self.displayVolume(vol)

    def display_volume(self,vol):
        self.displayVolume(vol)

    def displayVolume(self,vol):
        try:
            import napari
            if len(vol.shape) == 3 and (vol.shape[0] == 1 or vol.shape[1] == 1 or vol.shape[2] == 1):
                viewer = napari.view_image(np.squeeze(vol), rgb=False)
            else:
                viewer = napari.view_image(vol, rgb=False)
            napari.run()
        except:
            print('Cannot load napari, to install run this command:')
            print('pip install napari[all]')
            
    def sketchSystem(self,whichView=None):
        self.sketchSystem(whichView)
            
    def sketch_system(self,whichView=None):
    
        if self.get_numAngles() <= 0 or self.get_numRows() <= 0 or self.get_numCols() <= 0:
            print('CT geometry not set!')
            return False
    
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        self.drawCT(ax,whichView)
        self.drawVolume(ax)
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])
        print(plot_radius)

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
        
        ax.view_init(90, -90)
        #ax.figure.set_size_inches(8, 8)
        plt.show()
    
    def drawCT(self, ax, whichView=None):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
        import matplotlib.pyplot as plt
        
        pixelWidth = self.get_pixelWidth()
        pixelHeight = self.get_pixelHeight()
        numCols = self.get_numCols()
        numRows = self.get_numRows()
        centerRow = self.get_centerRow()
        centerCol = self.get_centerCol()
        
        geometryText = self.get_geometry()
        
        if geometryText == 'CONE' or geometryText == 'FAN':
            sod = self.get_sod()
            sdd = self.get_sdd()
            tau = self.get_tau()
            #R = np.sqrt(sod*sod - tau*tau)
            #D = np.sqrt(sdd*sdd - tau*tau)
            R = sod
            D = sdd
            odd = D-R
            detectorWidth = numCols*pixelWidth
            detectorHeight = numRows*pixelHeight
            detectorLeft = -centerCol*pixelWidth + tau
            detectorRight = (numCols-1-centerCol)*pixelWidth + tau
            detectorBottom = -centerRow*pixelHeight
            detectorTop = (numRows-1-centerRow)*pixelHeight
            xs = [detectorLeft, detectorRight, detectorRight, detectorLeft, detectorLeft]
            ys = [-odd, -odd, -odd, -odd, -odd]
            zs = [detectorBottom, detectorBottom, detectorTop, detectorTop, detectorBottom]
            ax.plot(xs,ys,zs,color='black')

            if geometryText == 'CONE':
                ax.plot([tau, tau], [R, -odd], [0, 0], color='green') # pxcenter line
                #ax.plot([tau, tau-tau*(R+odd)/R], [R, -odd], [0, 0], color='green') # pxmidoff line
                ax.plot([tau, detectorLeft, tau, detectorRight, tau, detectorLeft, tau, detectorRight], [R, -odd, R, -odd, R, -odd, R, -odd], [0, detectorBottom, 0, detectorBottom, 0, detectorTop, 0, detectorTop],color='red')
            else:
                ax.plot([tau, tau], [R, -odd], [0, 0], color='green')  # pxcenter line
                #ax.plot([tau, tau - tau * (R + odd) / R], [R, -odd], [0, 0], color='green')  # pxmidoff line
                ax.plot([tau, detectorLeft, detectorRight, tau], [R, -odd, -odd, R], [detectorBottom, detectorBottom, detectorBottom, detectorBottom], color='red')
                ax.plot([tau, detectorLeft, detectorRight, tau], [R, -odd, -odd, R], [0, 0, 0, 0], color='red')
                ax.plot([tau, detectorLeft, detectorRight, tau], [R, -odd, -odd, R], [detectorTop, detectorTop, detectorTop, detectorTop], color='red')

            topLeft = np.array([detectorLeft, ys[0], detectorTop])
            topRight = np.array([detectorRight, ys[1], detectorTop])
            bottomLeft = np.array([detectorLeft, ys[2], detectorBottom])
            bottomRight = np.array([detectorRight, ys[3], detectorBottom])
            Z = np.squeeze(np.array([topLeft.tolist(), topRight.tolist(), bottomRight.tolist(), bottomLeft.tolist(), bottomLeft.tolist(), bottomRight.tolist(), topRight.tolist(), topLeft.tolist()]))
            ax.scatter3D(Z[:,0], Z[:,1], Z[:,2])
            verts = [[Z[0],Z[1],Z[2],Z[3]],
            [Z[4],Z[5],Z[6],Z[7]],
            [Z[1],Z[2],Z[5],Z[6]],
            [Z[0],Z[3],Z[4],Z[7]],
            [Z[0],Z[1],Z[6],Z[7]],
            [Z[2],Z[3],Z[4],Z[5]]]
            ax.add_collection3d(Poly3DCollection(verts, facecolors='black', linewidths=1, edgecolors='k', alpha=.20))

        elif geometryText == 'PARALLEL':
            tau = 0.0
            detectorWidth = numCols*pixelWidth
            detectorHeight = numRows*pixelHeight
            detectorLeft = -centerCol*pixelWidth + tau
            detectorRight = (numCols-1-centerCol)*pixelWidth + tau
            detectorBottom = -centerRow*pixelHeight
            detectorTop = (numRows-1-centerRow)*pixelHeight

            sod = detectorWidth
            sdd = 2*sod
            odd = sdd - sod

            xs = [detectorLeft, detectorRight, detectorRight, detectorLeft, detectorLeft]
            ys = [-odd, -odd, -odd, -odd, -odd]
            zs = [detectorBottom, detectorBottom, detectorTop, detectorTop, detectorBottom]
            ax.plot(xs, ys, zs, color='black')

            ax.plot([0, 0], [sod, -odd], [0, 0], color='green')
            ax.plot([detectorLeft, detectorLeft, detectorRight, detectorRight], [sod, -odd, -odd, sod], [detectorBottom, detectorBottom, detectorBottom, detectorBottom], color='red')
            ax.plot([detectorLeft, detectorLeft, detectorRight, detectorRight], [sod, -odd, -odd, sod], [0, 0, 0, 0], color='red')
            ax.plot([detectorLeft, detectorLeft, detectorRight, detectorRight], [sod, -odd, -odd, sod], [detectorTop, detectorTop, detectorTop, detectorTop], color='red')

            topLeft = np.array([detectorLeft, ys[0], detectorTop])
            topRight = np.array([detectorRight, ys[1], detectorTop])
            bottomLeft = np.array([detectorLeft, ys[2], detectorBottom])
            bottomRight = np.array([detectorRight, ys[3], detectorBottom])
            Z = np.squeeze(np.array([topLeft.tolist(), topRight.tolist(), bottomRight.tolist(), bottomLeft.tolist(), bottomLeft.tolist(), bottomRight.tolist(), topRight.tolist(), topLeft.tolist()]))
            ax.scatter3D(Z[:,0], Z[:,1], Z[:,2])
            verts = [[Z[0],Z[1],Z[2],Z[3]],
            [Z[4],Z[5],Z[6],Z[7]],
            [Z[1],Z[2],Z[5],Z[6]],
            [Z[0],Z[3],Z[4],Z[7]],
            [Z[0],Z[1],Z[6],Z[7]],
            [Z[2],Z[3],Z[4],Z[5]]]
            ax.add_collection3d(Poly3DCollection(verts, facecolors='black', linewidths=1, edgecolors='k', alpha=.20))
            
        elif geometryText == 'MODULAR':
            sourcePositions = self.get_sourcePositions()
            moduleCenters = self.get_moduleCenters()
            rowVecs = self.get_rowVectors()
            colVecs = self.get_colVectors()
            
            ind_lo = 0
            ind_hi = self.get_numAngles()-1
            if whichView is not None:
                ind_lo = whichView
                ind_hi = whichView
            
            for n in range(ind_lo,ind_hi+1):
                sourcePos = sourcePositions[n,:]
                detPos = moduleCenters[n,:]
                rowVec = rowVecs[n,:]
                colVec = colVecs[n,:]
                
                rowMin = -0.5*(numRows-1)*pixelHeight*rowVec
                rowMax = 0.5*(numRows-1)*pixelHeight*rowVec
                colMin = -0.5*(numCols-1)*pixelWidth*colVec
                colMax = 0.5*(numCols-1)*pixelWidth*colVec
                
                topLeft = detPos + rowMax + colMin
                topRight = detPos + rowMax + colMax
                bottomLeft = detPos + rowMin + colMin
                bottomRight = detPos + rowMin + colMax
                
                ax.plot([sourcePos[0], topLeft[0]], [sourcePos[1], topLeft[1]], [sourcePos[2], topLeft[2]], color='red')
                ax.plot([sourcePos[0], topRight[0]], [sourcePos[1], topRight[1]], [sourcePos[2], topRight[2]], color='red')
                ax.plot([sourcePos[0], bottomLeft[0]], [sourcePos[1], bottomLeft[1]], [sourcePos[2], bottomLeft[2]], color='red')
                ax.plot([sourcePos[0], bottomRight[0]], [sourcePos[1], bottomRight[1]], [sourcePos[2], bottomRight[2]], color='red')
                
                Z = np.squeeze(np.array([topLeft.tolist(), topRight.tolist(), bottomRight.tolist(), bottomLeft.tolist(), bottomLeft.tolist(), bottomRight.tolist(), topRight.tolist(), topLeft.tolist()]))
                ax.scatter3D(Z[:,0], Z[:,1], Z[:,2])
                verts = [[Z[0],Z[1],Z[2],Z[3]],
                [Z[4],Z[5],Z[6],Z[7]],
                [Z[1],Z[2],Z[5],Z[6]],
                [Z[0],Z[3],Z[4],Z[7]],
                [Z[0],Z[1],Z[6],Z[7]],
                [Z[2],Z[3],Z[4],Z[5]]]
                ax.add_collection3d(Poly3DCollection(verts, facecolors='black', linewidths=1, edgecolors='k', alpha=.20))
                
        if geometryText == 'PARALLEL' or geometryText == 'FAN' or geometryText == 'CONE':
            if geometryText == 'CONE':
                pitch = self.get_helicalPitch()
                z_source_offset = self.get_z_source_offset()
            else:
                pitch = 0.0
                z_source_offset = 0.0
                
            phis = np.pi/180.0*self.get_angles() - 0.5*np.pi
            ax.plot(sod*np.cos(phis), sod*np.sin(phis), (pitch*phis+z_source_offset), '.', color='green')
    
    def drawVolume(self, ax):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
        import matplotlib.pyplot as plt
        N_x = self.get_numX()
        N_y = self.get_numY()
        N_z = self.get_numZ()
        if N_x > 0 and N_y > 0 and N_z > 0:
            x_c = self.get_offsetX()
            y_c = self.get_offsetY()
            z_c = self.get_offsetZ()
            T = self.get_voxelWidth()
            T_z = self.get_voxelHeight()
            x_min = -0.5*float(N_x-1)*T + x_c
            x_max = 0.5*float(N_x-1)*T + x_c
            y_min = -0.5*float(N_y-1)*T + y_c
            y_max = 0.5*float(N_y-1)*T + y_c
            z_min = self.get_z0()
            z_max = float(N_z-1)*T_z + z_min

            topLeft_front = np.array([x_min, y_max, z_max])
            topRight_front = np.array([x_max, y_max, z_max])
            bottomLeft_front = np.array([x_min, y_min, z_max])
            bottomRight_front = np.array([x_max, y_min, z_max])
            topLeft_back = np.array([x_min, y_max, z_min])
            topRight_back = np.array([x_max, y_max, z_min])
            bottomLeft_back = np.array([x_min, y_min, z_min])
            bottomRight_back = np.array([x_max, y_min, z_min])
            Z = np.squeeze(np.array([topLeft_front.tolist(), topRight_front.tolist(), bottomRight_front.tolist(), bottomLeft_front.tolist(), bottomLeft_back.tolist(), bottomRight_back.tolist(), topRight_back.tolist(), topLeft_back.tolist()]))
            ax.scatter3D(Z[:,0], Z[:,1], Z[:,2])
            verts = [[Z[0],Z[1],Z[2],Z[3]],
            [Z[4],Z[5],Z[6],Z[7]],
            [Z[1],Z[2],Z[5],Z[6]],
            [Z[0],Z[3],Z[4],Z[7]],
            [Z[0],Z[1],Z[6],Z[7]],
            [Z[2],Z[3],Z[4],Z[5]]]
            ax.add_collection3d(Poly3DCollection(verts, facecolors='magenta', linewidths=1, edgecolors='k', alpha=.20))
            
    def addObject(self, f, type, c, r, val):
        self.libprojectors.addObject.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_float]
        self.libprojectors.addObject.restype = ctypes.c_bool
        return self.libprojectors.addObject(f, type, c.astype(np.float32), r.astype(np.float32), val)
        
        