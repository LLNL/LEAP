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
            self.libprojectors = windll.LoadLibrary(os.path.join(current_dir, r'..\win_build\bin\Release\libLEAP.dll'))

    def reset(self):
        return self.libprojectors.reset()

    def printParameters(self):
        self.libprojectors.printParameters.restype = ctypes.c_bool
        return self.libprojectors.printParameters()
        
    def setGPU(self, which):
        self.libprojectors.setGPU.argtypes = [ctypes.c_int]
        self.libprojectors.setGPU.restype = ctypes.c_bool
        return self.libprojectors.setGPU(which)
        
    def setGPUs(self, listOfGPUs):
        self.libprojectors.setGPUs.argtypes = [ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_int]
        self.libprojectors.setGPUs.restype = ctypes.c_bool
        listOfGPUs = np.ascontiguousarray(listOfGPUs, dtype=np.int32)
        return self.libprojectors.setGPUs(listOfGPUs, int(listOfGPUs.size))
        
    def setDiameterFOV(self, d):
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
        
    def setProjector(self,which):
        self.libprojectors.setProjector.argtypes = [ctypes.c_int]
        self.libprojectors.setProjector.restype = ctypes.c_bool
        return self.libprojectors.setProjector(which)
        
    def setRampFilter(self,which):
        self.libprojectors.set_rampID.argtypes = [ctypes.c_int]
        self.libprojectors.set_rampID.restype = ctypes.c_bool
        return self.libprojectors.set_rampID(which)
    
    def setVolumeParams(self, numX, numY, numZ, voxelWidth=None, voxelHeight=None, offsetX=None, offsetY=None, offsetZ=None):
        self.libprojectors.setVolumeParams.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
        self.libprojectors.setVolumeParams.restype = ctypes.c_bool
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
        return self.libprojectors.setVolumeParams(numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ)
        
    def setDefaultVolume(self,scale=1.0):
        self.libprojectors.setDefaultVolumeParameters.argtypes = [ctypes.c_float]
        self.libprojectors.setDefaultVolumeParameters.restype = ctypes.c_bool
        return self.libprojectors.setDefaultVolumeParameters(scale)
        
    def setVolumeDimensionOrder(self,which):
        self.libprojectors.setVolumeDimensionOrder.argtypes = [ctypes.c_int]
        self.libprojectors.setVolumeDimensionOrder.restype = ctypes.c_bool
        return self.libprojectors.setVolumeDimensionOrder(which)
        
    def getVolumeDimensionOrder(self):
        return self.libprojectors.getVolumeDimensionOrder()

    def setAngleArray(self,numAngles,angularRange):
        return np.array(range(numAngles)).astype(np.float32) * angularRange/float(numAngles)
    
    def setConeBeamParams(self, numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd):
        self.libprojectors.setConeBeamParams.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_float, ctypes.c_float]
        self.libprojectors.setConeBeamParams.restype = ctypes.c_bool
        return self.libprojectors.setConeBeamParams(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd)
        
    def setFanBeamParams(self, numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd):
        self.libprojectors.setFanBeamParams.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_float, ctypes.c_float]
        self.libprojectors.setFanBeamParams.restype = ctypes.c_bool
        return self.libprojectors.setFanBeamParams(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd)

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
        
    def filterProjections(self, g):
        self.libprojectors.filterProjections.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool, ctypes.c_float]
        self.libprojectors.filterProjections.restype = ctypes.c_bool
        self.libprojectors.filterProjections(g,True,1.0)
        return g
        
    def rampFilterProjections(self, g):
        self.libprojectors.rampFilterProjections.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool, ctypes.c_float]
        self.libprojectors.rampFilterProjections.restype = ctypes.c_bool
        self.libprojectors.rampFilterProjections(g,True,1.0)
        return g
    
    def backproject(self,g,f):
        self.libprojectors.backproject.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
        self.libprojectors.backproject.restype = ctypes.c_bool
        self.libprojectors.backproject(g,f,True)
        return f
        
    def rampFilterVolume(self, f):
        self.libprojectors.rampFilterVolume.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
        self.libprojectors.rampFilterVolume.restype = ctypes.c_bool
        self.libprojectors.rampFilterVolume(f,True)
        return f

    def get_FBPscalar(self):
        self.libprojectors.get_FBPscalar.argtypes = []
        self.libprojectors.get_FBPscalar.restype = ctypes.c_float
        return self.libprojectors.get_FBPscalar()

    def FBP(self, g, f, inplace=False):
        '''
        self.rampFilterProjections(g)
        self.backproject(g,f)
        '''
        self.libprojectors.FBP.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
        self.libprojectors.FBP.restype = ctypes.c_bool
        if inplace:
            self.libprojectors.FBP(g,f,True)
        else:
            q  = g.copy()
            self.libprojectors.FBP(q,f,True)
        return f
        
    def BPF(self, g, f):
        self.backproject(g,f)
        self.rampFilterVolume(f)
        f *= self.get_FBPscalar()
        return f
        
    def MLEM(self, g, f, numIter):
        if not np.any(f):
            f[:] = 1.0
        ones = self.allocateProjections()
        ones[:] = 1.0
        Pstar1 = self.allocateVolume()
        self.backproject(ones, Pstar1)
        Pstar1[Pstar1==0.0] = 1.0
        d = self.allocateVolume()
        Pd = ones
        for n in range(numIter):
            self.project(Pd,f)
            ind = Pd != 0.0
            Pd[ind] = g[ind]/Pd[ind]
            self.backproject(Pd,d)
            f *= d/Pstar1
        return f
        
    def SART(self, g, f, numIter):
        ones = self.allocateVolume()
        ones[:] = 1.0
        P1 = self.allocateProjections()
        self.project(P1,ones)
        P1[P1==0.0] = 1.0
        
        Pstar1 = ones
        ones = self.allocateProjections()
        ones[:] = 1.0
        self.backproject(ones, Pstar1)
        Pstar1[Pstar1==0.0] = 1.0
        
        Pd = ones
        d = self.allocateVolume()

        for n in range(numIter):
            self.project(Pd,f)
            Pd = (g-Pd) / P1
            self.backproject(Pd,d)
            f += 0.9*d / Pstar1
        return f
        
    def RLS(self, g, f, numIter, delta=0.0, beta=0.0):
        return self.RWLS(g, f, numIter, delta, beta, 1.0)
        
    def RWLS(self, g, f, numIter, delta=0.0, beta=0.0, W=None):
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

    def get_numAngles(self):
        return self.libprojectors.get_numAngles()
        
    def get_numRows(self):
        return self.libprojectors.get_numRows()
        
    def get_numCols(self):
        return self.libprojectors.get_numCols()
        
    def get_numX(self):
        return self.libprojectors.get_numX()
    
    def get_numY(self):
        return self.libprojectors.get_numY()
    
    def get_numZ(self):
        return self.libprojectors.get_numZ()
        
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

    def allocateProjections(self):
        N_phis = self.get_numAngles()
        N_rows = self.get_numRows()
        N_cols = self.get_numCols()
        if N_phis > 0 and N_rows > 0 and N_cols > 0:
            return np.ascontiguousarray(np.zeros((N_phis,N_rows,N_cols)).astype(np.float32), dtype=np.float32)
        else:
            return None
        
    def allocateVolume(self):
        N_x = self.get_numX()
        N_y = self.get_numY()
        N_z = self.get_numZ()
        if N_x > 0 and N_y > 0 and N_z > 0:
            if self.getVolumeDimensionOrder() == 0:
                return np.ascontiguousarray(np.zeros((N_x,N_y,N_z)).astype(np.float32), dtype=np.float32)
            else:
                return np.ascontiguousarray(np.zeros((N_z,N_y,N_x)).astype(np.float32), dtype=np.float32)
        else:
            return None
    
    def displayVolume(self,vol):
        try:
            import napari
            viewer = napari.view_image(vol, rgb=False)
            napari.run()
        except:
            print('Cannot load napari, to install run this command:')
            print('pip install napari[all]')
