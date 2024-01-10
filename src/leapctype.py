################################################################################
# Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
# LEAP project developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# LivermorE AI Projector for Computed Tomography (LEAP)
# ctype tomographicModels class
################################################################################

import ctypes
import os
import sys
from sys import platform as _platform
from numpy.ctypeslib import ndpointer
import numpy as np
try:
    import torch
    has_torch = True
except:
    has_torch = False

class tomographicModels:
    """ Python class for tomographicModels bindings
    Usage Example:
    from leapctype import *
    leapct = tomographicModels()
    leapct.set_conebeam(...)
    leapct.set_default_volume(...)
    ...
    leapct.project(g,f)
    """

    def __init__(self, param_id=None, lib_dir=""):
        """Constructor

        The functions in this class can take as input and output data that is either on the CPU or the GPU.
        Note that all input and ouput data for all functions must lie either on a specific GPU or on the CPU.
        You cannot have some data on the CPU and some on a GPU.
        
        If the data is on the CPU (works for both numpy arrays or pytorch tensors):
            1) and one wishes the computations to take place on the CPU, then run the following
               command: set_gpu(-1).
            2) and one wishes the computations to take place on one of more GPUs, then run
               the following command: set_gpus(list_of_gpu_indices), e.g., set_gpus([0,1])
               The default setting is for LEAP to use all GPUs, so if this is what you want
               there is no need to run the set_gpus function
        If the data is on the GPU (only possible with torch tensors):
            then the computations must also take place on this particular GPU and
            you must run the following command: set_gpu(index), where index is GPU index
            of where the data resides.
            
        Providing data that is already on a GPU is best for small data sets, where data
        transfers to and from the GPU are more time consuming than the computations
        for the actual data processing.
        
        Providing data that is on the CPU is best for medium to large data sets because this
        allows for multi-GPU processing and LEAP will automatically divide up the data
        into small enough chunks so that is does not exceed the GPU memory.
        
        The LEAP library has the ability to generate and track several parameter sets.
        By default every new object instance will create a new parameter set.
        Otherwise one can use the param_id parameter so that the object will utilize a parameter set
        that is also shared by another object of this class.  See the param_id argument description below.
        
        Args:
            param_id (int): If no value is given, then a new parameter set is generated, otherwise one can specify a parameter set index to use
            lib_dir (string): Path to the LEAP dynamic library, default value is the same path as this file
        
        """
        if len(lib_dir) > 0:
            current_dir = lib_dir
        else:
            current_dir = os.path.abspath(os.path.dirname(__file__))

        if _platform == "linux" or _platform == "linux2":
            import readline
            from ctypes import cdll
            
            fullPath = os.path.join(current_dir, 'libleap.so')
            fullPath_backup = os.path.join(current_dir, '../build/lib/libleap.so')
            
            if os.path.isfile(fullPath):
                self.libprojectors = cdll.LoadLibrary(fullPath)
            elif os.path.isfile(fullPath_backup):
                self.libprojectors = cdll.LoadLibrary(fullPath_backup)
            else:
                print('Error: could not find LEAP dynamic library at')
                print(fullPath)
                print('or')
                print(fullPath_backup)
                self.libprojectors = None
            
        elif _platform == "win32":
            from ctypes import windll
        
            fullPath = os.path.join(current_dir, 'libleap.dll')
            fullPath_backup = os.path.join(current_dir, r'..\win_build\bin\Release\libleap.dll')
        
            if os.path.isfile(fullPath):
                try:
                    self.libprojectors = windll.LoadLibrary(fullPath)
                except:
                    self.libprojectors = ctypes.CDLL(fullPath, winmode=0)
            elif os.path.isfile(fullPath_backup):
                try:
                    self.libprojectors = windll.LoadLibrary(fullPath_backup)
                except:
                    self.libprojectors = ctypes.CDLL(fullPath_backup, winmode=0)
            else:
                print('Error: could not find LEAP dynamic library at')
                print(fullPath)
                print('or')
                print(fullPath_backup)
                self.libprojectors = None
        
        elif _platform == "darwin":  # Darwin is the name for MacOS in Python's platform module
            # there is current no support for LEAP on Mac, but maybe someone can figure this out
            from ctypes import cdll
            
            fullPath = os.path.join(current_dir, 'libleap.dylib')
            fullPath_backup = os.path.join(current_dir, '../build/lib/libleap.dylib')
            
            if os.path.isfile(fullPath):
                self.libprojectors = cdll.LoadLibrary(fullPath)
            elif os.path.isfile(fullPath_backup):
                self.libprojectors = cdll.LoadLibrary(fullPath_backup)
            else:
                print('Error: could not find LEAP dynamic library at')
                print(fullPath)
                print('or')
                print(fullPath_backup)
                self.libprojectors = None
            
        if param_id is not None:
            self.param_id = param_id
        else:
            self.param_id = self.create_new_model()
        self.set_model()

    def set_model(self, i=None):
        self.libprojectors.set_model.restype = ctypes.c_bool
        self.libprojectors.set_model.argtypes = [ctypes.c_int]
        if i is None:
            return self.libprojectors.set_model(self.param_id)
        else:
            return self.libprojectors.set_model(i)

    def create_new_model(self):
        self.libprojectors.create_new_model.restype = ctypes.c_int
        return self.libprojectors.create_new_model()

    def reset(self):
        """reset
        Resets and clears all parameters
        """
        self.set_model()
        return self.libprojectors.reset()

    def about(self):
        """prints info about LEAP, including the version number"""
        self.set_model()
        return self.libprojectors.about()

    def printParameters(self):
        """printParameters
        prints all CT geometry and CT volume parameters to the screen
        """
        return self.print_parameters()

    def print_param(self):
        """printParameters
        prints all CT geometry and CT volume parameters to the screen
        """
        return self.print_parameters()

    def print_parameters(self):
        """print_parameters
        prints all CT geometry and CT volume parameters to the screen
        """
        self.libprojectors.print_parameters.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.print_parameters()

    ###################################################################################################################
    ###################################################################################################################
    # THIS SECTION OF FUNCTIONS SET THE CT SCANNER GEOMETRY PARAMETERS
    ###################################################################################################################
    ###################################################################################################################
    def set_conebeam(self, numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau=0.0, helicalPitch=0.0):
        """Sets the parameters for a cone-beam CT geometry
        
        The origin of the coordinate system is always at the center of rotation
        
        Args:
            The first three parameters specify the projection data array size
            numAngles (int): number of projection angles
            numRows (int): number of rows in the x-ray detector
            numCols (int): number of columns in the x-ray detector
        
            The next two parameters specify the detector pixel size in mm
            pixelHeight (float): the detector pixel pitch (i.e., pixel size) between detector rows, measured in mm
            pixelWidth (float): the detector pixel pitch (i.e., pixel size) between detector columns, measured in mm
        
            The next two parameters specify the placement of the detector, i.e.,
            changing these parameters causes shifts in the detector location relative to the source
            centerRow (float): the detector pixel row index for the ray that passes from the source, through the origin, and hits the detector
            centerCol (float): the detector pixel column index for the ray that passes from the source, through the origin, and hits the detector
        
            phis (float32 numpy array):  a numpy array for specifying the angles of each projection, measured in degrees
        
            sod (float): source to object distance, measured in mm; this can also be viewed as the source to center of rotation distance
            sdd (float): source to detector distance, measured in mm
            
            tau (float): center of rotation offset
            helicalPitch (float): the helical pitch (mm/radians)
            
        Returns:
            True if the parameters were valid, false otherwise
        """
        self.libprojectors.set_conebeam.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
        self.libprojectors.set_conebeam.restype = ctypes.c_bool
        if has_torch and type(phis) is torch.Tensor:
            phis = phis.numpy()
        elif type(phis) is not np.ndarray:
            angularRange = float(phis)
            phis = self.setAngleArray(numAngles, angularRange)
        self.set_model()
        return self.libprojectors.set_conebeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau, helicalPitch)
    
    def set_coneBeam(self, numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau=0.0, helicalPitch=0.0):
        """Alias for set_conebeam
        """
        return self.set_conebeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau, helicalPitch)
    
    def set_fanbeam(self, numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau=0.0):
        """Sets the parameters for a fan-beam CT geometry
        
        The origin of the coordinate system is always at the center of rotation
        
        Args:
            The first three parameters specify the projection data array size
            numAngles (int): number of projection angles
            numRows (int): number of rows in the x-ray detector
            numCols (int): number of columns in the x-ray detector
        
            The next two parameters specify the detector pixel size in mm
            pixelHeight (float): the detector pixel pitch (i.e., pixel size) between detector rows, measured in mm
            pixelWidth (float): the detector pixel pitch (i.e., pixel size) between detector columns, measured in mm
        
            The next two parameters specify the placement of the detector, i.e.,
            changing these parameters causes shifts in the detector location relative to the source
            centerRow (float): the detector pixel row index for the ray that passes from the source, through the origin, and hits the detector
            centerCol (float): the detector pixel column index for the ray that passes from the source, through the origin, and hits the detector
        
            phis (float32 numpy array):  a numpy array for specifying the angles of each projection, measured in degrees
        
            sod (float): source to object distance, measured in mm; this can also be viewed as the source to center of rotation distance
            sdd (float): source to detector distance, measured in mm
            
            tau (float): center of rotation offset
            
        Returns:
            True if the parameters were valid, false otherwise
        """
        self.libprojectors.set_fanbeam.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_float, ctypes.c_float, ctypes.c_float]
        self.libprojectors.set_fanbeam.restype = ctypes.c_bool
        if has_torch and type(phis) is torch.Tensor:
            phis = phis.numpy()
        elif type(phis) is not np.ndarray:
            angularRange = float(phis)
            phis = self.setAngleArray(numAngles, angularRange)
        self.set_model()
        return self.libprojectors.set_fanbeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau)
        
    def set_fanBeam(self, numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau=0.0):
        """Alias for set_fanbeam
        """
        return self.set_fanbeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau)

    def set_parallelbeam(self, numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis):
        """Sets the parameters for a parallel-beam CT geometry
        
        The origin of the coordinate system is always at the center of rotation
        
        Args:
            The first three parameters specify the projection data array size
            numAngles (int): number of projection angles
            numRows (int): number of rows in the x-ray detector
            numCols (int): number of columns in the x-ray detector
        
            The next two parameters specify the detector pixel size in mm
            pixelHeight (float): the detector pixel pitch (i.e., pixel size) between detector rows, measured in mm
            pixelWidth (float): the detector pixel pitch (i.e., pixel size) between detector columns, measured in mm
        
            The next two parameters specify the placement of the detector, i.e.,
            changing these parameters causes shifts in the detector location relative to the source
            centerRow (float): the detector pixel row index for the ray that passes from the source, through the origin, and hits the detector
            centerCol (float): the detector pixel column index for the ray that passes from the source, through the origin, and hits the detector
        
            phis (float32 numpy array):  a numpy array for specifying the angles of each projection, measured in degrees
            
        Returns:
            True if the parameters were valid, false otherwise
        """
        self.libprojectors.set_parallelbeam.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
        self.libprojectors.set_parallelbeam.restype = ctypes.c_bool
        if has_torch and type(phis) is torch.Tensor:
            phis = phis.numpy()
        elif type(phis) is not np.ndarray:
            angularRange = float(phis)
            phis = self.setAngleArray(numAngles, angularRange)
        self.set_model()
        return self.libprojectors.set_parallelbeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis)

    def set_parallelBeam(self, numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis):
        """Alias for set_parallelbeam
        """
        return self.set_parallelbeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis)

    def set_modularbeam(self, numAngles, numRows, numCols, pixelHeight, pixelWidth, sourcePositions, moduleCenters, rowVectors, colVectors):
        """Sets the parameters for a modular-beam CT geometry
        
        The origin of the coordinate system is always at the center of rotation
        
        Args:
            The first three parameters specify the projection data array size
            numAngles (int): number of projection angles
            numRows (int): number of rows in the x-ray detector
            numCols (int): number of columns in the x-ray detector
        
            The next two parameters specify the detector pixel size in mm
            pixelHeight (float): the detector pixel pitch (i.e., pixel size) between detector rows, measured in mm
            pixelWidth (float): the detector pixel pitch (i.e., pixel size) between detector columns, measured in mm
        
            The next two parameters specify the placement of the source and detector pairs
            sourcePositions ((numAngles X 3) numpy array): the (x,y,z) position of each x-ray source
            moduleCenters ((numAngles X 3) numpy array): the (x,y,z) position of the center of the front face of the detectors
        
            The next two parameters specify the orientation of the detectors
            rowVectors ((numAngles X 3) numpy array):  the (x,y,z) unit vector point along the positive detector row direction
            colVectors ((numAngles X 3) numpy array):  the (x,y,z) unit vector point along the positive detector column direction
            
        Returns:
            True if the parameters were valid, false otherwise
        """
        self.libprojectors.set_modularbeam.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
        self.libprojectors.set_modularbeam.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_modularbeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, sourcePositions, moduleCenters, rowVectors, colVectors)
    
    def set_modularBeam(self, numAngles, numRows, numCols, pixelHeight, pixelWidth, sourcePositions, moduleCenters, rowVectors, colVectors):
        """Alias for set_modularbeam
        """
        return self.set_modularbeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, sourcePositions, moduleCenters, rowVectors, colVectors)
    
    def set_tau(self, tau):
        """Set the tau parameter
        
        Args:
            tau (float): center of rotation offset in mm (fan- and cone-beam data only)
            
        Returns:
            True if the parameters were valid, false otherwise
        """
        self.libprojectors.set_tau.argtypes = [ctypes.c_float]
        self.libprojectors.set_tau.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_tau(tau)
    
    def set_helicalPitch(self, helicalPitch):
        """Set the helicalPitch parameter
        
        Args:
            helicalPitch (float): the helical pitch (mm/radians) (cone-beam data only)
            
        Returns:
            True if the parameters were valid, false otherwise
        """
        self.libprojectors.set_helicalPitch.argtypes = [ctypes.c_float]
        self.libprojectors.set_helicalPitch.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_helicalPitch(h)
        
    def set_normalizedHelicalPitch(self, normalizedHelicalPitch):
        """Set the helicalPitch parameter
        
        Args:
            normalizedHelicalPitch (float): the normalized helical pitch (unitless) (cone-beam data only)
            
        Returns:
            True if the parameters were valid, false otherwise
        """
        self.libprojectors.set_normalizedHelicalPitch.argtypes = [ctypes.c_float]
        self.libprojectors.set_normalizedHelicalPitch.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_normalizedHelicalPitch(normalizedHelicalPitch)
        
    def set_flatDetector(self):
        """Set the detectorType to FLAT"""
        self.set_model()
        self.libprojectors.set_flatDetector.restype = ctypes.c_bool
        return self.libprojectors.set_flatDetector()
        
    def set_curvedDetector(self):
        """Set the detectorType to CURVED"""
        self.set_model()
        self.libprojectors.set_curvedDetector.restype = ctypes.c_bool
        return self.libprojectors.set_curvedDetector()
        
    def get_detectorType(self):
        """Get the detectorType"""
        self.set_model()
        self.libprojectors.get_detectorType.restype = ctypes.c_int
        if self.libprojectors.get_detectorType() == 0:
            return 'FLAT'
        else:
            return 'CURVED'
            
    def set_centerCol(self, centerCol):
        """Set centerCol parameter"""
        self.set_model()
        self.libprojectors.set_centerCol.restype = ctypes.c_bool
        self.libprojectors.set_centerCol.argtypes = [ctypes.c_float]
        return self.libprojectors.set_centerCol(centerCol)
        
    def set_centerRow(self, centerRow):
        """Set centerRow parameter"""
        self.set_model()
        self.libprojectors.set_centerRow.restype = ctypes.c_bool
        self.libprojectors.set_centerRow.argtypes = [ctypes.c_float]
        return self.libprojectors.set_centerRow(centerRow)
        
    def convert_conebeam_to_modularbeam(self):
        """sets modular-beam parameters from a cone-beam specification"""
        self.set_model()
        self.libprojectors.convert_conebeam_to_modularbeam.restype = ctypes.c_bool
        return self.libprojectors.convert_conebeam_to_modularbeam()
    
    ###################################################################################################################
    ###################################################################################################################
    # THIS SECTION OF FUNCTIONS SET THE CT VOLUME PARAMETERS
    ###################################################################################################################
    ###################################################################################################################
    def set_volume(self, numX, numY, numZ, voxelWidth=None, voxelHeight=None, offsetX=None, offsetY=None, offsetZ=None):
        """Set the CT volume parameters
        
        Args:
            The first three parameters specify the volume array size
            For parallel- and fan-beam data, it is required that numRows=numZ
            numX (int): number of voxels in the x-dimension
            numY (int): number of voxels in the y-dimension
            numZ (int): number of voxels in the z-dimension
        
            The next two parameters specify the size of the voxels in mm
            For parallel- and fan-beam data, it is required that pixelHeight=voxelHeigh
            voxelWidth (float): voxel pitch (size) in the x and y dimensions
            voxelHeight (float): voxel pitch (size) in the z dimension
        
            The final three parameters specify the location of the voxel array
            The default position is that the voxels are centered around the origin
            and these parameters allow one to shift the volume around
            For parallel- and fan-beam data, it is required that offsetZ=0.0
            offsetX (float): shift the volume in the x-dimension, measured in mm
            offsetY (float): shift the volume in the y-dimension, measured in mm
            offsetZ (float): shift the volume in the z-dimension, measured in mm
            
        Returns:
            True if the parameters were valid, false otherwise
        """
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
        self.set_model()
        return self.libprojectors.set_volume(numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ)
        
    def set_default_volume(self,scale=1.0):
        """Sets the default volume parameters
        
        The default volume parameters are those that fill the field of view of the CT system and use the native voxel sizes
        
        Args:
            scale (float): this value scales the voxel size by this value to create denser or sparser voxel representations (not recommended for fast reconstruction)
        
        Returns:
            True if the operation was successful, false otherwise (this usually happens if the CT geometry has not yet been set)
        """
        self.libprojectors.set_default_volume.argtypes = [ctypes.c_float]
        self.libprojectors.set_default_volume.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_default_volume(scale)
    
    def set_defaultVolume(self,scale=1.0):
        """Alias for set_default_volume
        """
        return self.set_default_volume(scale)
    
    def set_volumeDimensionOrder(self,which):
        """Sets the order of the dimensions of the volume
        
        WARNING: multi-GPU processing only works for ZYX order
        
        Args:
            setVolumeDimensionOrder (int): 0 sets the order to XYZ, 1 sets the order to ZYX (this is the default value)
            
        Returns:
            True if the operation was successful, false otherwise
        """
        self.libprojectors.set_volumeDimensionOrder.argtypes = [ctypes.c_int]
        self.libprojectors.set_volumeDimensionOrder.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_volumeDimensionOrder(which)
        
    def get_volumeDimensionOrder(self):
        """Alias for set_volumeDimensionOrder
        """
        return self.libprojectors.get_volumeDimensionOrder()
        
    ###################################################################################################################
    ###################################################################################################################
    # THIS SECTION OF FUNCTIONS PROVIDE CONVENIENT ROUTINES TO MAKE THE PROJECTION DATA AND VOLUME DATA NUMPY ARRAYS
    ###################################################################################################################
    ###################################################################################################################
    def allocate_projections(self, val=0.0, astensor=False):
        """Alias for allocateProjections"""
        return self.allocateProjections(val, astensor)
    
    def allocateProjections(self, val=0.0, astensor=False):
        """Allocates projection data
        
        It is not necessary to use this function. It is included simply for convenience.

        Args:
            val (float): value to fill the array with
            astensor (bool): if true turns array into a pytorch tensor
            
        Returns:
            numpy array/ pytorch tensor if numAngles, numRows, and numCols are all positive, None otherwise
        """
        N_phis = self.get_numAngles()
        N_rows = self.get_numRows()
        N_cols = self.get_numCols()
        if N_phis > 0 and N_rows > 0 and N_cols > 0:
            if val == 0.0:
                g = np.ascontiguousarray(np.zeros((N_phis,N_rows,N_cols),dtype=np.float32), dtype=np.float32)
            else:
                g = np.ascontiguousarray(val*np.ones((N_phis,N_rows,N_cols),dtype=np.float32), dtype=np.float32)
            if has_torch and astensor:
                g = torch.from_numpy(g)
            return g
        else:
            return None
            
    def allocateProjections_gpu(self, val=0.0):
        """Allocates projection data as a pytorch tensor on the gpu
        
        It is not necessary to use this function. It is included simply for convenience.

        Args:
            val (float): value to fill the array with
            
        Returns:
            pytorch tensor if numAngles, numRows, and numCols are all positive, None otherwise
        """
        if has_torch == False:
            print('Error: you must have pytorch installed for this function')
            return None
        N_phis = self.get_numAngles()
        N_rows = self.get_numRows()
        N_cols = self.get_numCols()
        if N_phis > 0 and N_rows > 0 and N_cols > 0:
            if val == 0.0:
                g = np.ascontiguousarray(np.zeros((N_phis,N_rows,N_cols),dtype=np.float32), dtype=np.float32)
            else:
                g = np.ascontiguousarray(val*np.ones((N_phis,N_rows,N_cols),dtype=np.float32), dtype=np.float32)
            if has_torch:
                g = torch.from_numpy(g).float().to(max(0,self.get_gpu()))
            return g
        else:
            return None
        
        
    def get_volume_dim(self):
        """Get dimension sizes of volume data
        
        It is not necessary to use this function. It is included simply for convenience.

        Returns:
            numpy array of the dimension sizes of the volume
        """
        N_x = self.get_numX()
        N_y = self.get_numY()
        N_z = self.get_numZ()
        if self.get_volumeDimensionOrder() == 0:
            return np.array([N_x,N_y,N_z],dtype=np.int32)
        else:
            return np.array([N_z,N_y,N_x],dtype=np.int32)
            
    def get_projection_dim(self):
        """Get dimension sizes of projection data
        
        It is not necessary to use this function. It is included simply for convenience.

        Returns:
            numpy array of the dimension sizes of the projections
        """
        return np.array([self.get_numAngles(),self.get_numRows(),self.get_numCols()],dtype=np.int32)
        
    def allocate_volume(self, val=0.0, astensor=False):
        """Alias for allocateVolume"""
        return self.allocateVolume(val, astensor)
        
    def allocateVolume(self, val=0.0, astensor=False):
        """Allocates reconstruction volume data
        
        It is not necessary to use this function. It is included simply for convenience.

        Args:
            val (float): value to fill the array with
            
        Returns:
            numpy array/ pytorch tensor if numAngles, numRows, and numCols are all positive, None otherwise
        """
        dim1, dim2, dim3 = self.get_volume_dim()
        if dim1 > 0 and dim2 > 0 and dim3 > 0:
            if val == 0.0:
                f = np.ascontiguousarray(np.zeros((dim1,dim2,dim3),dtype=np.float32), dtype=np.float32)
            else:
                f = np.ascontiguousarray(val*np.ones((dim1,dim2,dim3),dtype=np.float32), dtype=np.float32)
            if has_torch and astensor:
                f = torch.from_numpy(f)
            return f
        else:
            return None
            
    def allocateVolume_gpu(self, val=0.0):
        """Allocates reconstruction volume data as a pytorch tensor on the gpu
        
        It is not necessary to use this function. It is included simply for convenience.

        Args:
            val (float): value to fill the array with
            
        Returns:
            pytorch tensor if numAngles, numRows, and numCols are all positive, None otherwise
        """
        if has_torch == False:
            print('Error: you must have pytorch installed for this function')
            return None
            
        dim1, dim2, dim3 = self.get_volume_dim()
        if dim1 > 0 and dim2 > 0 and dim3 > 0:
            if val == 0.0:
                f = np.ascontiguousarray(np.zeros((dim1,dim2,dim3),dtype=np.float32), dtype=np.float32)
            else:
                f = np.ascontiguousarray(val*np.ones((dim1,dim2,dim3),dtype=np.float32), dtype=np.float32)
            if has_torch:
                f = torch.from_numpy(f).float().to(max(0,self.get_gpu()))
            return f
        else:
            return None
            
    def allocateData(self, x, val=0.0):
        if has_torch == True and type(x) is torch.Tensor:
            if x.is_cuda:
                if val == 0.0:
                    y = torch.zeros([x.shape[0], x.shape[1], x.shape[2]], dtype=torch.float32, device=torch.device('cuda:'+str(self.get_gpu())))
                else:
                    y = val*torch.ones([x.shape[0], x.shape[1], x.shape[2]], dtype=torch.float32, device=torch.device('cuda:'+str(self.get_gpu())))
            else:
                if val == 0.0:
                    y = torch.zeros([x.shape[0], x.shape[1], x.shape[2]], dtype=torch.float32)
                else:
                    y = val*torch.ones([x.shape[0], x.shape[1], x.shape[2]], dtype=torch.float32)
        else:
            if val == 0.0:
                y = np.ascontiguousarray(np.zeros((x.shape[0], x.shape[1], x.shape[2]),dtype=np.float32), dtype=np.float32)
            else:
                y = np.ascontiguousarray(val*np.ones((x.shape[0], x.shape[1], x.shape[2]),dtype=np.float32), dtype=np.float32)
        return y
            
    def setAngleArray(self,numAngles,angularRange):
        """Sets the angle array, i.e., phis which specifies the projection angles for parallel-, fan-, and cone-beam data
        
        It is not necessary to use this function. It is included simply for convenience.

        Args:
            val (float): value to fill the array with
            
        Returns:
            numpy array is numAngles, numRows, and numCols are all positive, None otherwise
        """
        return np.array(range(numAngles)).astype(np.float32) * angularRange/float(numAngles)
        
    def copyData(self, x):
        if has_torch == True and type(x) is torch.Tensor:
            x_copy = x.clone()
            #x_copy.to(x.get_device())
            return x_copy
        else:
            return x.copy()
    
    ###################################################################################################################
    ###################################################################################################################
    # THIS SECTION OF FUNCTIONS EXECUTE THE MAIN CPU/GPU ROUTINES IN LEAP
    ###################################################################################################################
    ###################################################################################################################
    def project(self, g, f, param_id=None):
        """Calculate the forward projection of f and stores the result in g
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        This function take the argument g and returns the same g.
        Returning g is just there for nesting several algorithms.
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data
            f (C contiguous float32 numpy array or torch tensor): volume data
            param_id (int): optional parameter to project a particular parameter index
            
        Returns:
            g, the same as the input with the same name
        """
        self.libprojectors.project.restype = ctypes.c_bool
        self.set_model(param_id)
        if has_torch == True and type(g) is torch.Tensor:
            self.libprojectors.project.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool]
            self.libprojectors.project(g.data_ptr(), f.data_ptr(), g.is_cuda == False)
        else:
            self.libprojectors.project.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
            self.libprojectors.project(g, f, True)
        return g
        
    def project_cpu(self, g, f, param_id=None):
        """Calculate the forward projection of f and stores the result in g
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        This function take the argument g and returns the same g.
        Returning g is just there for nesting several algorithms.
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data on the CPU
            f (C contiguous float32 numpy array or torch tensor): volume data on the CPU
            param_id (int): optional parameter to project a particular parameter index
            
        Returns:
            g, the same as the input with the same name
        """
        self.libprojectors.project_cpu.restype = ctypes.c_bool
        self.set_model(param_id)
        if has_torch == True and type(g) is torch.Tensor:
            if g.is_cuda:
                print('Error: project_cpu requires that the data be on the CPU')
            else:
                self.libprojectors.project_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                self.libprojectors.project_cpu(g.data_ptr(), f.data_ptr())
        else:
            self.libprojectors.project_cpu.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
            self.libprojectors.project_cpu(g, f)
        return g
        
    def project_gpu(self, g, f, param_id=None):
        """Calculate the forward projection of f and stores the result in g
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        This function take the argument g and returns the same g.
        Returning g is just there for nesting several algorithms.
        
        Args:
            g (C contiguous float32 torch tensor): projection data on a GPU
            f (C contiguous float32 torch tensor): volume data on a GPU
            param_id (int): optional parameter to project a particular parameter index
            
        Returns:
            g, the same as the input with the same name
        """
        self.libprojectors.project_gpu.restype = ctypes.c_bool
        self.set_model(param_id)
        if has_torch == True and type(g) is torch.Tensor:
            if g.is_cuda == False:
                print('Error: project_gpu requires that the data be on the GPU')
            else:
                self.libprojectors.project_gpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                self.libprojectors.project_gpu(g.data_ptr(), f.data_ptr())
        else:
            print('Error: project_gpu requires that the data be pytorch tensors on the GPU')
        return g
            
    def backproject(self, g, f, param_id=None):
        """Calculate the backprojection (adjoint of the forward projection) of g and stores the result in f
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        This function take the argument f and returns the same f.
        Returning f is just there for nesting several algorithms.
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data
            f (C contiguous float32 numpy array or torch tensor): volume data
            param_id (int): optional parameter to project a particular parameter index
            
        Returns:
            f, the same as the input with the same name
        """
        self.libprojectors.backproject.restype = ctypes.c_bool
        self.set_model(param_id)
        if has_torch == True and type(g) is torch.Tensor:
            self.libprojectors.backproject.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool]
            self.libprojectors.backproject(g.data_ptr(), f.data_ptr(), g.is_cuda == False)
        else:
            self.libprojectors.backproject.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
            self.libprojectors.backproject(g, f, True)
        return f
        
    def backproject_cpu(self, g, f, param_id=None):
        """Calculate the backprojection (adjoint of the forward projection) of g and stores the result in f
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        This function take the argument f and returns the same f.
        Returning f is just there for nesting several algorithms.
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data
            f (C contiguous float32 numpy array or torch tensor): volume data
            param_id (int): optional parameter to project a particular parameter index
            
        Returns:
            f, the same as the input with the same name
        """
        self.libprojectors.backproject_cpu.restype = ctypes.c_bool
        self.set_model(param_id)
        if has_torch == True and type(g) is torch.Tensor:
            if g.is_cuda:
                print('Error: backproject_cpu requires that the data be on the CPU')
            else:
                self.libprojectors.backproject_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                self.libprojectors.backproject_cpu(g.data_ptr(), f.data_ptr())
        else:
            self.libprojectors.backproject_cpu.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
            self.libprojectors.backproject_cpu(g, f)
        return f
        
    def backproject_gpu(self, g, f, param_id=None):
        """Calculate the backprojection (adjoint of the forward projection) of g and stores the result in f
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        This function take the argument f and returns the same f.
        Returning f is just there for nesting several algorithms.
        
        Args:
            g (C contiguous float32 torch tensor): projection data
            f (C contiguous float32 torch tensor): volume data
            param_id (int): optional parameter to project a particular parameter index
            
        Returns:
            f, the same as the input with the same name
        """
        self.libprojectors.backproject_gpu.restype = ctypes.c_bool
        self.set_model(param_id)
        if has_torch == True and type(g) is torch.Tensor:
            if g.is_cuda == False:
                print('Error: backproject_gpu requires that the data be on the GPU')
            else:
                self.libprojectors.backproject_gpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                self.libprojectors.backproject_gpu(g.data_ptr(), f.data_ptr())
        else:
            print('Error: backproject_gpu requires that the data be pytorch tensors on the GPU')
        return f
        
    def filterProjections(self, g):
        """Filters the projection data, g, so that its backprojection results in an FBP reconstruction
        
        The CT geometry parameters must be set prior to running this function.
        This function take the argument g and returns the same g.
        Returning g is just there for nesting several algorithms.
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data
            
        Returns:
            g, the same as the input with the same name
        """
        self.libprojectors.filterProjections.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(g) is torch.Tensor:
            self.libprojectors.filterProjections.argtypes = [ctypes.c_void_p, ctypes.c_bool]
            self.libprojectors.filterProjections(g.data_ptr(), g.is_cuda == False)
        else:
            self.libprojectors.filterProjections.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
            self.libprojectors.filterProjections(g, True)
        return g
        
    def rampFilterProjections(self, g):
        """Applies the ramp filter to the projection data, g, which is a subset of the operations in the filterProjections function
        
        The CT geometry parameters must be set prior to running this function.
        This function take the argument g and returns the same g.
        Returning g is just there for nesting several algorithms.
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data
            
        Returns:
            g, the same as the input with the same name
        """
        self.libprojectors.rampFilterProjections.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(g) is torch.Tensor:
            self.libprojectors.rampFilterProjections.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_float]
            self.libprojectors.rampFilterProjections(g.data_ptr(), g.is_cuda == False, 1.0)
        else:
            self.libprojectors.rampFilterProjections.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool, ctypes.c_float]
            self.libprojectors.rampFilterProjections(g, True, 1.0)
        return g
        
    def HilbertFilterProjections(self, g):
        """Applies the Hilbert filter to the projection data, g, which is a subset of the operations in some reconstruction algorithms
        
        The CT geometry parameters must be set prior to running this function.
        This function take the argument g and returns the same g.
        Returning g is just there for nesting several algorithms.
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data
            
        Returns:
            g, the same as the input with the same name
        """
        self.libprojectors.HilbertFilterProjections.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(g) is torch.Tensor:
            self.libprojectors.HilbertFilterProjections.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_float]
            self.libprojectors.HilbertFilterProjections(g.data_ptr(), g.is_cuda == False, 1.0)
        else:
            self.libprojectors.HilbertFilterProjections.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool, ctypes.c_float]
            self.libprojectors.HilbertFilterProjections(g, True, 1.0)
        return g
    
    def weightedBackproject(self,g,f):
        """Calculate the weighted backprojection of g and stores the result in f
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        This function take the argument f and returns the same f.
        Returning f is just there for nesting several algorithms.
        
        Some geometries require a weighted backprojection for FBP reconstruction,
        such as fan-beam, helical cone-beam, Attenuated Radon Transform, and symmetric objects
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data
            f (C contiguous float32 numpy array or torch tensor): volume data
            
        Returns:
            f, the same as the input with the same name
        """
        self.libprojectors.weightedBackproject.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(g) is torch.Tensor:
            self.libprojectors.weightedBackproject.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool]
            self.libprojectors.weightedBackproject(g.data_ptr(), f.data_ptr(), g.is_cuda == False)
        else:
            self.libprojectors.weightedBackproject.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
            self.libprojectors.weightedBackproject(g, f, True)
        return f
        
    def rampFilterVolume(self, f):
        """Applies the 2D ramp filter to the volume data, f, for each z-slice
        
        Args:
            f (C contiguous float32 numpy array or torch tensor): volume data
            
        Returns:
            f, the same as the input with the same name
        """
        self.libprojectors.rampFilterVolume.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(f) is torch.Tensor:
            self.libprojectors.rampFilterVolume.argtypes = [ctypes.c_void_p, ctypes.c_bool]
            self.libprojectors.rampFilterVolume(f.data_ptr(), f.is_cuda == False)
        else:
            self.libprojectors.rampFilterVolume.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
            self.libprojectors.rampFilterVolume(f, True)
        return f
        
    def Laplacian(self, g):
        """Applies a Laplacian operation to each projection"""
        self.libprojectors.Laplacian.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(g) is torch.Tensor:
            self.libprojectors.Laplacian.argtypes = [ctypes.c_void_p, ctypes.c_bool]
            self.libprojectors.Laplacian(g.data_ptr(), g.is_cuda == False)
        else:
            self.libprojectors.Laplacian.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
            self.libprojectors.Laplacian(g, True)
        return g
        
    def AzimuthalBlur(self, f, FWHM):
        """Applies an low pass filter to the volume data in the azimuthal direction, f, for each z-slice
        
        Args:
            f (C contiguous float32 numpy array or torch tensor): volume data
            FWHM (float): full width at half maximum of the filter (in degrees)
            
        Returns:
            f, the same as the input with the same name
        """
        self.libprojectors.AzimuthalBlur.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(f) is torch.Tensor:
            self.libprojectors.AzimuthalBlur.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_bool]
            self.libprojectors.AzimuthalBlur(f.data_ptr(), FWHM, f.is_cuda == False)
        else:
            self.libprojectors.AzimuthalBlur.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_float, ctypes.c_bool]
            self.libprojectors.AzimuthalBlur(f, FWHM, True)
        return f

    def get_FBPscalar(self):
        """Returns the scalar necessary for quantitative reconstruction when using the filterProjections and backproject functions
        """
        self.libprojectors.get_FBPscalar.argtypes = []
        self.libprojectors.get_FBPscalar.restype = ctypes.c_float
        self.set_model()
        return self.libprojectors.get_FBPscalar()

    def fbp(self, g, f=None, inplace=False):
        """Alias for FBP"""
        return self.FBP(g, f, inplace)

    def FBP(self, g, f=None, inplace=False):
        """Performs a Filtered Backprojection (FBP) reconstruction of the projection data, g, and stores the result in f
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        This function take the argument f and returns the same f.
        Returning f is just there for nesting several algorithms.
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data
            f (C contiguous float32 numpy array or torch tensor): volume data
            
        Returns:
            f, the same as the input with the same name
        """
        
        # Make a copy of g if necessary
        if inplace == False:
            q = self.copyData(g)
        else:
            q = g
        
        self.libprojectors.FBP.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(q) is torch.Tensor:
            if f is None:
                f = self.allocateVolume(0.0,True)
                f = f.to(g.get_device())
            self.libprojectors.FBP.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool]
            self.libprojectors.FBP(q.data_ptr(), f.data_ptr(), q.is_cuda == False)
        else:
            if f is None:
                f = self.allocateVolume()
            self.libprojectors.FBP.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
            self.libprojectors.FBP(q, f, True)
        return f
        
    def fbp_cpu(self, g, f, inplace=False):
        return self.FBP_cpu(g, f, inplace)
        
    def fbp_gpu(self, g, f, inplace=False):
        return self.FBP_gpu(g, f, inplace)
        
    def FBP_cpu(self, g, f, inplace=False):
        """Performs a Filtered Backprojection (FBP) reconstruction of the projection data, g, and stores the result in f
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        This function take the argument f and returns the same f.
        Returning f is just there for nesting several algorithms.
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data
            f (C contiguous float32 numpy array or torch tensor): volume data
            
        Returns:
            f, the same as the input with the same name
        """
        
        # First make validation checks that the data is on the CPU
        if has_torch == True and type(g) is torch.Tensor and g.is_cuda == True:
            print('Error: FBP_cpu requires that the data be on the CPU')
            return f

        # Make a copy of g if necessary
        if inplace == False:
            q = self.copyData(g)
        else:
            q = g
            
        self.libprojectors.FBP_cpu.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(q) is torch.Tensor:
            self.libprojectors.FBP_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            self.libprojectors.FBP_cpu(q.data_ptr(), f.data_ptr())
        else:
            self.libprojectors.FBP_cpu.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
            self.libprojectors.FBP_cpu(q, f)
        return f
        
    def FBP_gpu(self, g, f, inplace=False):
        """Performs a Filtered Backprojection (FBP) reconstruction of the projection data, g, and stores the result in f
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        This function take the argument f and returns the same f.
        Returning f is just there for nesting several algorithms.
        
        Args:
            g (C contiguous float32 torch tensor): projection data
            f (C contiguous float32 torch tensor): volume data
            
        Returns:
            f, the same as the input with the same name
        """
        
        # First make validation checks that the data is on the CPU
        if has_torch == True and type(g) is torch.Tensor:
            if g.is_cuda == False:
                print('Error: FBP_gpu requires that the data be on the GPU')
                return f
        else:
            print('Error: FBP_gpu requires that the data be pytorch tensors on the GPU')
            return f

        # Make a copy of g if necessary
        if inplace == False:
            q = self.copyData(g)
        else:
            q = g
        
        if has_torch == True and type(q) is torch.Tensor:
            self.libprojectors.FBP_gpu.restype = ctypes.c_bool
            self.libprojectors.FBP_gpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            self.set_model()
            self.libprojectors.FBP_gpu(q.data_ptr(), f.data_ptr())
        return f
        
    def BPF(self, g, f):
        """Performs a Backprojection Filtration (BPF) reconstruction of the projection data, g, and stores the result in f
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        This function take the argument f and returns the same f.
        Returning f is just there for nesting several algorithms.
        This reconstruction only works for parallel-beam data
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data
            f (C contiguous float32 numpy array or torch tensor): volume data
            
        Returns:
            f, the same as the input with the same name
        """
        if self.get_geometry() == 'PARALLEL':
            self.backproject(g,f)
            self.rampFilterVolume(f)
            f *= self.get_FBPscalar()
            return f
        else:
            return None
        
    def sensitivity(self, f=None):
        """Performs a calculation of the sensitivity of a CT geometry, i.e., the backprojection of data that is all ones
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        One can get the same result by backprojecting an array of projection data with all entries equal to one.
        The benefit of this function is that it is faster and uses less memory.
        
        In a volume is provided, the result will be stored there, otherwise a new volume will be allocated
        
        Args:
            f (C contiguous float32 numpy array or torch tensor): (optional argument) volume data to store the result
            
        Returns:
            f, the same as the input
        """
        self.libprojectors.sensitivity.restype = ctypes.c_bool
        if has_torch == True and f is not None and type(f) is torch.Tensor:
            #if f is None:
            #    f = self.allocateVolume(0.0, True)
            self.libprojectors.sensitivity.argtypes = [ctypes.c_void_p, ctypes.c_bool]
            self.set_model()
            self.libprojectors.sensitivity(f.data_ptr(), f.is_cuda == False)
        else:
            if f is None:
                f = self.allocateVolume()
            self.libprojectors.sensitivity.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
            self.set_model()
            self.libprojectors.sensitivity(f, True)
        return f
    
    def windowFOV(self, f):
        """Sets all voxels outside the field of view to zero
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        
        Args:
            f (C contiguous float32 numpy array or torch tensor): volume data to operate on
            
        Returns:
            f, the same as the input
        """
        self.libprojectors.windowFOV.restype = ctypes.c_bool
        if has_torch == True and type(f) is torch.Tensor:
            self.libprojectors.windowFOV.argtypes = [ctypes.c_void_p, ctypes.c_bool]
            self.set_model()
            self.libprojectors.windowFOV(f.data_ptr(), f.is_cuda == False)
        else:
            self.libprojectors.windowFOV.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
            self.set_model()
            self.libprojectors.windowFOV(f, True)
        return f
    
    def rowRangeNeededForBackprojection(self):
        """Calculates the detector rows necessary to reconstruct the current volume specification
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        For anything but cone-beam data this function will return np.array([0, numRows-1]).
        For cone-beam data, the function can be used to inform the user of the only detector rows that
        are necessary to reconstruct the volume.  This can be used to reduce the input data size which can
        be important to speed up calculations or reduce the CPU and/or GPU memory necessary to perform reconstruction.
        
        Returns:
            rowsNeeded, a 2X1 numpy array where the values are the first and last detector row index needed to reconstruct the volume.
        
        """
        rowsNeeded = np.zeros((2,1),dtype=np.float32)
        rowsNeeded[1] = self.get_numRows()-1
        self.libprojectors.rowRangeNeededForBackprojection.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
        self.libprojectors.rowRangeNeededForBackprojection.restype = ctypes.c_bool
        self.set_model()
        self.libprojectors.rowRangeNeededForBackprojection(rowsNeeded)
        return rowsNeeded
    
    ###################################################################################################################
    ###################################################################################################################
    # THIS SECTION PROVIDES ITERATIVE RECONSTRUCTION ALGORITHM THAT USE THE LEAP FORWARD AND BACKPROJECTION OPERATIONS
    ###################################################################################################################
    ###################################################################################################################
    def isAllZeros(self, f):
        if has_torch == True and type(f) is torch.Tensor:
            if torch.count_nonzero(f) == 0:
                return True
            else:
                return False
        else:
            if not np.any(f):
                return True
            else:
                return False
        
    def innerProd(self, x, y, w=None):
        if has_torch == True and type(x) is torch.Tensor:
            if w is None:
                return torch.sum(x*y)
            else:
                return torch.sum(x*y*w)
        else:
            if w is None:
                return np.sum(x*y)
            else:
                return np.sum(x*y*w)
    
    def breakIntoSubsets(self, g, numSubsets):
        if numSubsets <= 0 or len(g.shape) != 3:
            return None
        else:
            g_subsets = []
            for m in range(numSubsets):
                if m == g.shape[0]-1:
                    if has_torch == True and type(g) is torch.Tensor:
                        if g.is_cuda:
                            g_subset = torch.zeros([1, g.shape[1], g.shape[2]], dtype=torch.float32, device=torch.device('cuda:'+str(self.get_gpu())))
                        else:
                            g_subset = torch.zeros([1, g.shape[1], g.shape[2]], dtype=torch.float32)
                    else:
                        g_subset = np.zeros((1,g.shape[1],g.shape[2]),dtype=np.float32)
                    g_subset[0,:,:] = g[m,:,:]
                    g_subsets.append(g_subset)
                else:
                    if has_torch == True and type(g) is torch.Tensor:
                        dim1 = g[m:-1:numSubsets,0,0].shape[0]
                        if g.is_cuda:
                            g_subset = torch.zeros([dim1, g.shape[1], g.shape[2]], dtype=torch.float32, device=torch.device('cuda:'+str(self.get_gpu())))
                        else:
                            g_subset = torch.zeros([dim1, g.shape[1], g.shape[2]], dtype=torch.float32)
                        g_subset[:,:,:] = g[m:-1:numSubsets,:,:]
                    else:
                        g_subset = np.ascontiguousarray(g[m:-1:numSubsets,:,:], np.float32)
                    g_subsets.append(g_subset)
            return g_subsets
    
    def MLEM(self, g, f, numIter):
        """Maximum Likelihood-Expectation Maximization reconstruction
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        This reconstruction algorithms assumes the projection data, g, is Poisson distributed which is the
        correct model for SPECT data.
        CT projection data is not Poisson distributed because of the application of the -log
        
        Args:
            g (C contiguous float32 numpy array): projection data
            f (C contiguous float32 numpy array): volume data
            numIter (int): number of iterations
        
        Returns:
            f, the same as the input with the same name
        """
        #if has_torch == True and type(f) is torch.Tensor:
        #    print('ERROR: Iterative reconstruction algorithms not implemented for torch tensors!')
        #    print('Please convert to numpy array prior to running this algorithm.')
        #    return f
        if self.isAllZeros(f) == True:
            f[:] = 1.0
        else:
            f[f<0.0] = 0.0
 
        Pstar1 = self.sensitivity(self.copyData(f))
        Pstar1[Pstar1==0.0] = 1.0
        d = self.allocateData(f)
        Pd = self.allocateData(g)
        
        for n in range(numIter):
            print('MLEM iteration ' + str(n+1) + ' of ' + str(numIter))
            self.project(Pd,f)
            ind = Pd != 0.0
            Pd[ind] = g[ind]/Pd[ind]
            self.backproject(Pd,d)
            f *= d/Pstar1
        return f
    
    def OSEM(self, g, f, numIter, numSubsets=1):
        """Ordered Subsets-Expectation Maximization reconstruction
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        This reconstruction algorithms assumes the projection data, g, is Poisson distributed which is the
        correct model for SPECT data.
        CT projection data is not Poisson distributed because of the application of the -log
        
        Args:
            g (C contiguous float32 numpy array): projection data
            f (C contiguous float32 numpy array): volume data
            numIter (int): number of iterations
            numSubsets (int): number of subsets
        
        Returns:
            f, the same as the input with the same name
        """
        #if has_torch == True and type(f) is torch.Tensor:
        #    print('ERROR: Iterative reconstruction algorithms not implemented for torch tensors!')
        #    print('Please convert to numpy array prior to running this algorithm.')
        #    return f
        if self.isAllZeros(f) == True:
            f[:] = 1.0
        else:
            f[f<0.0] = 0.0
 
        numSubsets = min(numSubsets, self.get_numAngles())
        #if self.get_geometry() == 'MODULAR' and numSubsets > 1:
        #    print('WARNING: Subsets not yet implemented for modular-beam geometry, setting to 1.')
        #    numSubsets = 1
        if numSubsets <= 1:
            return self.MLEM(g,f,numIter)
        else:
        
            # divide g and phis
            subsetParams = subsetParameters(self, numSubsets)
            g_subsets = self.breakIntoSubsets(g, numSubsets)
                
            d = self.allocateData(f)
            for n in range(numIter):
                print('OSEM iteration ' + str(n+1) + ' of ' + str(numIter))
                for m in range(numSubsets):
                
                    # set angle array
                    #self.set_angles(phis_subsets[m])
                    subsetParams.setSubset(m)
                    
                    Pstar1 = self.sensitivity(self.copyData(f))
                    #Pstar1[Pstar1==0.0] = 1.0

                    Pd = self.allocateData(g_subsets[m])
                    self.project(Pd,f)
                    ind = Pd != 0.0
                    Pd[ind] = g_subsets[m][ind]/Pd[ind]
                    self.backproject(Pd,d)
                    f *= d/Pstar1
            subsetParams.setSubset(-1)
            return f
        
    def SART(self, g, f, numIter, numSubsets=1):
        """Simultaneous Algebraic Reconstruction Technique reconstruction
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        
        Args:
            g (C contiguous float32 numpy array): projection data
            f (C contiguous float32 numpy array): volume data
            numIter (int): number of iterations
            numSubsets (int): number of subsets
        
        Returns:
            f, the same as the input with the same name
        """
        #if has_torch == True and type(f) is torch.Tensor:
        #    print('ERROR: Iterative reconstruction algorithms not implemented for torch tensors!')
        #    print('Please convert to numpy array prior to running this algorithm.')
        #    return f
        numSubsets = min(numSubsets, self.get_numAngles())
        #if self.get_geometry() == 'MODULAR' and numSubsets > 1:
        #    print('WARNING: Subsets not yet implemented for modular-beam geometry, setting to 1.')
        #    numSubsets = 1
        if numSubsets <= 1:
            P1 = self.allocateData(g)
            self.project(P1,self.allocateData(f,1.0))
            P1[P1<=0.0] = 1.0
            
            Pstar1 = self.sensitivity(self.allocateData(f))
            Pstar1[Pstar1<=0.0] = 1.0
            
            Pd = self.allocateData(g)
            d = self.allocateData(f)

            for n in range(numIter):
                print('SART iteration ' + str(n+1) + ' of ' + str(numIter))
                self.project(Pd,f)
                Pd = (g-Pd) / P1
                self.backproject(Pd,d)
                f += 0.9*d / Pstar1
                f[f<0.0] = 0.0
            return f
        else:
            P1 = self.allocateData(g)
            self.project(P1,self.allocateData(f,1.0))
            P1[P1<=0.0] = 1.0
            
            # divide g, P1, and phis
            subsetParams = subsetParameters(self, numSubsets)
            g_subsets = self.breakIntoSubsets(g, numSubsets)
            P1_subsets = self.breakIntoSubsets(P1, numSubsets)
            
            d = self.allocateData(f)
            for n in range(numIter):
                print('SART iteration ' + str(n+1) + ' of ' + str(numIter))
                for m in range(numSubsets):
                
                    # set angle array
                    #self.set_angles(phis_subsets[m])
                    subsetParams.setSubset(m)
                    #self.print_parameters()
                    
                    Pstar1 = self.sensitivity(self.allocateData(f))
                    #Pstar1[Pstar1<=0.0] = 1.0

                    Pd = self.allocateData(g_subsets[m])
                    #print(self.get_numAngles())
                    #print(Pd.shape)
                    self.project(Pd,f)
                    
                    Pd = (g_subsets[m]-Pd) / P1_subsets[m]
                    self.backproject(Pd,d)
                    #print('P1 range: ' + str(np.min(P1_subsets[m])) + ' to ' + str(np.max(P1_subsets[m])))
                    #print('d range: ' + str(np.min(d)) + ' to ' + str(np.max(d)))
                    f += 0.9*d / Pstar1
                    f[f<0.0] = 0.0
            subsetParams.setSubset(-1)
            return f
            
    def ASDPOCS(self, g, f, numIter, numSubsets, numTV, delta=0.0):
        """Adaptive Steepest Descent-Projection onto Convex Subsets reconstruction
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        This function actually implements the iTV reconstruction method which is a slight varition to ASDPOCS
        which we find works slightly better.
        
        Here is the reference
        Ritschl, Ludwig, and Marc Kachelriess.
        "Improved total variation regularized image reconstruction (iTV) applied to clinical CT data."
        In Medical Imaging 2011: Physics of Medical Imaging, vol. 7961, pp. 786-798. SPIE, 2011.
        
        Args:
            g (C contiguous float32 numpy array): projection data
            f (C contiguous float32 numpy array): volume data
            numIter (int): number of iterations
            numSubsets (int): number of subsets
            numTV (int): number of TV diffusion steps
            delta (float): parameter for the Huber-like loss function used in TV
        
        Returns:
            f, the same as the input with the same name
        """
        #if has_torch == True and type(f) is torch.Tensor:
        #    print('ERROR: Iterative reconstruction algorithms not implemented for torch tensors!')
        #    print('Please convert to numpy array prior to running this algorithm.')
        #    return f
        if numTV <= 0:
            return self.SART(g,f,numIter,numSubsets)
        numSubsets = min(numSubsets, self.get_numAngles())
        #if self.get_geometry() == 'MODULAR' and numSubsets > 1:
        #    print('WARNING: Subsets not yet implemented for modular-beam geometry, setting to 1.')
        #    numSubsets = 1
        omega = 0.8
        P1 = self.allocateData(g)
        self.project(P1,self.allocateData(f,1.0))
        P1[P1==0.0] = 1.0

        subsetParams = subsetParameters(self, numSubsets)
        g_subsets = []
        P1_subsets = []
        if numSubsets > 1:
            g_subsets = self.breakIntoSubsets(g, numSubsets)
            P1_subsets = self.breakIntoSubsets(P1, numSubsets)
        else:
            Pstar1 = self.sensitivity(self.allocateData(f))
            Pstar1[Pstar1==0.0] = 1.0
        
        #Pd = self.allocateProjections()
        Pf_minus_g = self.allocateData(g)
        Pf_TV_minus_g = self.allocateData(g)
        d = self.allocateData(f)
        f_TV = self.allocateData(f)

        self.project(Pf_minus_g, f)
        Pf_minus_g -= g
        
        curCost = self.innerProd(Pf_minus_g, Pf_minus_g)
        
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
                    #self.set_angles(phis_subsets[m])
                    subsetParams.setSubset(m)
                    Pstar1 = self.sensitivity(self.allocateData(f))
                    #Pstar1[Pstar1==0.0] = 1.0

                    Pd = self.allocateData(g_subsets[m])
                    self.project(Pd,f)
                    Pd = (g_subsets[m]-Pd) / P1_subsets[m]
                    self.backproject(Pd,d)
                    f += 0.9*d / Pstar1
                    f[f<0.0] = 0.0
                subsetParams.setSubset(-1)

            # Calculate SART error sinogram and calculate cost            
            self.project(Pf_minus_g, f)
            Pf_minus_g = Pf_minus_g - g
            
            epsilon_SART = self.innerProd(Pf_minus_g, Pf_minus_g)

            #'''            
            # TV step(s)
            f_TV[:] = f[:]
            self.diffuse(f_TV, delta, numTV)
            #self.displayVolume(f_TV)
            self.project(Pf_TV_minus_g, f_TV)
            Pf_TV_minus_g = Pf_TV_minus_g - g
            
            epsilon_TV = self.innerProd(Pf_TV_minus_g, Pf_TV_minus_g)
            
            # Combine SART and TV Steps
            temp = self.innerProd(Pf_minus_g, Pf_TV_minus_g)
            a = epsilon_SART - 2.0 * temp + epsilon_TV
            b = temp - epsilon_SART
            c = epsilon_SART - ((1.0 - omega) * epsilon_SART + omega * curCost)
            
            if has_torch == True and type(f) is torch.Tensor:
                a = a.cpu().detach().numpy()
                b = b.cpu().detach().numpy()
                c = c.cpu().detach().numpy()
            
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
            
            curCost = self.innerProd(Pf_minus_g, Pf_minus_g)
            #'''
                        
        return f
        
        
    def LS(self, g, f, numIter, SQS=False, nonnegativityConstraint=True):
        """Least Squares reconstruction
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        This function minimizes the Least Squares cost function using Preconditioned Conjugate Gradient.
        The optional preconditioner is the Separable Quadratic Surrogate for the Hessian of the cost function
        which is given by (P*P1)^-1, where 1 is a volume of all ones, P is forward projection, and P* is backprojection
        
        Args:
            g (C contiguous float32 numpy array): projection data
            f (C contiguous float32 numpy array): volume data
            numIter (int): number of iterations
            SQS (bool): specifies whether or not to use the SQS preconditioner
        
        Returns:
            f, the same as the input with the same name
        """
        return self.RWLS(g, f, numIter, 0.0, 0.0, 1.0, SQS, nonnegativityConstraint)
        
    def WLS(self, g, f, numIter, W=None, SQS=False, nonnegativityConstraint=True):
        """Weighted Least Squares reconstruction
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        This function minimizes the Weighted Least Squares cost function using Preconditioned Conjugate Gradient.
        The optional preconditioner is the Separable Quadratic Surrogate for the Hessian of the cost function
        which is given by (P*WP1)^-1, where 1 is a volume of all ones, W are the weights, P is forward projection, and P* is backprojection
        
        Args:
            g (C contiguous float32 numpy array): projection data
            f (C contiguous float32 numpy array): volume data
            numIter (int): number of iterations
            W (C contiguous float32 numpy array): weights, should be the same size as g, if not given, W=np.exp(-g)
            SQS (bool): specifies whether or not to use the SQS preconditioner
        
        Returns:
            f, the same as the input with the same name
        """
        return self.RWLS(g, f, numIter, 0.0, 0.0, W, SQS, nonnegativityConstraint)
        
    def RLS(self, g, f, numIter, delta=0.0, beta=0.0, SQS=False, nonnegativityConstraint=True):
        """Regularized Least Squares reconstruction
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        This function minimizes the Regularized Least Squares cost function using Preconditioned Conjugate Gradient.
        The optional preconditioner is the Separable Quadratic Surrogate for the Hessian of the cost function
        which is given by (P*P1)^-1, where 1 is a volume of all ones, P is forward projection, and P* is backprojection
        
        Args:
            g (C contiguous float32 numpy array): projection data
            f (C contiguous float32 numpy array): volume data
            numIter (int): number of iterations
            delta (float): parameter for the Huber-like loss function used in TV
            beta (float): regularization strength
            SQS (bool): specifies whether or not to use the SQS preconditioner
        
        Returns:
            f, the same as the input with the same name
        """
        return self.RWLS(g, f, numIter, delta, beta, 1.0, SQS, nonnegativityConstraint)
       
    def RWLS(self, g, f, numIter, delta=0.0, beta=0.0, W=None, SQS=False, nonnegativityConstraint=True):
        """Regularized Weighted Least Squares reconstruction
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        This function minimizes the Regularized Weighted Least Squares cost function using Preconditioned Conjugate Gradient.
        The optional preconditioner is the Separable Quadratic Surrogate for the Hessian of the cost function
        which is given by (P*WP1)^-1, where 1 is a volume of all ones, W are the weights, P is forward projection, and P* is backprojection
        
        Args:
            g (C contiguous float32 numpy array): projection data
            f (C contiguous float32 numpy array): volume data
            numIter (int): number of iterations
            delta (float): parameter for the Huber-like loss function used in TV
            beta (float): regularization strength
            W (C contiguous float32 numpy array): weights, should be the same size as g, if not given, W=np.exp(-g)
            SQS (bool): specifies whether or not to use the SQS preconditioner
        
        Returns:
            f, the same as the input with the same name
        """
        #if has_torch == True and type(f) is torch.Tensor:
        #    print('ERROR: RWLS reconstruction algorithms not implemented for torch tensors!')
        #    print('Please convert to numpy array prior to running this algorithm.')
        #    return f
        conjGradRestart = 50
        if W is None:
            W = self.copyData(g)
            if has_torch == True and type(g) is torch.Tensor:
                W = torch.exp(-W)
            else:
                W = np.exp(-W)
        Pf = self.copyData(g)
        if self.isAllZeros(f) == False:
            # fix scaling
            if nonnegativityConstraint:
                f[f<0.0] = 0.0
            self.project(Pf,f)
            Pf_dot_Pf = self.innerProd(Pf,Pf)
            g_dot_Pf = self.innerProd(g,Pf)
            if Pf_dot_Pf > 0.0 and g_dot_Pf > 0.0:
                f *= g_dot_Pf / Pf_dot_Pf
                Pf *= g_dot_Pf / Pf_dot_Pf
        else:
            Pf[:] = 0.0
        Pf_minus_g = Pf
        Pf_minus_g -= g
        
        grad = self.allocateData(f)
        u = self.allocateData(f)
        Pu = self.allocateData(g)
        
        d = self.allocateData(f)
        Pd = self.allocateData(g)
        
        grad_old_dot_grad_old = 0.0
        grad_old = self.allocateData(f)
        
        if SQS == True:
            # Calculate the SQS preconditioner
            # Reuse some of the memory allocated above
            #Q = 1.0 / P*WP1
            Q = self.allocateData(f)
            Q[:] = 1.0
            self.project(Pu,Q)
            Pu *= W
            self.backproject(Pu,Q)
            Q[Q==0.0] = 1.0
            Q = 1.0 / Q
        else:
            Q = 1.0
        
        for n in range(numIter):
            print('RWLS iteration ' + str(n+1) + ' of ' + str(numIter))
            #WPf_minus_g = self.copyData(Pf_minus_g)
            WPf_minus_g = Pf_minus_g
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
                gamma = (self.innerProd(u,grad) - self.innerProd(u,grad_old)) / grad_old_dot_grad_old

                d = u + gamma*d
                Pd = Pu + gamma*Pd

                if self.innerProd(d,grad) <= 0.0:
                    print('\tRLWS-CG: CG descent condition violated, must use GD descent direction')
                    d[:] = u[:]
                    Pd[:] = Pu[:]
            
            grad_old_dot_grad_old = self.innerProd(u,grad)
            grad_old[:] = grad[:]
            
            stepSize = self.RWLSstepSize(f, grad, d, Pd, W, delta, beta)
            if stepSize <= 0.0:
                print('invalid step size; quitting!')
                break
            
            f[:] = f[:] - stepSize*d[:]
            if nonnegativityConstraint:
                f[f<0.0] = 0.0
                self.project(Pf,f)
            else:
                Pf[:] = Pf[:] - stepSize*Pd[:]
            Pf_minus_g[:] = Pf[:] - g[:]
                
        return f

    def RWLSstepSize(self, f, grad, d, Pd, W, delta, beta):
        """Calculates the step size for an RWLS iteration

        Args:
            f (C contiguous float32 numpy array): volume data
            grad (C contiguous float32 numpy array): gradient of the RWLS cost function
            d (C contiguous float32 numpy array): descent direction of the RWLS cost function
            Pd (C contiguous float32 numpy array): forward projection of d
            W (C contiguous float32 numpy array): weights, should be the same size as g, if not given, W=np.exp(-g)
            delta (float): parameter for the Huber-like loss function used in TV
            beta (float): regularization strength
        
        Returns:
            step size (float)
        """
        num = self.innerProd(d,grad)
        if W is not None:
            denomA = self.innerProd(Pd,Pd,W)
        else:
            denomA = self.innerProd(Pd,Pd)
        denomB = 0.0;
        if beta > 0.0:
            denomB = self.TVquadForm(f, d, delta, beta)
            #print('denomB = ' + str(denomA))
        denom = denomA + denomB
        if has_torch == True and type(denom) is torch.Tensor:
            denom = denom.cpu().detach().numpy()
            num = num.cpu().detach().numpy()

        stepSize = 0.0
        if np.abs(denom) > 1.0e-16:
            stepSize = num / denom
        print('\tlambda = ' + str(stepSize))
        return stepSize
        
    def RDLS(self, g, f, numIter, delta=0.0, beta=0.0, preconditionerFWHM=1.0, nonnegativityConstraint=False):
        """Regularized Derivative Least Squares reconstruction
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        This function minimizes the Regularized Derivative Least Squares cost function using Preconditioned Conjugate Gradient.
        The optional preconditioner is a 2D blurring for each z-slice
        
        Args:
            g (C contiguous float32 numpy array): projection data
            f (C contiguous float32 numpy array): volume data
            numIter (int): number of iterations
            delta (float): parameter for the Huber-like loss function used in TV
            beta (float): regularization strength
            preconditionerFWHM (float): specifies the FWHM of the blur preconditioner
        
        Returns:
            f, the same as the input with the same name
        """
        #if has_torch == True and type(f) is torch.Tensor:
        #    print('ERROR: RDLS reconstruction algorithms not implemented for torch tensors!')
        #    print('Please convert to numpy array prior to running this algorithm.')
        #    return f
        conjGradRestart = 50
        Pf = self.copyData(g)
        if self.isAllZeros(f) == False:
            # fix scaling
            if nonnegativityConstraint:
                f[f<0.0] = 0.0
            self.project(Pf,f)
            Pf_dot_Pf = self.innerProd(Pf,Pf)
            g_dot_Pf = self.innerProd(g,Pf)
            if Pf_dot_Pf > 0.0 and g_dot_Pf > 0.0:
                f *= g_dot_Pf / Pf_dot_Pf
                Pf *= g_dot_Pf / Pf_dot_Pf
        else:
            Pf[:] = 0.0
        Pf_minus_g = Pf
        Pf_minus_g -= g
        
        LPf_minus_g = self.copyData(Pf)
        
        grad = self.allocateData(f)
        u = self.allocateData(f)
        Pu = self.allocateData(g)
        
        d = self.allocateData(f)
        Pd = self.allocateData(g)
        
        grad_old_dot_grad_old = 0.0
        grad_old = self.allocateData(f)
                
        for n in range(numIter):
            print('RDLS iteration ' + str(n+1) + ' of ' + str(numIter))
            LPf_minus_g[:] = Pf_minus_g[:]
            self.Laplacian(LPf_minus_g)
            LPf_minus_g *= -1.0
            self.backproject(LPf_minus_g, grad)
            if beta > 0.0:
                Sf1 = self.TVgradient(f, delta, beta)
                grad += Sf1

            u[:] = grad[:]
            if preconditionerFWHM > 1.0:
                self.BlurFilter2D(u,preconditionerFWHM)
            self.project(Pu, u)
            
            if n == 0 or (n % conjGradRestart) == 0:
                d[:] = u[:]
                Pd[:] = Pu[:]
            else:
                gamma = (self.innerProd(u,grad) - self.innerProd(u,grad_old)) / grad_old_dot_grad_old

                d = u + gamma*d
                Pd = Pu + gamma*Pd

                if self.innerProd(d,grad) <= 0.0:
                    print('\tRLDS-CG: CG descent condition violated, must use GD descent direction')
                    d[:] = u[:]
                    Pd[:] = Pu[:]
            
            grad_old_dot_grad_old = self.innerProd(u,grad)
            grad_old[:] = grad[:]
            
            stepSize = self.RDLSstepSize(f, grad, d, Pd, delta, beta)
            #if stepSize <= 0.0:
            #    print('invalid step size; quitting!')
            #    break
            
            f[:] = f[:] - stepSize*d[:]
            if nonnegativityConstraint:
                f[f<0.0] = 0.0
                self.project(Pf,f)
                Pf_minus_g[:] = Pf[:] - g[:]
            else:
                Pf_minus_g[:] = Pf_minus_g[:] - stepSize*Pd[:]
        return f

    def RDLSstepSize(self, f, grad, d, Pd, delta, beta):
        """Calculates the step size for an RDLS iteration

        Args:
            f (C contiguous float32 numpy array): volume data
            grad (C contiguous float32 numpy array): gradient of the RWLS cost function
            d (C contiguous float32 numpy array): descent direction of the RWLS cost function
            Pd (C contiguous float32 numpy array): forward projection of d
            delta (float): parameter for the Huber-like loss function used in TV
            beta (float): regularization strength
        
        Returns:
            step size (float)
        """
        num = self.innerProd(d,grad)
        LPd = self.copyData(Pd)
        self.Laplacian(LPd)
        LPd *= -1.0
        denomA = self.innerProd(LPd,Pd)
        denomB = 0.0;
        if beta > 0.0:
            denomB = self.TVquadForm(f, d, delta, beta)
            #print('denomB = ' + str(denomA))
        denom = denomA + denomB

        if has_torch == True and type(denom) is torch.Tensor:
            denom = denom.cpu().detach().numpy()
            num = num.cpu().detach().numpy()

        stepSize = 0.0
        if np.abs(denom) > 1.0e-16:
            stepSize = num / denom
        print('\tlambda = ' + str(stepSize))
        return stepSize

    def MLTR(self, g, f, numIter, numSubsets=1, delta=0.0, beta=0.0):
        """Maximum Likelihood Transmission reconstruction
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        This function maximizes the Maximum Likelihood function of CT transmission data which assumes a Poisson noise model.
        This algorithm best models the noise for very low transmission/ low count rate data.
        
        Args:
            g (C contiguous float32 numpy array): projection data
            f (C contiguous float32 numpy array): volume data
            numIter (int): number of iterations
            numSubsets (int): number of subsets
            delta (float): parameter for the Huber-like loss function used in TV
            beta (float): regularization strength
        
        Returns:
            f, the same as the input with the same name
        """
        #if has_torch == True and type(f) is torch.Tensor:
        #    print('ERROR: MLTR reconstruction algorithms not implemented for torch tensors!')
        #    print('Please convert to numpy array prior to running this algorithm.')
        #    return f
        beta = max(0.0, beta/float(numSubsets))
    
        if has_torch == True and type(g) is torch.Tensor:
            t = torch.exp(-g)
        else:
            t = np.exp(-g)

        if self.isAllZeros(f) == False:
            f[f<=0.0] = 0.0
        
        d = self.allocateData(f)
        
        d[:] = 1.0
        P1 = self.allocateData(g)
        self.project(P1,d)
        P1[P1<=0.0] = 1.0

        SQS = self.allocateData(f)
        if numSubsets <= 1:
            Pf = self.allocateData(g)
            
            transDiff = self.allocateData(g)
            for n in range(numIter):
                print('ML-TR iteration ' + str(n+1) + ' of ' + str(numIter))
                self.project(Pf,f)
                
                if has_torch == True and type(g) is torch.Tensor:
                    transDiff[:] = torch.exp(-Pf[:])
                else:
                    transDiff[:] = np.exp(-Pf[:])
                
                transDiff[:] = transDiff[:] * P1[:]
                self.backproject(transDiff, SQS)
                SQS[SQS<=0.0] = 1.0
                SQS[:] = 1.0 / SQS[:]
                
                transDiff[:] = transDiff[:]/P1[:] - t[:]
                self.backproject(transDiff, d)
                
                # Regularizer and divide by SQS
                stepMultiplier = 1.0
                if beta > 0.0:
                    Sf1 = self.TVgradient(f, delta, beta)
                    d[:] -= Sf1[:]
                    grad_dot_descent = self.innerProd(d,d,SQS)
                
                    d[:] = d[:] * SQS[:]
                
                    stepMultiplier = grad_dot_descent / (grad_dot_descent + self.TVquadForm(f,d, delta, beta))
                else:
                    d[:] = d[:] * SQS[:]
                
                f[:] = f[:] + stepMultiplier*d[:]
                f[f<0.0] = 0.0
                
        else:
            subsetParams = subsetParameters(self, numSubsets)
            t_subsets = self.breakIntoSubsets(t, numSubsets)
            P1_subsets = self.breakIntoSubsets(P1, numSubsets)
            for n in range(numIter):
                print('ML-TR iteration ' + str(n+1) + ' of ' + str(numIter))
                for m in range(numSubsets):
                    subsetParams.setSubset(m)
                    transDiff = self.allocateData(t_subsets[m])
                    Pf = self.allocateData(t_subsets[m])
                    
                    self.project(Pf,f)
                    
                    if has_torch == True and type(g) is torch.Tensor:
                        transDiff[:] = torch.exp(-Pf[:])
                    else:
                        transDiff[:] = np.exp(-Pf[:])
                    
                    transDiff[:] = transDiff[:] * P1_subsets[m][:]
                    self.backproject(transDiff, SQS)
                    SQS[SQS<=0.0] = 1.0
                    SQS[:] = 1.0 / SQS[:]
                    
                    transDiff[:] = transDiff[:]/P1_subsets[m][:] - t_subsets[m][:]
                    self.backproject(transDiff, d)
                    
                    # Regularizer and divide by SQS
                    stepMultiplier = 1.0
                    if beta > 0.0:
                        Sf1 = self.TVgradient(f, delta, beta)
                        d[:] -= Sf1[:]
                        grad_dot_descent = self.innerProd(d,d,SQS)
                    
                        d[:] = d[:] * SQS[:]
                    
                        stepMultiplier = grad_dot_descent / (grad_dot_descent + self.TVquadForm(f,d, delta, beta))
                    else:
                        d[:] = d[:] * SQS[:]
                    
                    f[:] = f[:] + stepMultiplier*d[:]
                    f[f<0.0] = 0.0
        
            subsetParams.setSubset(-1)
        
        if has_torch == True and type(g) is torch.Tensor:
            g = -torch.log(t)
        else:
            g = -np.log(t)
        
        return f
            

    ###################################################################################################################
    ###################################################################################################################
    # THIS SECTION OF FUNCTIONS EXECUTE LEAP'S GPU DENOISING FILTERS
    ###################################################################################################################
    ###################################################################################################################
    def applyTransferFunction(self, x,  LUT, sampleRate, firstSample=0.0):
        """Applies a transfer function to arbitrary 3D data, i.e., x = LUT(x)
        
        Args:
            x (C contiguous float32 numpy array or torch tensor): 3D data (input and output)
            LUT (C contiguous float32 numpy array or torch tensor): lookup table with transfer function values
            sampleRate (float): the step size between samples
            firstSample (float): the value of the first sample in the lookup table

        Returns:            
            true if operation  was sucessful, false otherwise
        """
        #bool applyTransferFunction(float* x, int N_1, int N_2, int N_3, float* LUT, float firstSample, float sampleRate, int numSamples, bool data_on_cpu)
        self.libprojectors.applyTransferFunction.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(x) is torch.Tensor:
            self.libprojectors.applyTransferFunction.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_bool]
            return self.libprojectors.applyTransferFunction(x.data_ptr(), x.shape[0], x.shape[1], x.shape[2], LUT.data_ptr(), firstSample, sampleRate, LUT.size, f.is_cuda == False)
        else:
            self.libprojectors.applyTransferFunction.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_bool]
            return self.libprojectors.applyTransferFunction(x, x.shape[0], x.shape[1], x.shape[2], LUT, firstSample, sampleRate, LUT.size, True)
    
    def BlurFilter(self, f, FWHM=2.0):
        """Applies a blurring filter to the provided numpy array
        
        The provided input does not have to be projection or volume data. It can be any 3D numpy array of any size
        The filter is given by cos^2(pi/(2*FWHM) * i), i = -ceil(FWHM), ..., ceil(FWHM)
        This filter is very simular to a Gaussian filter, but is a FIR
        
        Args:
            f (C contiguous float32 numpy array): numpy array to smooth
            FWHM (float): the full width at half maximum (in number of pixels) of the filter
        
        Returns:
            f, the same as the input
        """
        #bool BlurFilter(float* f, int, int, int, float FWHM);
        self.libprojectors.BlurFilter.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(f) is torch.Tensor:
            self.libprojectors.BlurFilter.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_bool]
            return self.libprojectors.BlurFilter(f.data_ptr(), f.shape[0], f.shape[1], f.shape[2], FWHM, f.is_cuda == False)
        else:
            self.libprojectors.BlurFilter.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_bool]
            return self.libprojectors.BlurFilter(f, f.shape[0], f.shape[1], f.shape[2], FWHM, True)
            
    def BlurFilter2D(self, f, FWHM=2.0):
        """Applies a 2D blurring filter to the provided numpy array
        
        The provided input does not have to be projection or volume data. It can be any 3D numpy array of any size
        The filter is given by cos^2(pi/(2*FWHM) * i), i = -ceil(FWHM), ..., ceil(FWHM)
        This filter is very simular to a Gaussian filter, but is a FIR
        
        Args:
            f (C contiguous float32 numpy array): numpy array to smooth
            FWHM (float): the full width at half maximum (in number of pixels) of the filter
        
        Returns:
            f, the same as the input
        """
        #bool BlurFilter2D(float* f, int, int, int, float FWHM);
        self.libprojectors.BlurFilter2D.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(f) is torch.Tensor:
            self.libprojectors.BlurFilter2D.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_bool]
            return self.libprojectors.BlurFilter2D(f.data_ptr(), f.shape[0], f.shape[1], f.shape[2], FWHM, f.is_cuda == False)
        else:
            self.libprojectors.BlurFilter2D.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_bool]
            return self.libprojectors.BlurFilter2D(f, f.shape[0], f.shape[1], f.shape[2], FWHM, True)
    
    def MedianFilter(self, f, threshold=0.0):
        """Applies a thresholded 3D median filter (3x3x3) to the provided numpy array
        
        The provided input does not have to be projection or volume data. It can be any 3D numpy array of any size
        This algorithm performs a 3D (3x3x3) median around each data value and then replaces this value only if
        |original value - median value| >= threshold*|median value|
        Note that if threshold is zero, then this is simply a median filter
        
        Args:
            f (C contiguous float32 numpy array): numpy array to smooth
            threshold (float): the threshold of whether to use the filtered value or not
        
        Returns:
            f, the same as the input
        """
        #bool MedianFilter(float* f, int, int, int, float threshold);
        self.libprojectors.MedianFilter.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(f) is torch.Tensor:
            self.libprojectors.MedianFilter.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_bool]
            return self.libprojectors.MedianFilter(f.data_ptr(), f.shape[0], f.shape[1], f.shape[2], threshold, f.is_cuda == False)
        else:
            self.libprojectors.MedianFilter.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_bool]
            return self.libprojectors.MedianFilter(f, f.shape[0], f.shape[1], f.shape[2], threshold, True)
            
    def MedianFilter2D(self, f, threshold=0.0, windowSize=3):
        """Applies a thresholded 2D median filter (windowSize x windowSize) to the provided numpy array
        
        The provided input does not have to be projection or volume data. It can be any 3D numpy array of any size
        This algorithm performs a 2D (windowSize x windowSize) median around each data value and then replaces this value only if
        |original value - median value| >= threshold*|median value|
        Note that if threshold is zero, then this is simply a median filter
        
        Args:
            f (C contiguous float32 numpy array): numpy array to smooth
            threshold (float): the threshold of whether to use the filtered value or not
        
        Returns:
            f, the same as the input
        """
        #bool MedianFilter2D(float* f, int, int, int, float threshold, int windowSize);
        self.libprojectors.MedianFilter2D.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(f) is torch.Tensor:
            self.libprojectors.MedianFilter2D.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_bool]
            return self.libprojectors.MedianFilter2D(f.data_ptr(), f.shape[0], f.shape[1], f.shape[2], threshold, windowSize, f.is_cuda == False)
        else:
            self.libprojectors.MedianFilter2D.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_bool]
            return self.libprojectors.MedianFilter2D(f, f.shape[0], f.shape[1], f.shape[2], threshold, windowSize, True)
    
    def TVcost(self, f, delta, beta=0.0):
        """Calculates the anisotropic Total Variation (TV) functional, i.e., cost of the provided numpy array
        
        The provided input does not have to be projection or volume data. It can be any 3D numpy array of any size
        
        Args:
            f (C contiguous float32 numpy array): 3D numpy array
            delta (float): parameter for the Huber-like loss function used in TV
            beta (float): TV multiplier (sometimes called the regularizaion strength)
        
        Returns:
            TV functional value
        """
        #float TVcost(float* f, int N_1, int N_2, int N_3, float delta, float beta);
        self.libprojectors.TVcost.restype = ctypes.c_float
        self.set_model()
        if has_torch == True and type(f) is torch.Tensor:
            self.libprojectors.TVcost.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_bool]
            return self.libprojectors.TVcost(f.data_ptr(), f.shape[0], f.shape[1], f.shape[2], delta, beta, f.is_cuda == False)
        else:
            self.libprojectors.TVcost.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_bool]
            return self.libprojectors.TVcost(f, f.shape[0], f.shape[1], f.shape[2], delta, beta, True)
        
    def TVgradient(self, f, delta, beta=0.0):
        """Calculates the gradient of the anisotropic Total Variation (TV) functional of the provided numpy array
        
        The provided input does not have to be projection or volume data. It can be any 3D numpy array of any size
        
        Args:
            f (C contiguous float32 numpy array): 3D numpy array
            delta (float): parameter for the Huber-like loss function used in TV
            beta (float): TV multiplier (sometimes called the regularizaion strength)
        
        Returns:
            Df (C contiguous float32 numpy array): the gradient of the TV functional applied to the input
        """
        #bool TVgradient(float* f, float* Df, int N_1, int N_2, int N_3, float delta, float beta);
        self.libprojectors.TVgradient.restype = ctypes.c_bool
        
        if has_torch == True and type(f) is torch.Tensor:
            Df = f.clone()
            self.set_model()
            self.libprojectors.TVgradient.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_bool]
            self.libprojectors.TVgradient(f.data_ptr(), Df.data_ptr(), f.shape[0], f.shape[1], f.shape[2], delta, beta, f.is_cuda == False)
            return Df
        else:
            Df = np.ascontiguousarray(np.zeros(f.shape,dtype=np.float32), dtype=np.float32)
            self.set_model()
            self.libprojectors.TVgradient.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_bool]
            self.libprojectors.TVgradient(f, Df, f.shape[0], f.shape[1], f.shape[2], delta, beta, True)
            return Df
    
    def TVquadForm(self, f, d, delta, beta=0.0):
        """Calculates the quadratic form of the anisotropic Total Variation (TV) functional of the provided numpy arrays
        
        The provided inputs does not have to be projection or volume data. It can be any 3D numpy array of any size
        This function calculates the following inner product <d, R''(f)d>, where R'' is the Hessian of the TV functional
        The quadraitc surrogate is used here, so this function can be used to calculate the step size of a cost function
        that includes a TV regularization term.
        See the same  cost in the diffuse function below for an example of its usage
        
        Args:
            f (C contiguous float32 numpy array): 3D numpy array
            d (C contiguous float32 numpy array): 3D numpy array
            delta (float): parameter for the Huber-like loss function used in TV
            beta (float): TV multiplier (sometimes called the regularizaion strength)
        
        Returns:
            Df (C contiguous float32 numpy array): the gradient of the TV functional applied to the input
        """
        #float TVquadForm(float* f, float* d, int N_1, int N_2, int N_3, float delta, float beta);
        self.libprojectors.TVquadForm.restype = ctypes.c_float
        self.set_model()
        if has_torch == True and type(f) is torch.Tensor:
            self.libprojectors.TVquadForm.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_bool]
            return self.libprojectors.TVquadForm(f.data_ptr(), d.data_ptr(), f.shape[0], f.shape[1], f.shape[2], delta, beta, f.is_cuda == False)
        else:
            self.libprojectors.TVquadForm.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_bool]
            return self.libprojectors.TVquadForm(f, d, f.shape[0], f.shape[1], f.shape[2], delta, beta, True)
        
    def diffuse(self, f, delta, numIter):
        """Performs anisotropic Total Variation (TV) smoothing to the provided 3D numpy array
        
        The provided inputs does not have to be projection or volume data. It can be any 3D numpy array of any size
        This function performs a specifies number of iterations of minimizing the aTV functional using gradient descent
        
        Args:
            f (C contiguous float32 numpy array): 3D numpy array
            delta (float): parameter for the Huber-like loss function used in TV
            numIter (int): number of iterations
        
        Returns:
            f, the same array as the input denoised
        """
        self.libprojectors.Diffuse.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(f) is torch.Tensor:
            self.libprojectors.Diffuse.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_bool]
            self.libprojectors.Diffuse(f.data_ptr(), f.shape[0], f.shape[1], f.shape[2], delta, numIter, f.is_cuda == False)
        else:
            self.libprojectors.Diffuse.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_bool]
            self.libprojectors.Diffuse(f, f.shape[0], f.shape[1], f.shape[2], delta, numIter, True)
        return f
        ''' Here is equivalent code to run this algorithm using the TV functions above
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
        """Set which GPU to use, use -1 to do CPU calculations"""
        return self.set_GPU(which)
    
    def set_GPU(self, which):
        """Set which GPU to use, use -1 to do CPU calculations"""
        self.libprojectors.set_GPU.argtypes = [ctypes.c_int]
        self.libprojectors.set_GPU.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_GPU(which)
        
    def set_gpus(self, listOfGPUs):
        """Set which GPUs to use when doing multi-GPU calculations"""
        return self.set_GPUs(listOfGPUs)
        
    def set_GPUs(self, listOfGPUs):
        """Set which GPUs to use when doing multi-GPU calculations"""
        self.libprojectors.set_GPUs.argtypes = [ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_int]
        self.libprojectors.set_GPUs.restype = ctypes.c_bool
        listOfGPUs = np.ascontiguousarray(listOfGPUs, dtype=np.int32)
        self.set_model()
        return self.libprojectors.set_GPUs(listOfGPUs, int(listOfGPUs.size))
        
    def get_gpu(self):
        """Get the index of the primary GPU that is being used"""
        return self.get_GPU()
        
    def get_GPU(self):
        """Get the index of the primary GPU that is being used"""
        self.libprojectors.get_GPU.restype = ctypes.c_int
        self.set_model()
        return self.libprojectors.get_GPU()
        
    def set_diameterFOV(self, d):
        """Set the diameterFOV parameter"""
        self.libprojectors.set_rFOV.argtypes = [ctypes.c_float]
        self.libprojectors.set_rFOV.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_rFOV(0.5*d)
        
    def set_truncatedScan(self, aFlag):
        """Set the truncatedScan parameter"""
        self.libprojectors.set_truncatedScan.argtypes = [ctypes.c_bool]
        self.libprojectors.set_truncatedScan.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_truncatedScan(aFlag)
        
    def set_offsetScan(self, aFlag):
        """Set the offsetScan parameter"""
        self.libprojectors.set_offsetScan.argtypes = [ctypes.c_bool]
        self.libprojectors.set_offsetScan.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_offsetScan(aFlag)
    
    def set_axisOfSymmetry(self,val):
        """Set the axisOfSymmetry parameter"""
        self.libprojectors.set_axisOfSymmetry.argtypes = [ctypes.c_float]
        self.libprojectors.set_axisOfSymmetry.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_axisOfSymmetry(val)
        
    def clear_axisOfSymmetry(self):
        """Clears the axisOfSymmetry parameter (revert back to voxelized volume models)"""
        self.libprojectors.clear_axisOfSymmetry.argtypes = []
        self.libprojectors.clear_axisOfSymmetry.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.clear_axisOfSymmetry()
        
    def set_projector(self,which):
        """Set which projector model to use (deprecated)"""
        self.libprojectors.set_projector.argtypes = [ctypes.c_int]
        self.libprojectors.set_projector.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_projector(which)
        
    def set_rampFilter(self,which):
        """Set the ramp filter to use: 0, 2, 4, 6, 8, or 10"""
        self.libprojectors.set_rampID.argtypes = [ctypes.c_int]
        self.libprojectors.set_rampID.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_rampID(which)
        
    def set_attenuationMap(self, mu):
        """Set the voxelized attenuation map for Attenuated Radon Transform calculations"""
        self.libprojectors.set_attenuationMap.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(mu) is torch.Tensor:
            self.libprojectors.set_attenuationMap.argtypes = [ctypes.c_void_p]
            return self.libprojectors.set_attenuationMap(mu.data_ptr())
        else:
            self.libprojectors.set_attenuationMap.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
            return self.libprojectors.set_attenuationMap(mu)
            
    def muSpecified(self):
        """Returns a boolean of whether or not the attenuation model has been set for the Attenuated Radon Transform
        
        Returns:
            true if (muCoeff != 0 and muRadius > 0) or mu != NULL, false otherwise
        """
        self.set_model()
        self.libprojectors.muSpecified.restype = ctypes.c_bool
        return self.libprojectors.muSpecified()
        
    def flipAttenuationMapSign(self):
        """Changes the sign of the attenuation model has been set for the Attenuated Radon Transform

        WARNING: does not work if mu is stored on a GPU!
        
        If muCoeff != 0.0, muCoeff *= -1
        if mu != NULL, mu *= -1
        """
        self.set_model()
        self.libprojectors.flipAttenuationMapSign.restype = ctypes.c_bool
        self.libprojectors.flipAttenuationMapSign.argtypes = [ctypes.c_bool]
        return self.libprojectors.flipAttenuationMapSign(True)
        
    def set_cylindircalAttenuationMap(self, c, R):
        """Set the parameters for a cylindrical attenuation map for Attenuated Radon Transform calculations"""
        self.libprojectors.set_cylindircalAttenuationMap.argtypes = [ctypes.c_float, ctypes.c_float]
        self.libprojectors.set_cylindircalAttenuationMap.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_cylindircalAttenuationMap(c, R)
        
    def clear_attenuationMap(self):
        """Clears the attenuation map parameters and reverts back to using the X-ray Transform"""
        self.libprojectors.clear_attenuationMap.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.clear_attenuationMap()
        
    def get_angles(self):
        """Get a numpy array of the projection angles"""
        phis = np.ascontiguousarray(np.zeros(self.get_numAngles()).astype(np.float32), dtype=np.float32)
        self.libprojectors.get_angles.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
        self.libprojectors.get_angles.restype = ctypes.c_bool
        self.set_model()
        self.libprojectors.get_angles(phis)
        return phis
        
    def set_angles(self,phis):
        """Set the projection angles"""
        self.libprojectors.set_angles.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int]
        self.libprojectors.set_angles.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_angles(phis, int(phis.size))

    ###################################################################################################################
    ###################################################################################################################
    # THIS SECTION RETRIEVES THE VARIOUS CT GEOMETRY AND VOLUME PARAMETERS THAT HAS BEEN SET IN LEAP
    ###################################################################################################################
    ###################################################################################################################
    def get_geometry(self):
        """Get the CT geometry type"""
        self.libprojectors.get_geometry.restype = ctypes.c_int
        self.set_model()
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
        """Get the sod parameter"""
        self.libprojectors.get_sod.restype = ctypes.c_float
        self.set_model()
        return self.libprojectors.get_sod()
        
    def get_sdd(self):
        """Get the sdd parameter"""
        self.libprojectors.get_sdd.restype = ctypes.c_float
        self.set_model()
        return self.libprojectors.get_sdd()
        
    def get_helicalPitch(self):
        """Get the helicalPitch parameter"""
        self.libprojectors.get_helicalPitch.restype = ctypes.c_float
        self.set_model()
        return self.libprojectors.get_helicalPitch()
        
    def get_z_source_offset(self):
        """Get the source position z-coordinate for the first projection"""
        self.libprojectors.get_z_source_offset.restype = ctypes.c_float
        self.set_model()
        return self.libprojectors.get_z_source_offset()
    
    def get_numAngles(self):
        """Get the numAngles parameter"""
        self.set_model()
        return self.libprojectors.get_numAngles()
        
    def get_numRows(self):
        """Get the numRows parameter"""
        self.set_model()
        return self.libprojectors.get_numRows()
        
    def get_numCols(self):
        """Get the numCols parameter"""
        self.set_model()
        return self.libprojectors.get_numCols()
        
    def get_pixelHeight(self):
        """Get the pixelHeight parameter"""
        self.libprojectors.get_pixelHeight.restype = ctypes.c_float
        self.set_model()
        return self.libprojectors.get_pixelHeight()
        
    def get_pixelWidth(self):
        """Get the pixelWidth parameter"""
        self.libprojectors.get_pixelWidth.restype = ctypes.c_float
        self.set_model()
        return self.libprojectors.get_pixelWidth()
        
    def get_centerRow(self):
        """Get the centerRow parameter"""
        self.libprojectors.get_centerRow.restype = ctypes.c_float
        self.set_model()
        return self.libprojectors.get_centerRow()
        
    def get_centerCol(self):
        """Get the centerCol parameter"""
        self.libprojectors.get_centerCol.restype = ctypes.c_float
        self.set_model()
        return self.libprojectors.get_centerCol()
        
    def get_tau(self):
        """Get the tau parameter"""
        self.libprojectors.get_tau.restype = ctypes.c_float
        self.set_model()
        return self.libprojectors.get_tau()
        
    def get_sourcePositions(self):
        """Get the sourcePositions parameter (modular-beam only)"""
        #bool get_sourcePositions(float*);
        if self.get_numAngles() <= 0:
            return None
        else:
            x = np.ascontiguousarray(np.zeros((self.get_numAngles(),3)).astype(np.float32), dtype=np.float32)
            self.libprojectors.get_sourcePositions.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
            self.libprojectors.get_sourcePositions.restype = ctypes.c_bool
            self.set_model()
            self.libprojectors.get_sourcePositions(x)
            return x
        
    def get_moduleCenters(self):
        """Get the moduleCenters parameter (modular-beam only)"""
	    #bool get_moduleCenters(float*);
        if self.get_numAngles() <= 0:
            return None
        else:
            x = np.ascontiguousarray(np.zeros((self.get_numAngles(),3)).astype(np.float32), dtype=np.float32)
            self.libprojectors.get_moduleCenters.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
            self.libprojectors.get_moduleCenters.restype = ctypes.c_bool
            self.set_model()
            self.libprojectors.get_moduleCenters(x)
            return x
        
    def get_rowVectors(self):
        """Get the rowVectors parameter (modular-beam only)"""
	    #bool get_rowVectors(float*);
        if self.get_numAngles() <= 0:
            return None
        else:
            x = np.ascontiguousarray(np.zeros((self.get_numAngles(),3)).astype(np.float32), dtype=np.float32)
            self.libprojectors.get_rowVectors.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
            self.libprojectors.get_rowVectors.restype = ctypes.c_bool
            self.set_model()
            self.libprojectors.get_rowVectors(x)
            return x
        
    def get_colVectors(self):
        """Get the colVectors parameter (modular-beam only)"""
	    #bool get_colVectors(float*);
        if self.get_numAngles() <= 0:
            return None
        else:
            x = np.ascontiguousarray(np.zeros((self.get_numAngles(),3)).astype(np.float32), dtype=np.float32)
            self.libprojectors.get_colVectors.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
            self.libprojectors.get_colVectors.restype = ctypes.c_bool
            self.set_model()
            self.libprojectors.get_colVectors(x)
            return x
        
    def get_numX(self):
        """Get the numX parameter"""
        self.set_model()
        return self.libprojectors.get_numX()
    
    def get_numY(self):
        """Get the numY parameter"""
        self.set_model()
        return self.libprojectors.get_numY()
    
    def get_numZ(self):
        """Get the numZ parameter"""
        self.set_model()
        return self.libprojectors.get_numZ()
        
    def get_voxelHeight(self):
        """Get the voxelHeight parameter"""
        self.libprojectors.get_voxelHeight.restype = ctypes.c_float
        self.set_model()
        return self.libprojectors.get_voxelHeight()
        
    def get_voxelWidth(self):
        """Get the voxelWidth parameter"""
        self.libprojectors.get_voxelWidth.restype = ctypes.c_float
        self.set_model()
        return self.libprojectors.get_voxelWidth()
        
    def get_offsetX(self):
        """Get the offsetX parameter"""
        self.libprojectors.get_offsetX.restype = ctypes.c_float
        self.set_model()
        return self.libprojectors.get_offsetX()
        
    def get_offsetY(self):
        """Get the offsetY parameter"""
        self.libprojectors.get_offsetY.restype = ctypes.c_float
        self.set_model()
        return self.libprojectors.get_offsetY()
        
    def get_offsetZ(self):
        """Get the offsetZ parameter"""
        self.libprojectors.get_offsetZ.restype = ctypes.c_float
        self.set_model()
        return self.libprojectors.get_offsetZ()
        
    def get_z0(self):
        """Get the z-coordinate of the first voxel"""
        self.libprojectors.get_z0.restype = ctypes.c_float
        self.set_model()
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
        """Uses napari to display the provided 3D data
        """
        self.displayVolume(vol)

    def display_volume(self,vol):
        """Uses napari to display the provided 3D data
        """
        self.displayVolume(vol)

    def displayVolume(self,vol):
        """Uses napari to display the provided 3D data
        """
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
        """Alias for sketch_system
        """
        self.sketch_system(whichView)
            
    def sketch_system(self,whichView=None):
        """ Uses matplot lib to sketch the CT geometry and CT volume
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        
        Args:
            whichView (int): if provided displays the source and detector at the specified view index
        """
        if self.get_numAngles() <= 0 or self.get_numRows() <= 0 or self.get_numCols() <= 0:
            print('CT geometry not set!')
            return False
    
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if whichView is None or isinstance(whichView, int):
            self.drawCT(ax,whichView)
        else:
            for i in range(len(whichView)):
                self.drawCT(ax,whichView[i])
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
    

    ###################################################################################################################
    ###################################################################################################################
    # FILE I/O FUNCTIONS
    ###################################################################################################################
    ###################################################################################################################
    def parse_param_dic(self, param_fn):
        pdic = {}
        not_scalar_params = ['proj_geometry', 'proj_phis', 'proj_srcpos', 'proj_modcenter', 'proj_rowvec', 'proj_colvec', 'geometry', 'phis', 'sourcePositions', 'moduleCenters', 'rowVectors', 'colVectors']
        with open(param_fn, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if len(line.strip()) == 0 or line[0] == '#':
                    continue
                key = line.split('=')[0].strip()
                value = line.split('=')[1].strip()
                #if key == 'proj_phis' or key == 'proj_geometry' or key == 'proj_srcpos' or key == 'proj_modcenter' or key == 'proj_rowvec' or key == 'proj_colvec':
                if any(item == key for item in not_scalar_params): 
                    pdic[key] = value
                else:
                    pdic[key] = float(value)
        return pdic

    def load_parameters(self, param_fn, param_type=0): # param_type 0: cfg, 1: dict
        return self.load_param(param_fn, param_type)

    def load_param(self, param_fn, param_type=0): # param_type 0: cfg, 1: dict
        pdic = {}
        if param_type == 0:
            pdic = self.parse_param_dic(param_fn)
        elif param_type == 1:
            pdic = param_fn

        if 'geometry' not in pdic.keys():
            # Could not load a basic parameter
            # Try loading the legacy parameters
            return self.load_param_legacy(param_fn, param_type)

        if 'phis' not in pdic.keys():
            phis_str = ''
        else:
            phis_str = pdic['phis']
        if len(phis_str) > 0:
            phis = np.array([float(x.strip()) for x in phis_str.split(',')]).astype(np.float32)
        elif 'angularRange' in pdic.keys():
            phis = self.setAngleArray(int(pdic['numAngles']), pdic['angularRange'])
        else:
            if pdic['geometry'] == 'parallel' or pdic['geometry'] == 'fan' or pdic['geometry'] == 'cone':
                print('ERROR: invalid LEAP parameter file')
                return False
        if pdic['geometry'] == 'modular':
            if 'sourcePositions' not in pdic.keys():
                print('ERROR: invalid LEAP parameter file')
                return False
        
        self.set_volume(int(pdic['numX']), int(pdic['numY']), int(pdic['numZ']),
                        pdic['voxelWidth'], pdic['voxelHeight'], 
                        pdic['offsetX'], pdic['offsetY'], pdic['offsetZ'])
        if pdic['geometry'] == 'parallel':
            self.set_parallelbeam(int(pdic['numAngles']), int(pdic['numRows']), int(pdic['numCols']), 
                                   pdic['pixelHeight'], pdic['pixelWidth'], 
                                   pdic['centerRow'], pdic['centerCol'], phis)
            if 'muCoeff' in pdic.keys() and 'muRadius' in pdic.keys():
                self.set_cylindircalAttenuationMap(pdic['muCoeff'], pdic['muRadius'])
        elif pdic['geometry'] == 'fan':
            self.set_fanbeam(int(pdic['numAngles']), int(pdic['numRows']), int(pdic['numCols']), 
                               pdic['pixelHeight'], pdic['pixelWidth'], 
                               pdic['centerRow'], pdic['centerCol'], 
                               phis, pdic['sod'], pdic['sdd'], pdic['tau'])
        elif pdic['geometry'] == 'cone':
            self.set_conebeam(int(pdic['numAngles']), int(pdic['numRows']), int(pdic['numCols']), 
                               pdic['pixelHeight'], pdic['pixelWidth'], 
                               pdic['centerRow'], pdic['centerCol'], 
                               phis, pdic['sod'], pdic['sdd'], pdic['tau'], pdic['helicalPitch'])
        elif pdic['geometry'] == 'modular':
        
            sourcePositions = np.array([float(x.strip()) for x in pdic['sourcePositions'].split(',')]).astype(np.float32)
            moduleCenters = np.array([float(x.strip()) for x in pdic['moduleCenters'].split(',')]).astype(np.float32)
            rowVectors = np.array([float(x.strip()) for x in pdic['rowVectors'].split(',')]).astype(np.float32)
            colVectors = np.array([float(x.strip()) for x in pdic['colVectors'].split(',')]).astype(np.float32)
            self.set_modularbeam(int(pdic['numAngles']), int(pdic['numRows']), int(pdic['numCols']), 
                                  pdic['pixelHeight'], pdic['pixelWidth'], 
                                  sourcePositions, moduleCenters, rowVectors, colVectors)

        self.set_flatDetector()
        if 'detectorType' in pdic.keys():
            if pdic['detectorType'] == 'curved':
                self.set_curvedDetector()
        
        if 'axisOfSymmetry' in pdic.keys():
            self.set_axisOfSymmetry(pdic['axisOfSymmetry'])
        
        return True

    def load_param_legacy(self, param_fn, param_type=0): # param_type 0: cfg, 1: dict
        pdic = {}
        if param_type == 0:
            pdic = self.parse_param_dic(param_fn)
        elif param_type == 1:
            pdic = param_fn

        if 'proj_geometry' not in pdic.keys():
            print('ERROR: invalid LEAP parameter file')
            return False

        if 'proj_phis' not in pdic.keys():
            phis_str = ''
        else:
            phis_str = pdic['proj_phis']
        if len(phis_str) > 0:
            #phis = torch.from_numpy(np.array([float(x.strip()) for x in phis_str.split(',')]).astype(np.float32))
            phis = np.array([float(x.strip()) for x in phis_str.split(',')]).astype(np.float32)
        elif 'proj_arange' in pdic.keys():
            phis = self.setAngleArray(int(pdic['proj_nangles']), pdic['proj_arange'])
            #phis = torch.from_numpy(phis)
        else:
            if pdic['proj_geometry'] == 'parallel' or pdic['proj_geometry'] == 'fan' or pdic['proj_geometry'] == 'cone':
                print('ERROR: invalid LEAP parameter file')
                return False
        if pdic['proj_geometry'] == 'modular':
            if 'proj_srcpos' not in pdic.keys():
                print('ERROR: invalid LEAP parameter file')
                return False
        
        self.set_volume(int(pdic['img_dimx']), int(pdic['img_dimy']), int(pdic['img_dimz']),
                        pdic['img_pwidth'], pdic['img_pheight'], 
                        pdic['img_offsetx'], pdic['img_offsety'], pdic['img_offsetz'])
        if pdic['proj_geometry'] == 'parallel':
            self.set_parallelbeam(int(pdic['proj_nangles']), int(pdic['proj_nrows']), int(pdic['proj_ncols']), 
                                   pdic['proj_pheight'], pdic['proj_pwidth'], 
                                   pdic['proj_crow'], pdic['proj_ccol'], phis)
        elif pdic['proj_geometry'] == 'fan':
            self.set_fanbeam(int(pdic['proj_nangles']), int(pdic['proj_nrows']), int(pdic['proj_ncols']), 
                               pdic['proj_pheight'], pdic['proj_pwidth'], 
                               pdic['proj_crow'], pdic['proj_ccol'], 
                               phis, pdic['proj_sod'], pdic['proj_sdd'])
        elif pdic['proj_geometry'] == 'cone':
            self.set_conebeam(int(pdic['proj_nangles']), int(pdic['proj_nrows']), int(pdic['proj_ncols']), 
                               pdic['proj_pheight'], pdic['proj_pwidth'], 
                               pdic['proj_crow'], pdic['proj_ccol'], 
                               phis, pdic['proj_sod'], pdic['proj_sdd'])
        elif pdic['proj_geometry'] == 'modular':
            self.set_modularbeam(int(pdic['proj_nangles']), int(pdic['proj_nrows']), int(pdic['proj_ncols']), 
                                  pdic['proj_pheight'], pdic['proj_pwidth'], 
                                  pdic['proj_srcpos'], pdic['proj_modcenter'], pdic['proj_rowvec'], pdic['proj_colvec'])
        
        return True
    
    def save_parameters(self, fileName):
        """Alias for save_param"""
        return self.save_param(fileName)
    
    def save_param(self, fileName):
        """Save the CT volume and CT geometry parameters to file"""
        if sys.version_info[0] == 3:
            fileName = bytes(str(fileName), 'ascii')
        self.libprojectors.saveParamsToFile.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.saveParamsToFile(fileName)
        
        return self.leapct.save_param(fileName)
    
    def save_projections(self, fileName, g):
        """Alias for saveProjections"""
        return self.saveProjections(fileName, g)
        
    def saveProjections(self, fileName, g):
        """Save projection data to file (tif sequence, nrrd, or npy)"""
        if self.get_numAngles() > 0 and self.get_numRows() > 0 and self.get_numCols() > 0:
            pixelWidth = self.get_pixelWidth()
            pixelHeight = self.get_pixelHeight()
            numCols = self.get_numCols()
            numRows = self.get_numRows()
            centerRow = self.get_centerRow()
            centerCol = self.get_centerCol()
            phis = self.get_angles()
        
            phi_0 = phis[0]
            row_0 = -centerRow*pixelHeight
            col_0 = -centerCol*pixelWidth
            T = pixelWidth
        else:
            phi_0 = 0.0
            row_0 = 0.0
            col_0 = 0.0
            T = 1.0
        return self.saveData(fileName, g, T, phi_0, row_0, col_0)
    
    def save_volume(self, fileName, f):
        """Alias for saveVolume"""
        return self.saveVolume(fileName, f)
    
    def saveVolume(self, fileName, f):
        """Save volume data to file (tif sequence, nrrd, or npy)"""
        if self.get_numX() > 0 and self.get_numY() > 0 and self.get_numZ() > 0:
            z_0 = self.z_samples()[0]
            y_0 = self.y_samples()[0]
            x_0 = self.x_samples()[0]
            T = self.get_voxelWidth()
        else:
            x_0 = 0.0
            y_0 = 0.0
            z_0 = 0.0
            T = 1.0
        return self.saveData(fileName, f, T, x_0, y_0, z_0)
        
    def saveData(self, fileName, x, T=1.0, offset_0=0.0, offset_1=0.0, offset_2=0.0):
        """Save 3D data to file (tif sequence, nrrd, or npy)"""
        volFilePath, dontCare = os.path.split(fileName)
        if os.path.isdir(volFilePath) == False or os.access(volFilePath, os.W_OK) == False:
            print('Folder to save data either does not exist or not accessible!')
            return False
        if fileName.endswith('.npy'):
            if has_torch == True and type(x) is torch.Tensor:
                np.save(fileName, x.numpy())
            else:
                np.save(fileName, x)
            return True
        elif fileName.endswith('.nrrd'):
            try:
                import nrrd
                
                # https://pynrrd.readthedocs.io/en/latest/examples.html
                header = {'units': ['mm', 'mm', 'mm'], 'spacings': [T, T, T], 'axismins': [offset_0, offset_1, offset_2], 'thicknesses': [T, T, T],}
                nrrd.write(fileName, x, header)
                return True
            except:
                print('Error: Failed to load nrrd library!')
                print('To install this package do: pip install pynrrd')
                return False
        elif fileName.endswith('.tif') or fileName.endswith('.tiff'):
            try:
                #from PIL import Image
                import imageio
                
                baseName, fileExtension = os.path.splitext(fileName)
                
                for i in range(x.shape[0]):
                    if has_torch == True and type(x) is torch.Tensor:
                        #im = Image.fromarray(x[i,:,:].numpy())
                        im = x[i,:,:].numpy()
                    else:
                        #im = Image.fromarray(x[i,:,:])
                        im = x[i,:,:]
                    #im.save(baseName + '_' + str(int(i)) + fileExtension)
                    imageio.imwrite(baseName + '_' + str(int(i)) + fileExtension, im)
                return True
                
            except:
                #print('Error: Failed to load PIL library!')
                #print('To install this package do: pip install Pillow')
                print('Error: Failed to load imageio library!')
                print('To install PIL do: pip install imageio')
                return False
        else:
            print('Error: must be a tif, npy, or nrrd file!')
            return False
            
    def loadVolume(self, fileName):
        """Load 3D volume data from file (tif sequence, nrrd, or npy)"""
        return self.loadData(fileName)
        
    def load_volume(self, fileName):
        """Load 3D volume data from file (tif sequence, nrrd, or npy)"""
        return self.loadData(fileName)
        
    def loadProjections(self, fileName):
        """Load 3D projection data from file (tif sequence, nrrd, or npy)"""
        return self.loadData(fileName)
        
    def load_projections(self, fileName):
        """Load 3D projection data from file (tif sequence, nrrd, or npy)"""
        return self.loadData(fileName)
            
    def loadData(self, fileName):
        """Load 3D data from file (tif sequence, nrrd, or npy)"""
        if fileName.endswith('.npy'):
            if os.path.isfile(fileName) == False:
                print('file does not exist')
                return None
            else:
                return np.load(fileName)
        elif fileName.endswith('.nrrd'):
            if os.path.isfile(fileName) == False:
                print('file does not exist')
                return None
            try:
                import nrrd
                x, header = nrrd.read(fileName)
                T_fromFile = header['spacings'][0]
                return x
            except:
                print('Error: Failed to load nrrd library!')
                print('To install this package do: pip install pynrrd')
                return None
        elif fileName.endswith('.tif') or fileName.endswith('.tiff'):
            
            try:
                #from PIL import Image
                import imageio
                import glob
                hasPIL = True
            except:
                #print('Error: Failed to load PIL or glob library!')
                #print('To install PIL do: pip install Pillow')
                print('Error: Failed to load imageio or glob library!')
                print('To install PIL do: pip install imageio')
                return None
            if hasPIL == True:
                currentWorkingDirectory = os.getcwd()
                dataFolder, baseFileName = os.path.split(fileName)
                if len(dataFolder) > 0:
                    os.chdir(dataFolder)
                baseFileName, fileExtension = os.path.splitext(os.path.basename(baseFileName))
                templateFile = baseFileName + '_*' + fileExtension
                fileList = glob.glob(os.path.split(templateFile)[1])
                if len(fileList) == 0:
                    os.chdir(currentWorkingDirectory)
                    print('file sequence does not exist')
                    return None
                justDigits = []
                for i in range(len(fileList)):
                    digitStr = fileList[i].replace(baseFileName+'_','').replace('.tif','')
                    justDigits.append(int(digitStr))
                ind = np.argsort(justDigits)

                #print('found ' + str(len(fileList)) + ' images')
                #print('reading first image: ' + str(fileList[0]))
                #firstImg = np.array(Image.open(fileList[0]))
                firstImg = np.array(imageio.imread(fileList[0]))
                x = np.zeros((len(fileList), firstImg.shape[0], firstImg.shape[1]), dtype=np.float32)
                print('found ' + str(x.shape[0]) + ' images of size ' + str(x.shape[1]) + ' x ' + str(x.shape[2]))
                for i in range(len(fileList)):
                    #anImage = np.array(Image.open(fileList[ind[i]]))
                    #anImage = np.array(Image.open(fileList[ind[i]]).rotate(-0.5))
                    anImage = np.array(imageio.imread(fileList[ind[i]]))
                    x[i,:,:] = anImage[:,:]
                os.chdir(currentWorkingDirectory)
                return x
            '''
            try:
                from PIL import Image
                
                baseName, fileExtension = os.path.splitext(fileName)
                
                for i in range(x.shape[0]):
                    im = Image.fromarray(x[i,:,:])
                    im.save(baseName + '_' + str(int(i)) + fileExtension)
                return x
                
            except:
                print('Error: Failed to load PIL library!')
                print('To install this package do: pip install Pillow')
                return None
            #'''
        else:
            print('Error: must be a tif, npy, or nrrd file!')
            return None
    
    
    ###################################################################################################################
    ###################################################################################################################
    # PHANTOM SPECIFICATION FUNCTIONS
    ###################################################################################################################
    ###################################################################################################################
    def addObject(self, f, typeOfObject, c, r, val, A=None, clip=None, oversampling=1):
        """Adds a geometric object to the volume
        
        The CT volume parameters must be specified prior to running this functions
        
        Args:
            f (C contiguous float32 numpy array): CT volume
            type (int): ELLIPSOID=0, PARALLELEPIPED=1, CYLINDER_X=2, CYLINDER_Y=3, CYLINDER_Z=4, CONE_X=5, CONE_Y=6, CONE_Z=7
            c (3X1 numpy array): x,y,z coordinates of the center of the object
            r (3X1 numpy array): radii in the x,y,z directions of the object
            val (float): the values to ascribe inside this object
            A (3x3 numpy array): rotation matrix to rotate the object about its center
            clip (3X1 numpy array): specifies the clipping planes, if any
        """
        self.libprojectors.addObject.restype = ctypes.c_bool
        if A is None:
            A = np.zeros((3,3),dtype=np.float32)
            A[0,0] = 1.0
            A[1,1] = 1.0
            A[2,2] = 1.0
        if clip is None:
            clip = np.zeros(3,dtype=np.float32)
        
        c = np.ascontiguousarray(c, dtype=np.float32)
        r = np.ascontiguousarray(r, dtype=np.float32)
        A = np.ascontiguousarray(A, dtype=np.float32)
        clip = np.ascontiguousarray(clip, dtype=np.float32)
        self.set_model()
        if has_torch == True and type(f) is torch.Tensor:
            self.libprojectors.addObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_float, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int]
            return self.libprojectors.addObject(f.data_ptr(), int(typeOfObject), c, r, float(val), A, clip, oversampling)
        else:
            self.libprojectors.addObject.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_float, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int]
            return self.libprojectors.addObject(f, int(typeOfObject), c, r, float(val), A, clip, oversampling)
        
    def set_FORBILD(self, f, includeEar=False, oversampling=1):
        """Sets the FORBILD head phantom
        
        Note that the values of the FORBILD head phantom are all scaled by 0.02
        which is the LAC of water at around 60 keV
        """
        has_scipy = False
        try:
            from scipy.spatial.transform import Rotation as R
            has_scipy = True
        except:
            print('Warning: scipy package cannot be found, so not included rotated objects')
            has_scipy = False
        self.addObject(f, 0, 10.0*np.array([0.0, 0.0, 0.0]), 10.0*np.array([9.6, 12.0, 12.5]), 1.800*0.02, None, None, oversampling)
        self.addObject(f, 0, 10.0*np.array([0.0, 0.0, 0.0]), 10.0*np.array([9.0, 11.4, 11.9]), 1.050*0.02, None, None, oversampling)
        self.addObject(f, 0, 10.0*np.array([-4.7, 4.3, 0.872]), 10.0*np.array([2.0, 2.0, 2.0]), 1.060*0.02, None, None, oversampling)
        self.addObject(f, 0, 10.0*np.array([4.7, 4.3, 0.872]), 10.0*np.array([2.0, 2.0, 2.0]), 1.060*0.02, None, None, oversampling)
        self.addObject(f, 0, 10.0*np.array([-1.08, -9, 0.0]), 10.0*np.array([0.4, 0.4, 0.4]), 1.0525*0.02, None, None, oversampling)
        self.addObject(f, 0, 10.0*np.array([1.08, -9, 0.0]), 10.0*np.array([0.4, 0.4, 0.4]), 1.0475*0.02, None, None, oversampling)
        self.addObject(f, 0, 10.0*np.array([0.0, 8.4, 0.0]), 10.0*np.array([1.8, 3.0, 3.0]), 0.0, None, None, oversampling)
        if has_scipy:
            self.addObject(f, 0, 10.0*np.array([-1.9, 5.4, 0.0]), 10.0*np.array([1.206483*np.cos(15*np.pi/180.0), 0.420276*np.cos(15*np.pi/180.0), 3.0]), 1.800*0.02, R.from_euler("xyz", [0, 0, -120], degrees=True).as_matrix(), None, oversampling)
            self.addObject(f, 0, 10.0*np.array([1.9, 5.4, 0.0]), 10.0*np.array([1.2*np.cos(15*np.pi/180.0), 0.42*np.cos(15*np.pi/180.0), 3.0]), 1.800*0.02, R.from_euler("xyz", [0, 0, 120], degrees=True).as_matrix(), None, oversampling)
            self.addObject(f, 4, 10.0*np.array([-4.3, 6.8, -1.0]), 10.0*np.array([1.8, 0.24, 2.0]), 1.800*0.02, R.from_euler("xyz", [0, 0, -150], degrees=True).as_matrix(), None, oversampling)
            self.addObject(f, 4, 10.0*np.array([4.3, 6.8, -1.0]), 10.0*np.array([1.8, 0.24, 2.0]), 1.800*0.02, R.from_euler("xyz", [0, 0, -30], degrees=True).as_matrix(), None, oversampling)
        self.addObject(f, 0, 10.0*np.array([0.0, -3.6, 0.0]), 10.0*np.array([1.8, 3.6, 3.6]), 1.045*0.02, None, None, oversampling)
        if has_scipy:
            self.addObject(f, 0, 10.0*np.array([6.393945, -6.393945, 0.0]), 10.0*np.array([1.2, 0.42, 1.2]), 1.055*0.02, R.from_euler("xyz", [0, 0, -58.1], degrees=True).as_matrix(), None, oversampling)
            self.addObject(f, 4, 10.0*np.array([0.0, 3.6, 0.0]), 10.0*np.array([1.2, 4.0, 0.25*np.cos(15*np.pi/180.0)]), 1.800*0.02, R.from_euler("xyz", [60, 0, 0], degrees=True).as_matrix(), None, oversampling)
            self.addObject(f, 2, 10.0*np.array([0.0, 9.6, 0.0]), 10.0*np.array([0.525561/2.0, 2.0, 0.4]), 1.800*0.02, R.from_euler("xyz", [-60, 0, 0], degrees=True).as_matrix(), None, oversampling)
        self.addObject(f, 6, 10.0*np.array([0.0, -11.15, -0.2]), 10.0*np.array([0.5, 0.75, 0.2]), 1.800*0.02, None, None, oversampling)
        self.addObject(f, 6, 10.0*np.array([0.0, -11.15, 0.2]), 10.0*np.array([0.5, 0.75, 0.2]), 1.800*0.02, None, None, oversampling)
        if self.get_numAngles() == 1 and self.get_numX() == 1:
            pass
        else:
            self.addObject(f, 0, 10.0*np.array([9.1, 0.0, 0.0]), 10.0*np.array([4.2, 1.8, 1.8]), 1.800*0.02, None, np.array([1.0, 0.0, 0.0]), oversampling)

        #'''
        if includeEar:
            xyzs = np.array([8.80, -1.03920, -1.03920,  8.40, -1.03920, -1.03920,  8.0, -1.03920, -1.03920,  7.60, -1.03920, -1.03920,  8.60, -0.69280, -1.03920,  8.20, -0.69280, -1.03920,  7.80, -0.69280, -1.03920,  7.40, -0.69280, -1.03920,  7.0, -0.69280, -1.03920,  8.80, -0.34640, -1.03920,  8.40, -0.34640, -1.03920,  8.0, -0.34640, -1.03920,  7.60, -0.34640, -1.03920,  7.20, -0.34640, -1.03920,  6.80, -0.34640, -1.03920,  8.80, 1.03920, -1.03920,  8.40, 1.03920, -1.03920,  8.0, 1.03920, -1.03920,  7.60, 1.03920, -1.03920,  8.60, 0.69280, -1.03920,  8.20, 0.69280, -1.03920,  7.80, 0.69280, -1.03920,  7.40, 0.69280, -1.03920,  7.0, 0.69280, -1.03920,  8.80, 0.34640, -1.03920,  8.40, 0.34640, -1.03920,  8.0, 0.34640, -1.03920,  7.60, 0.34640, -1.03920,  7.20, 0.34640, -1.03920,  6.80, 0.34640, -1.03920,  8.60, 0.0, -1.03920,  8.20, 0.0, -1.03920,  7.80, 0.0, -1.03920,  7.40, 0.0, -1.03920,  7.0, 0.0, -1.03920,  6.60, 0.0, -1.03920,  8.80, -1.03920, 1.03920,  8.40, -1.03920, 1.03920,  8.0, -1.03920, 1.03920,  7.60, -1.03920, 1.03920,  8.60, -0.69280, 1.03920,  8.20, -0.69280, 1.03920,  7.80, -0.69280, 1.03920,  7.40, -0.69280, 1.03920,  7.0, -0.69280, 1.03920,  8.80, -0.34640, 1.03920,  8.40, -0.34640, 1.03920,  8.0, -0.34640, 1.03920,  7.60, -0.34640, 1.03920,  7.20, -0.34640, 1.03920,  6.80, -0.34640, 1.03920,  8.80, 1.03920, 1.03920,  8.40, 1.03920, 1.03920,  8.0, 1.03920, 1.03920,  7.60, 1.03920, 1.03920,  8.60, 0.69280, 1.03920,  8.20, 0.69280, 1.03920,  7.80, 0.69280, 1.03920,  7.40, 0.69280, 1.03920,  7.0, 0.69280, 1.03920,  8.80, 0.34640, 1.03920,  8.40, 0.34640, 1.03920,  8.0, 0.34640, 1.03920,  7.60, 0.34640, 1.03920,  7.20, 0.34640, 1.03920,  6.80, 0.34640, 1.03920,  8.60, 0.0, 1.03920,  8.20, 0.0, 1.03920,  7.80, 0.0, 1.03920,  7.40, 0.0, 1.03920,  7.0, 0.0, 1.03920,  6.60, 0.0, 1.03920,  8.60, -1.03920, -0.69280,  8.20, -1.03920, -0.69280,  7.80, -1.03920, -0.69280,  7.40, -1.03920, -0.69280,  7.0, -1.03920, -0.69280,  8.80, -0.69280, -0.69280,  8.40, -0.69280, -0.69280,  8.0, -0.69280, -0.69280,  7.60, -0.69280, -0.69280,  7.20, -0.69280, -0.69280,  6.80, -0.69280, -0.69280,  6.40, -0.69280, -0.69280,  8.60, -0.34640, -0.69280,  8.20, -0.34640, -0.69280,  7.80, -0.34640, -0.69280,  7.40, -0.34640, -0.69280,  7.0, -0.34640, -0.69280,  6.60, -0.34640, -0.69280,  6.20, -0.34640, -0.69280,  8.60, 1.03920, -0.69280,  8.20, 1.03920, -0.69280,  7.80, 1.03920, -0.69280,  7.40, 1.03920, -0.69280,  7.0, 1.03920, -0.69280,  8.80, 0.69280, -0.69280,  8.40, 0.69280, -0.69280,  8.0, 0.69280, -0.69280,  7.60, 0.69280, -0.69280,  7.20, 0.69280, -0.69280,  6.80, 0.69280, -0.69280,  6.40, 0.69280, -0.69280,  8.60, 0.34640, -0.69280,  8.20, 0.34640, -0.69280,  7.80, 0.34640, -0.69280,  7.40, 0.34640, -0.69280,  7.0, 0.34640, -0.69280,  6.60, 0.34640, -0.69280,  6.20, 0.34640, -0.69280,  8.80, 0.0, -0.69280,  8.40, 0.0, -0.69280,  8.0, 0.0, -0.69280,  7.60, 0.0, -0.69280,  7.20, 0.0, -0.69280,  6.80, 0.0, -0.69280,  6.40, 0.0, -0.69280,  6.0, 0.0, -0.69280,  8.60, -1.03920, 0.69280,  8.20, -1.03920, 0.69280,  7.80, -1.03920, 0.69280,  7.40, -1.03920, 0.69280,  7.0, -1.03920, 0.69280,  8.80, -0.69280, 0.69280,  8.40, -0.69280, 0.69280,  8.0, -0.69280, 0.69280,  7.60, -0.69280, 0.69280,  7.20, -0.69280, 0.69280,  6.80, -0.69280, 0.69280,  6.40, -0.69280, 0.69280,  8.60, -0.34640, 0.69280,  8.20, -0.34640, 0.69280,  7.80, -0.34640, 0.69280,  7.40, -0.34640, 0.69280,  7.0, -0.34640, 0.69280,  6.60, -0.34640, 0.69280,  6.20, -0.34640, 0.69280,  8.60, 1.03920, 0.69280,  8.20, 1.03920, 0.69280,  7.80, 1.03920, 0.69280,  7.40, 1.03920, 0.69280,  7.0, 1.03920, 0.69280,  8.80, 0.69280, 0.69280,  8.40, 0.69280, 0.69280,  8.0, 0.69280, 0.69280,  7.60, 0.69280, 0.69280,  7.20, 0.69280, 0.69280,  6.80, 0.69280, 0.69280,  6.40, 0.69280, 0.69280,  8.60, 0.34640, 0.69280,  8.20, 0.34640, 0.69280,  7.80, 0.34640, 0.69280,  7.40, 0.34640, 0.69280,  7.0, 0.34640, 0.69280,  6.60, 0.34640, 0.69280,  6.20, 0.34640, 0.69280,  8.80, 0.0, 0.69280,  8.40, 0.0, 0.69280,  8.0, 0.0, 0.69280,  7.60, 0.0, 0.69280,  7.20, 0.0, 0.69280,  6.80, 0.0, 0.69280,  6.40, 0.0, 0.69280,  6.0, 0.0, 0.69280,  8.80, -1.03920, -0.34640,  8.40, -1.03920, -0.34640,  8.0, -1.03920, -0.34640,  7.60, -1.03920, -0.34640,  7.20, -1.03920, -0.34640,  6.80, -1.03920, -0.34640,  8.60, -0.69280, -0.34640,  8.20, -0.69280, -0.34640,  7.80, -0.69280, -0.34640,  7.40, -0.69280, -0.34640,  7.0, -0.69280, -0.34640,  6.60, -0.69280, -0.34640,  6.20, -0.69280, -0.34640,  8.80, -0.34640, -0.34640,  8.40, -0.34640, -0.34640,  8.0, -0.34640, -0.34640,  7.60, -0.34640, -0.34640,  7.20, -0.34640, -0.34640,  6.80, -0.34640, -0.34640,  6.40, -0.34640, -0.34640,  6.0, -0.34640, -0.34640,  8.80, 1.03920, -0.34640,  8.40, 1.03920, -0.34640,  8.0, 1.03920, -0.34640,  7.60, 1.03920, -0.34640,  7.20, 1.03920, -0.34640,  6.80, 1.03920, -0.34640,  8.60, 0.69280, -0.34640,  8.20, 0.69280, -0.34640,  7.80, 0.69280, -0.34640,  7.40, 0.69280, -0.34640,  7.0, 0.69280, -0.34640,  6.60, 0.69280, -0.34640,  6.20, 0.69280, -0.34640,  8.80, 0.34640, -0.34640,  8.40, 0.34640, -0.34640,  8.0, 0.34640, -0.34640,  7.60, 0.34640, -0.34640,  7.20, 0.34640, -0.34640,  6.80, 0.34640, -0.34640,  6.40, 0.34640, -0.34640,  6.0, 0.34640, -0.34640,  8.60, 0.0, -0.34640,  8.20, 0.0, -0.34640,  7.80, 0.0, -0.34640,  7.40, 0.0, -0.34640,  7.0, 0.0, -0.34640,  6.60, 0.0, -0.34640,  6.20, 0.0, -0.34640,  5.80, 0.0, -0.34640,  8.80, -1.03920, 0.34640,  8.40, -1.03920, 0.34640,  8.0, -1.03920, 0.34640,  7.60, -1.03920, 0.34640,  7.20, -1.03920, 0.34640,  6.80, -1.03920, 0.34640,  8.60, -0.69280, 0.34640,  8.20, -0.69280, 0.34640,  7.80, -0.69280, 0.34640,  7.40, -0.69280, 0.34640,  7.0, -0.69280, 0.34640,  6.60, -0.69280, 0.34640,  6.20, -0.69280, 0.34640,  8.80, -0.34640, 0.34640,  8.40, -0.34640, 0.34640,  8.0, -0.34640, 0.34640,  7.60, -0.34640, 0.34640,  7.20, -0.34640, 0.34640,  6.80, -0.34640, 0.34640,  6.40, -0.34640, 0.34640,  6.0, -0.34640, 0.34640,  8.80, 1.03920, 0.34640,  8.40, 1.03920, 0.34640,  8.0, 1.03920, 0.34640,  7.60, 1.03920, 0.34640,  7.20, 1.03920, 0.34640,  6.80, 1.03920, 0.34640,  8.60, 0.69280, 0.34640,  8.20, 0.69280, 0.34640,  7.80, 0.69280, 0.34640,  7.40, 0.69280, 0.34640,  7.0, 0.69280, 0.34640,  6.60, 0.69280, 0.34640,  6.20, 0.69280, 0.34640,  8.80, 0.34640, 0.34640,  8.40, 0.34640, 0.34640,  8.0, 0.34640, 0.34640,  7.60, 0.34640, 0.34640,  7.20, 0.34640, 0.34640,  6.80, 0.34640, 0.34640,  6.40, 0.34640, 0.34640,  6.0, 0.34640, 0.34640,  8.60, 0.0, 0.34640,  8.20, 0.0, 0.34640,  7.80, 0.0, 0.34640,  7.40, 0.0, 0.34640,  7.0, 0.0, 0.34640,  6.60, 0.0, 0.34640,  6.20, 0.0, 0.34640,  5.80, 0.0, 0.34640,  8.60, -1.03920, 0.0,  8.20, -1.03920, 0.0,  7.80, -1.03920, 0.0,  7.40, -1.03920, 0.0,  7.0, -1.03920, 0.0,  6.60, -1.03920, 0.0,  8.80, -0.69280, 0.0,  8.40, -0.69280, 0.0,  8.0, -0.69280, 0.0,  7.60, -0.69280, 0.0,  7.20, -0.69280, 0.0,  6.80, -0.69280, 0.0,  6.40, -0.69280, 0.0,  6.0, -0.69280, 0.0,  8.60, -0.34640, 0.0,  8.20, -0.34640, 0.0,  7.80, -0.34640, 0.0,  7.40, -0.34640, 0.0,  7.0, -0.34640, 0.0,  6.60, -0.34640, 0.0,  6.20, -0.34640, 0.0,  5.80, -0.34640, 0.0,  8.60, 1.03920, 0.0,  8.20, 1.03920, 0.0,  7.80, 1.03920, 0.0,  7.40, 1.03920, 0.0,  7.0, 1.03920, 0.0,  6.60, 1.03920, 0.0,  8.80, 0.69280, 0.0,  8.40, 0.69280, 0.0,  8.0, 0.69280, 0.0,  7.60, 0.69280, 0.0,  7.20, 0.69280, 0.0,  6.80, 0.69280, 0.0,  6.40, 0.69280, 0.0,  6.0, 0.69280, 0.0,  8.60, 0.34640, 0.0,  8.20, 0.34640, 0.0,  7.80, 0.34640, 0.0,  7.40, 0.34640, 0.0,  7.0, 0.34640, 0.0,  6.60, 0.34640, 0.0,  6.20, 0.34640, 0.0,  5.80, 0.34640, 0.0,  8.80, 0.0, 0.0,  8.40, 0.0, 0.0,  8.0, 0.0, 0.0,  7.60, 0.0, 0.0,  7.20, 0.0, 0.0,  6.80, 0.0, 0.0,  6.40, 0.0, 0.0,  6.0, 0.0, 0.0,  5.60, 0.0, 0.0])
            for n in range(xyzs.size//3):
                x = xyzs[3*n+0]
                y = xyzs[3*n+1]
                z = xyzs[3*n+2]
                self.addObject(f, 0, 10.0*np.array([x, y, z]), 10.0*np.array([0.15, 0.15, 0.15]), 0.0, None, None, oversampling)
        #'''
        
class subsetParameters:
    def __init__(self, ctModel, numSubsets):
        self.numSubsets = numSubsets
        self.ctModel = ctModel
        
        self.phis_subsets = []
        self.sourcePositions_subsets = []
        self.moduleCenters_subsets = []
        self.rowVectors_subsets = []
        self.colVectors_subsets = []
        
        self.phis = None
        self.sourcePositions = None
        self.moduleCenters = None
        self.rowVectors = None
        self.colVectors = None

        # First set parameters for all angles
        if self.ctModel.get_geometry() == 'MODULAR':
            self.sourcePositions = self.ctModel.get_sourcePositions()
            self.moduleCenters = self.ctModel.get_moduleCenters()
            self.rowVectors = self.ctModel.get_rowVectors()
            self.colVectors = self.ctModel.get_colVectors()
        else:
            self.phis = self.ctModel.get_angles()
            
        # Now set parameters for each subset
        if self.ctModel.get_geometry() == 'MODULAR':
            for m in range(self.numSubsets):
                if m == self.sourcePositions.shape[0]-1:
                    sourcePositions_subset = np.zeros((1,3),dtype=np.float32)
                    sourcePositions_subset[0,:] = self.sourcePositions[m,:]
                    self.sourcePositions_subsets.append(sourcePositions_subset)
                    
                    moduleCenters_subset = np.zeros((1,3),dtype=np.float32)
                    moduleCenters_subset[0,:] = self.moduleCenters[m,:]
                    self.moduleCenters_subsets.append(moduleCenters_subset)

                    rowVectors_subset = np.zeros((1,3),dtype=np.float32)
                    rowVectors_subset[0,:] = self.rowVectors[m,:]
                    self.rowVectors_subsets.append(rowVectors_subset)
                    
                    colVectors_subset = np.zeros((1,3),dtype=np.float32)
                    colVectors_subset[0,:] = self.colVectors[m,:]
                    self.colVectors_subsets.append(colVectors_subset)

                else:
                    sourcePositions_subset = np.ascontiguousarray(self.sourcePositions[m:-1:self.numSubsets,:], np.float32)
                    self.sourcePositions_subsets.append(sourcePositions_subset)
                    
                    moduleCenters_subset = np.ascontiguousarray(self.moduleCenters[m:-1:self.numSubsets,:], np.float32)
                    self.moduleCenters_subsets.append(moduleCenters_subset)
                    
                    rowVectors_subset = np.ascontiguousarray(self.rowVectors[m:-1:self.numSubsets,:], np.float32)
                    self.rowVectors_subsets.append(rowVectors_subset)
                    
                    colVectors_subset = np.ascontiguousarray(self.colVectors[m:-1:self.numSubsets,:], np.float32)
                    self.colVectors_subsets.append(colVectors_subset)
        else:
            phis_subsets = []
            for m in range(self.numSubsets):
                if m == self.phis.size-1:
                    phis_subset = np.zeros((1,1),dtype=np.float32)
                    phis_subset[0,0] = self.phis[m]
                    self.phis_subsets.append(phis_subset)
                else:
                    phis_subset = np.ascontiguousarray(self.phis[m:-1:self.numSubsets], np.float32)
                    self.phis_subsets.append(phis_subset)
        
    def setSubset(self, isubset):
        if 0 <= isubset and isubset < self.numSubsets:
            if self.ctModel.get_geometry() == 'MODULAR':
                numAngles = self.sourcePositions_subsets[isubset].shape[0]
                numRows = self.ctModel.get_numRows()
                numCols = self.ctModel.get_numCols()
                pixelHeight = self.ctModel.get_pixelHeight()
                pixelWidth = self.ctModel.get_pixelWidth()
                sourcePositions = self.sourcePositions_subsets[isubset]
                moduleCenters = self.moduleCenters_subsets[isubset]
                rowVectors = self.rowVectors_subsets[isubset]
                colVectors = self.colVectors_subsets[isubset]
                self.ctModel.set_modularBeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, sourcePositions, moduleCenters, rowVectors, colVectors)
                #print(sourcePositions)
            else:
                self.ctModel.set_angles(self.phis_subsets[isubset])
        else:
            if self.ctModel.get_geometry() == 'MODULAR':
                numAngles = self.sourcePositions.shape[0]
                numRows = self.ctModel.get_numRows()
                numCols = self.ctModel.get_numCols()
                pixelHeight = self.ctModel.get_pixelHeight()
                pixelWidth = self.ctModel.get_pixelWidth()
                sourcePositions = self.sourcePositions
                moduleCenters = self.moduleCenters
                rowVectors = self.rowVectors
                colVectors = self.colVectors
                self.ctModel.set_modularBeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, sourcePositions, moduleCenters, rowVectors, colVectors)
            else:
                self.ctModel.set_angles(self.phis)
