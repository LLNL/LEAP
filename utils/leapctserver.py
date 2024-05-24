################################################################################
# Copyright 2024 Kyle Champley
# SPDX-License-Identifier: MIT
#
# LivermorE AI Projector for Computed Tomography (LEAP)
# leapctserver class
# This class manages file I/O, memory, and spectra parameters for LEAP
# It is useful for running algorithms that require more CPU RAM
# than what is available by processing the data in smaller chunks and saving
# intermediate results to file.
# *** Currently under development and not ready for use
################################################################################
import os
import sys
import uuid
import numpy as np
from leapctype import *
import leap_preprocessing_algorithms

try:
    from xrayphysics import *
    has_physics = True
except:
    has_physics = False

class leapctserver:

    def __init__(self, leapct=None, path=None, outputDir=None):
        if leapct is None:
            self.leapct = tomographicModels()
        else:
            self.leapct = leapct
        
        if has_physics:
            physics = xrayPhysics()
        else:
            physics = None
        
        ### Section I: file names
        # Path to folder where inputs are stored and outputs may be written
        if path is None:
            self.path = os.getcwd()
        else:
            self.path = path
            
        # Output directory (must be a subfolder of path)
        if outputDir is None:
            self.outputDir = str('leapct_') + str(uuid.uuid4().hex)
        else:
            self.outputDir = outputDir

        # Create folders if they do not exist
        fullPath = os.path.join(self.path, self.outputDir)
        if not os.path.exists(fullPath):
            os.makedirs(fullPath)
        
        # File name for air scan data
        self.air_scan_file = None
        
        # File name for dark scan data
        self.dark_scan_file = None
        
        # File name for raw data projections
        self.raw_scan_file = None
        
        # File name for transmission or attenuation projections
        self.projection_file = None
        
        # File name for reconstructed slices
        self.reconstruction_file = None
        
        # File name where CT geometry and CT volume parameters are stored
        self.geometry_file = None
        
        # File name where all spectra parameters are stored
        self.spectra_model_file = None
        
        # File name where source spectra is stored
        self.source_spectra_file = None
        
        # File name where detector response is stored
        self.detector_response_file = None

        # Tags for the projection data type
        [self.UNSPECIFIED, self.RAW, self.RAW_DARK_SUBTRACTED, self.TRANSMISSION, self.ATTENUATION] = [0, 1, 2, 3, 4]
        self.data_type = self.UNSPECIFIED

        
        ### Section II: data
        # Projection data (numpy array or torch tensor)
        self.g = None
        
        # Reconstruction volume data (numpy array or torch tensor)
        self.f = None
        
        # The maximum amount of memory that leapctserver is allowed to use
        # Users are encouraged to change this!
        physicalMemory = self.total_RAM()
        if physicalMemory > 0.0:
            if physicalMemory < 0.8:
                self.max_CPU_memory_usage = physicalMemory
            elif physicalMemory < 8.0:
                self.max_CPU_memory_usage = 5.0/6.0*physicalMemory - 2.0/3.0 # 1 GB if have 2 GB of memory, 6 GB if 8 GB of memory
            elif physicalMemory < 32.0:
                self.max_CPU_memory_usage = 11.0/12.0*physicalMemory - 4.0/3.0 # 6 GB if 8 GB of memory, 28 GB if 32 GB of memory
            else
                self.max_CPU_memory_usage = min(physicalMemory - 4.0, 0.95*physicalMemory) # reserve 4 GB of memory; this is likely too much
        else:
            self.max_CPU_memory_usage = 128.0

        
        ### Section III: spectra parameters
        self.lowest_energy = -1.0
        self.energy_bin_width = -1.0
        self.kV = None
        self.takeoff_angle = 11.0
        self.anode_material = 74
        self.xray_filters = None
        self.detector_response = None
        
    
    ###################################################################################################################
    ###################################################################################################################
    # FILE NAMES
    ###################################################################################################################
    ###################################################################################################################
    def set_path(self, path):
        if os.path.exists(fullPath):
            self.path = path
        else:
            print('Error: specified path does not exist')
        
    def clear_path(self):
        self.path = None
        
    def set_raw_data_files(self, raw, air, dark=None):
        if raw is not None and air is None:
            print('Error: air scan file name must be specified, when raw data file name is specified')
            return
        self.dark_scan_file = dark
        self.air_scan_file = air
        self.raw_scan_file = raw
        if raw is None:
            self.data_type = self.UNSPECIFIED
        elif self.dark_scan_file is None:
            self.data_type = self.RAW_DARK_SUBTRACTED
        else:
            self.data_type = self.RAW
            
    def set_transmission_data_files(self, trans):
        if trans is None:
            self.data_type = self.UNSPECIFIED
        else:
            self.data_type = self.TRANSMISSION
        self.projection_file = trans
    
    def set_attenuation_data_files(self, atten):
        if atten is None:
            self.data_type = self.UNSPECIFIED
        else:
            self.data_type = self.ATTENUATION
        self.projection_file = atten
        
    def set_reconstruction_data_file(self, zslices):
        #if zslices is None:
        #    print('Error: reconstruction data file names must be specified')
        #    return
        self.reconstruction_file = zslices
    
    def save_geometry_file(self):
        self.leapct.save_parameters(self.geometry_file)
    
    def save_spectra_model(self):
        if has_physics:
            if self.source_spectra_defined():
                Es, s = self.source_spectra()
                self.physics.save_spectra(self.source_spectra_file, s, Es)
            if self.detector_response_defined():
                Es, d = self.detector_response()
                self.physics.save_spectra(self.detector_response_file, d, Es)
        
    def save_parameters(self):
        """Saves CT geometry, CT volume, and all spectra parameters to file"""
        self.save_geometry_file()
        self.save_spectra_model()
    
    def load_projection_angles(self, fileName=None, inds=None):
        """load selected angles of projections
        
        Args:
            fileName (string): full path
        """
        if fileName is None:
            if self.data_type == self.RAW or self.data_type == self.RAW_DARK_SUBTRACTED:
                if self.raw_scan_file is None:
                    print('Error: data_type is raw, but raw_scan_file is not specified!')
                    return
                fileName = os.path.join(self.path, self.raw_scan_file)
            else:
                if self.projection_file is None:
                    print('Error: projection_file is not specified!')
                    return
                fileName = os.path.join(self.path, self.projection_file)
        if os.path.isfile(fileName) == False:
            print('Error: ' + str(fileName) + ' does not exist!')
            return
        dataFolder, baseFileName = os.path.split(fileName)
        if "sino" in baseFileName:
            if inds is not None:
                g = np.zeros((inds[1]-inds[0]+1, self.leapct.get_numRows(), self.leapct.get_numCols()),dtype=np.float32)
            else:
                g = np.zeros((self.leapct.get_numAngles(), self.leapct.get_numRows(), self.leapct.get_numCols()),dtype=np.float32)
            g = np.swapaxes(g, 0, 1)
            g = self.leapct.load_data(fileName, x=g, fileRange=None, rowRange=inds, colRange=None)
            g = np.swapaxes(g, 0, 1)
            g = np.ascontiguousarray(g, dtype=np.float32)
        else:
            g = self.leapct.load_data(fileName, x=None, fileRange=inds, rowRange=None, colRange=None)
        #self.g = g # ?
        return g
        
    def load_projection_rows(self, fileName=None, inds=None):
        """load selected rows of projections
        
        Args:
            fileName (string): full path
        """
        if fileName is None:
            if self.data_type == self.RAW or self.data_type == self.RAW_DARK_SUBTRACTED:
                if self.raw_scan_file is None:
                    print('Error: data_type is raw, but raw_scan_file is not specified!')
                    return
                fileName = os.path.join(self.path, self.raw_scan_file)
            else:
                if self.projection_file is None:
                    print('Error: projection_file is not specified!')
                    return
                fileName = os.path.join(self.path, self.projection_file)
        if os.path.isfile(fileName) == False:
            print('Error: ' + str(fileName) + ' does not exist!')
            return
        dataFolder, baseFileName = os.path.split(fileName)
        if "sino" in baseFileName:
            if inds is not None:
                g = np.zeros((self.leapct.get_numAngles(), inds[1]-inds[0]+1, self.leapct.get_numCols()),dtype=np.float32)
            else:
                g = np.zeros((self.leapct.get_numAngles(), self.leapct.get_numRows(), self.leapct.get_numCols()),dtype=np.float32)
            g = np.swapaxes(g, 0, 1)
            g = self.leapct.load_data(fileName, x=g, fileRange=inds, rowRange=None, colRange=None)
            g = np.swapaxes(g, 0, 1)
            g = np.ascontiguousarray(g, dtype=np.float32)
        else:
            g = self.leapct.load_data(fileName, x=None, fileRange=None, rowRange=inds, colRange=None)
        #self.g = g # ?
        return g
    
    def save_projection_angles(self, g, seq_offset=0):
        #if self.data_type == self.RAW or self.data_type == self.RAW_DARK_SUBTRACTED:
        #    fileName = self.raw_scan_file
        #else:
        #    fileName = self.projection_file
        if self.data_type == self.RAW:
            fileName = 'raw.tif'
        elif self.data_type == self.RAW_DARK_SUBTRACTED:
            fileName = 'rawDarkSub.tif'
        elif self.data_type == self.TRANSMISSION:
            fileName = 'transRad.tif'
        elif self.data_type == self.ATTENUATION:
            fileName = 'attenRad.tif'
        else:
            fileName = 'image.tif'
        fileName = os.path.join(self.path, self.outputDir, fileName)
        
        self.leapct.save_projections(fileName, g, seq_offset)
        return fileName
        
    def save_projection_rows(self, g, seq_offset=0):
        if self.data_type == self.RAW:
            fileName = 'sino_raw.tif'
        elif self.data_type == self.RAW_DARK_SUBTRACTED:
            fileName = 'sino_rawDarkSub.tif'
        elif self.data_type == self.TRANSMISSION:
            fileName = 'sino_trans.tif'
        elif self.data_type == self.ATTENUATION:
            fileName = 'sino.tif'
        else:
            fileName = 'sino.tif'
        fileName = os.path.join(self.path, self.outputDir, fileName)
        
        g = np.swapaxes(g, 0, 1)
        self.leapct.save_projections(fileName, g, seq_offset)
        g = np.swapaxes(g, 0, 1)
        g = np.ascontiguousarray(g, dtype=np.float32)
        return fileName
    
    
    ###################################################################################################################
    ###################################################################################################################
    # DATA
    ###################################################################################################################
    ###################################################################################################################
    def set_projection_data(self, g):
        self.g = g
        
    def clear_projection_data(self):
        del self.g
        self.g = None
        
    def set_volume_data(self, f):
        self.f = f
        
    def clear_volume_data(self):
        del self.f
        self.f = None
        
    def available_RAM(self):
        """Returns the amount of available CPU RAM in GB"""
        try:
            import psutil
            return psutil.virtual_memory()[1]/2**30
        except:
            print('Error: cannot load psutil module which is used to calculate the amount of available CPU RAM!')
            return 0.0
            
    def total_RAM(self):
        """Returns the total amount of CPU RAM in GB"""
        try:
            import psutil
            return psutil.virtual_memory().total/2**30
        except:
            print('Error: cannot load psutil module which is used to calculate the total amount of CPU RAM!')
            return 0.0
    
    def memory_used_by_array(self, x):
        if x is None:
            return 0.0
        else:
            return x.nbytes / 2.0**30
        
    def memory_usage(self):
        return self.memory_used_by_array(self.g) + self.memory_used_by_array(self.f)
        
    
    ###################################################################################################################
    ###################################################################################################################
    # SPECTRA
    ###################################################################################################################
    ###################################################################################################################
    def source_spectra_defined(self):
        if self.kV >= 1.0:
            return True
        elif self.source_spectra_file is not None and os.path.isfile(self.source_spectra_file):
            return True
        else:
            return False
    
    def detector_response_defined(self):
        if self.detector_response_file is not None and os.path.isfile(self.detector_response_file):
            return True
        elif self.detector_response is not None:
            return True
        else:
            return False
    
    def set_source_spectra(self, kV, takeOffAngle=11.0, Z=74):
        self.kV = kV
        self.takeoff_angle = takeOffAngle
        self.anode_material = Z
    
    def add_filter(self, material, mass_density, thickness):
        if self.xray_filters is None:
            self.xray_filters = [material, mass_density, thickness]
        else:
            self.xray_filters = (self.xray_filters, [material, mass_density, thickness])
    
    def clear_filters(self):
        self.xray_filters = None
        
    def set_detector_response(self, material, mass_density, thickness):
        self.detector_response = [material, mass_density, thickness]
        
    def clear_detector_response(self):
        self.detector_response = None
    
    def source_spectra(self, do_normalize=False):
        if has_physics == False:
            print('Error: XrayPhysics library not found!')
            return None, None
        elif self.source_spectra_defined() == False:
            print('Error: spectra not defined!')
            return None, None
        else:
            # Set source spectra
            if self.source_spectra_file is not None and os.path.isfile(self.source_spectra_file):
                Es, s = physics.load_spectra(self.source_spectra_file)
                if Es is None or s is None:
                    return None, None
            else:
                Es, s = physics.simulateSpectra(self.kV, self.takeoff_angle, self.anode_material)
                
                if self.lowest_energy >= 1.0 or self.energy_bin_width >= 1.0:
                    if self.lowest_energy >= 1.0:
                        lowest_energy  = self.lowest_energy
                    else:
                        lowest_energy  = Es[0]
                    if self.energy_bin_width >= 1.0:
                        energy_bin_width  = self.energy_bin_width
                    else:
                        energy_bin_width  = Es[1]-Es[0]
                    N_E = int(np.ceil((Es[-1]-lowest_energy) / energy_bin_width))
                    Es = np.array(range(N_E))*energy_bin_width + lowest_energy
                    Es, s = physics.simulateSpectra(self.kV, self.takeoff_angle, self.anode_material, Es)
                    
            
            # Set filter response
            if self.xray_filters is not None:
                if len(self.xray_filters) == 1:
                    s *= physics.filterResponse(self.xray_filters[0], self.xray_filters[1], self.xray_filters[2], Es)
                else:
                    for n in range(len(self.xray_filters)):
                        s *= physics.filterResponse(self.xray_filters[n][0], self.xray_filters[n][1], self.xray_filters[n][2], Es)
            if do_normalize:
                s = physics.normalizeSpectrum(s, Es)
            return Es, s
            
    def detector_response(self, Es):
        if has_physics == False:
            print('Error: XrayPhysics library not found!')
            return None, None
        elif self.detector_response_file is not None and os.path.isfile(self.detector_response_file):
            Es_new, s = physics.load_spectra(self.detector_response_file)
            return Es_new, s
        elif Es is None:
            print('Error: energy bins not defined!')
            return None, None
        elif self.detector_response is not None:
            s = physics.detectorResponse(self.detector_response[0], self.detector_response[1], self.detector_response[2], Es)
            return Es, s
        else:
            s = Es.copy()
            s[:] = 1.0
            return Es, s
    
    def totalSystemSpectralResponse(self, do_normalize=False):
        if has_physics == False:
            print('Error: XrayPhysics library not found!')
            return None, None
        elif self.source_spectra_defined() == False:
            print('Error: spectra not defined!')
            return None, None
        else:
            Es, s = self.source_spectra()
            if Es is None or s is None:
                return None, None

            dont_care, d = self.detector_response(Es)
            s *= d
            
            if do_normalize:
                s = physics.normalizeSpectrum(s, Es)
            return Es, s
            
            
    ###################################################################################################################
    ###################################################################################################################
    # PREPROCESSING ALGORITHMS
    ###################################################################################################################
    ###################################################################################################################
    def makeAttenuationRadiographs(self, ROI=None):
        #preprocessing_algorithms.makeAttenuationRadiographs(self.leapct, ...)
        pass
        
    def outlierCorrection(self, threshold=0.03, windowSize=3, isAttenuationData=True):
        #preprocessing_algorithms.outlierCorrection(self.leapct, ...)
        pass
        
    def outlierCorrection_highEnergy(self, isAttenuationData=True):
        #preprocessing_algorithms.outlierCorrection_highEnergy(self.leapct, ...)
        pass
        
    def detectorDeblur_FourierDeconv(self, H, WienerParam=0.0, isAttenuationData=True):
        #preprocessing_algorithms.detectorDeblur_FourierDeconv(self.leapct, ...)
        pass
        
    def detectorDeblur_RichardsonLucy(self, H, numIter=10, isAttenuationData=True):
        #preprocessing_algorithms.detectorDeblur_RichardsonLucy(self.leapct, ...)
        pass
        
    def ringRemoval_fast(self, delta, numIter, maxChange):
        #preprocessing_algorithms.ringRemoval_fast(self.leapct, ...)
        pass
        
    def ringRemoval_median(self, threshold=0.0, windowSize=5, numIter=1):
        #preprocessing_algorithms.ringRemoval_median(self.leapct, ...)
        pass
        
    def ringRemoval(self, delta, beta, numIter):
        #preprocessing_algorithms.ringRemoval(self.leapct, ...)
        pass
        
    def parameter_sweep(self, values, param='centerCol', iz=None, algorithmName='FBP'):
        #preprocessing_algorithms.parameter_sweep(self.leapct, ...)
        pass
        
    
    ###################################################################################################################
    ###################################################################################################################
    # RECONSTRUCTION ALGORITHMS
    ###################################################################################################################
    ###################################################################################################################
    def project(self):
        #self.leapct.project(g, f)
        pass
        
    def backproject(self):
        #self.leapct.backproject(g, f)
        pass
        
    def FBP(self):
        #self.leapct.FBP(g, f)
        pass
        
    def inconsistencyReconstruction(self):
        #self.leapct.inconsistencyReconstruction(g, f)
        pass
        
    def SIRT(self, numIter, mask=None):
        #self.leapct.SIRT(g, f, numIter, mask)
        pass
        
    def SART(self, numIter, numSubsets=1, mask=None):
        #self.leapct.SART(g, f, numIter, numSubsets, mask)
        pass
        
    def ASDPOCS(self, numIter, numSubsets, numTV, filters=None, mask=None):
        #self.leapct.ASDPOCS(g, f, numIter, numSubsets, numTV, filters, mask)
        pass
        
    def LS(self, numIter, preconditioner=None, nonnegativityConstraint=True):
        #self.leapct.LS(g, f, numIter, preconditioner, nonnegativityConstraint)
        pass
        
    def WLS(self, numIter, W=None, preconditioner=None, nonnegativityConstraint=True):
        #self.leapct.WLS(g, f, numIter, W, preconditioner, nonnegativityConstraint)
        pass
        
    def RLS(self, numIter, filters=None, preconditioner=None, nonnegativityConstraint=True):
        #self.leapct.RLS(g, f, numIter, filters, preconditioner, nonnegativityConstraint)
        pass
        
    def RWLS(self, numIter, filters=None, W=None, preconditioner=None, nonnegativityConstraint=True):
        #self.leapct.RWLS(g, f, numIter, filters, W, preconditioner, nonnegativityConstraint)
        pass
        
    def DLS(self, numIter, preconditionerFWHM=1.0, nonnegativityConstraint=False, dimDeriv=2):
        #self.leapct.DLS(g, f, numIter, preconditionerFWHM, nonnegativityConstraint, dimDeriv)
        pass
        
    def RDLS(self, numIter, filters=None, preconditionerFWHM=1.0, nonnegativityConstraint=False, dimDeriv=1):
        #self.leapct.RDLS(g, f, numIter, filters, preconditionerFWHM, nonnegativityConstraint, dimDeriv)
        pass
        
    def MLTR(self, numIter, numSubsets=1, filters=None, mask=None):
        #self.leapct.MLTR(g, f, numIter, numSubsets, filters, mask)
        pass
    