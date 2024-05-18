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
import numpy as np
from leapctype import *
from preprocessing_algorithms import *

try:
    from xrayphysics import *
    has_physics = True
except:
    has_physics = False

class leapctserver:

    def __init__(self, leapct=None):
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
        self.path = None
        
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

        
        ### Section III: spectra parameters
        self.lowest_energy = -1.0
        self.energy_bin_width = -1.0
        self.kV = None
        self.takeoff_angle = 11.0
        self.anode_material = 74
        self.xray_filters = None
        self.detector_response = None
        
    def available_RAM(self):
        """Returns the amount of available CPU RAM in GB"""
        try:
            import psutil
            return psutil.virtual_memory()[1]/2**30
        except:
            print('Error: cannot load psutil module which is used to calculate the amount of available CPU RAM!')
            return 0.0
    
    
    ###################################################################################################################
    ###################################################################################################################
    # FILE NAMES
    ###################################################################################################################
    ###################################################################################################################
    def set_path(self, path):
        self.path = path
        
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
    
    def load_projection_angles(self, inds=None):
        pass
        
    def load_projection_rows(self, inds=None):
        pass
    
    
    ###################################################################################################################
    ###################################################################################################################
    # DATA
    ###################################################################################################################
    ###################################################################################################################
    def set_projection_data(self, g):
        self.g = g
        
    def clear_projection_data(self):
        self.g = None
        
    def set_volume_data(self, f):
        self.f = f
        
    def clear_volume_data(self):
        self.f = None
    
    
    ###################################################################################################################
    ###################################################################################################################
    # SPECTRA
    ###################################################################################################################
    ###################################################################################################################
    def source_spectra_defined(self):
        if self.kV >= 1.0:
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
        """
        FIXME: does not yet work with files
        """
        if has_physics == False:
            print('Error: XrayPhysics library not found!')
            return None, None
        elif self.source_spectra_defined() == False:
            print('Error: spectra not defined!')
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
        """
        FIXME: does not yet work with files
        """
        if has_physics == False:
            print('Error: XrayPhysics library not found!')
            return None, None
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
        """
        FIXME: does not yet work with files
        """
        if has_physics == False:
            print('Error: XrayPhysics library not found!')
            return None, None
        elif self.source_spectra_defined() == False:
            print('Error: spectra not defined!')
            return None, None
        else:
            Es, s = self.source_spectra()

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
        pass
        
    def outlierCorrection(self, threshold=0.03, windowSize=3, isAttenuationData=True):
        pass
        
    def outlierCorrection_highEnergy(self, isAttenuationData=True):
        pass
        
    def detectorDeblur_FourierDeconv(self, H, WienerParam=0.0, isAttenuationData=True):
        pass
        
    def detectorDeblur_RichardsonLucy(self, H, numIter=10, isAttenuationData=True):
        pass
        
    def ringRemoval_fast(self, delta, numIter, maxChange):
        pass
        
    def ringRemoval_median(self, threshold=0.0, windowSize=5, numIter=1):
        pass
        
    def ringRemoval(self, delta, beta, numIter):
        pass
        
    def parameter_sweep(self, values, param='centerCol', iz=None, algorithmName='FBP'):
        pass
        
    
    ###################################################################################################################
    ###################################################################################################################
    # RECONSTRUCTION ALGORITHMS
    ###################################################################################################################
    ###################################################################################################################
    def project(self):
        pass
        
    def backproject(self):
        pass
        
    def FBP(self):
        pass
        
    def inconsistencyReconstruction(self):
        pass
        
    def SIRT(self, numIter, mask=None):
        pass
        
    def SART(self, numIter, numSubsets=1, mask=None):
        pass
        
    def ASDPOCS(self, numIter, numSubsets, numTV, filters=None, mask=None):
        pass
        
    def LS(self, numIter, preconditioner=None, nonnegativityConstraint=True):
        pass
        
    def WLS(self, numIter, W=None, preconditioner=None, nonnegativityConstraint=True):
        pass
        
    def RLS(self, numIter, filters=None, preconditioner=None, nonnegativityConstraint=True):
        pass
        
    def RWLS(self, numIter, filters=None, W=None, preconditioner=None, nonnegativityConstraint=True):
        pass
        
    def DLS(self, numIter, preconditionerFWHM=1.0, nonnegativityConstraint=False, dimDeriv=2):
        pass
        
    def RDLS(self, numIter, filters=None, preconditionerFWHM=1.0, nonnegativityConstraint=False, dimDeriv=1):
        pass
        
    def MLTR(self, numIter, numSubsets=1, filters=None, mask=None):
        pass
    