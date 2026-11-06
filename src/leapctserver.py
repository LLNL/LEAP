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
################################################################################
import os
import sys
import uuid
import numpy as np
import matplotlib.pyplot as plt
from leapctype import *
import leap_preprocessing_algorithms
has_physics = True

root_path = os.path.dirname(os.path.realpath(__file__))

class leapctserver:
    """ This class handles many high-level tasks for LEAP-CT, including file I/O, data chunking, meta-data I/O, and integration of XrayPhysics
    
    :ivar leapct: tomographicModels (main LEAP-CT class) object, if none is provided on initialization, one will be created
    :ivar path(str): full path of where the data is stored and where the output will be saved; one must have R/W permissions to this path
    :ivar outputDir(str): subfolder of path where the output will be saved; if it does not exist it will be created
    :ivar air_scan_file(str): file name of the air scan file (relative to path)
    :ivar dark_scan_file(str): file name of the dark scan file (relative to path)
    :ivar raw_scan_file(str): file name of the raw radiograph sequence (relative to path)
    :ivar projection_file(str): file name of transmission or attenuation radiography sequence (relative to path)
    :ivar data_type(int): tags the data as RAW, RAW_DARK_SUBTRACTED, TRANSMISSION, or, ATTENUATION
    """

    def __init__(self, leapct=None, path=None, outputDir=None):
        if leapct is None:
            self.leapct = tomographicModels()
        elif isinstance(leapct, str):
            path = leapct
            self.leapct = tomographicModels()
        else:
            self.leapct = leapct
        
        if has_physics:
            self.physics = self.leapct
        else:
            self.physics = None
        self.physics.use_mm()
        
        self.leapct_backup = tomographicModels()
            
        self.reset(path, outputDir)
        
    def reset(self, path=None, outputDir=None):
    
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
        self.spectra_model_file = None # not sure what this is for, not currently being used
        
        # File name where source spectra is stored
        self.source_spectra_file = None
        
        # File name where detector response is stored
        self.detector_response_file = None

        # Tags for the projection data type
        [self.UNSPECIFIED, self.RAW, self.RAW_DARK_SUBTRACTED, self.TRANSMISSION, self.ATTENUATION] = [0, 1, 2, 3, 4]
        self.data_type = self.UNSPECIFIED

        # Output file name
        self.outName = None
        
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
            else:
                self.max_CPU_memory_usage = min(physicalMemory - 4.0, 0.95*physicalMemory) # reserve 4 GB of memory; this is likely too much
        else:
            self.max_CPU_memory_usage = 128.0
        if self.max_CPU_memory_usage > 1.0:
            self.max_CPU_memory_usage = np.floor(self.max_CPU_memory_usage)
            
        # Section III: data chunking
        [self.PROJECTION, self.DETECTOR_ROW, self.Z_SLICE] = [0, 1, 2]
        self.chunking_type = self.PROJECTION
        self.numOverlap = 0
        self.num_proj = 0
        self.num_vol = 0
        self.scratch_space = 0.125 # extra memory reserved
        self.chunk_size = 0
        
        ### Section IV: spectra parameters
        self.reference_energy = -1.0
        self.lowest_energy = -1.0
        self.energy_bin_width = -1.0
        self.kV = None
        self.takeoff_angle = 22.0
        self.anode_normal = None #np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.anode_material = 74
        self.source_filters = None
        self.detector_filters = None
        self.detector_response_model = None
        self.object_model = None
        
        self.init_angle = 0.0
        self.angular_range = 0.0
        self.angular_step = 0.0
        self.num_angles = 0.0
        
        ### Other
        self.default_algorithms = None
        #self.leapct.reset() # use clearAll
        self.lastImage = None
        
        self.restore_defaults()
        
    def clearAll(self):
        self.reset()
        self.leapct.reset()
        
    def save_defaults(self, additionalFile=None):
        defaults_file = os.path.join(root_path, "leapctserver_defaults.txt")
        f = open(defaults_file, "w")
        
        f.write('max_CPU_memory_usage = ' + str(self.max_CPU_memory_usage) + '\n')
        
        f.write('GPUs = [')
        data = self.leapct.get_gpus()
        f.write(','.join(str(i) for i in data))
        f.write(']\n')
        
        f.write('numTVneighbors = ' + str(self.leapct.get_numTVneighbors()) + '\n')
        f.write('backprojector = ' + str(self.leapct.get_projector()) + '\n')
        
        if self.default_algorithms is not None and len(self.default_algorithms) > 0:
            f.write('default_algorithms = ' + str(self.default_algorithms) + '\n')
        
        if additionalFile is not None and len(additionalFile) > 0 and os.path.isfile(additionalFile):
            f.write(additionalFile + '\n')
        
        f.close()
        print('leapctserver defaults saved to: ' + str(defaults_file))
        
    def restore_defaults(self):
        defaults_file = os.path.join(root_path, "leapctserver_defaults.txt")
        if os.path.isfile(defaults_file):
            self.load_parameters(defaults_file)
        
    def is_number(self, s):
        if s is None:
            return False
        try:
            float(s)
            return True
        except ValueError:
            return False
    
    def print_parameters(self):
        print('\n======== File I/O ========')
        if self.path is not None:
            print('path = ', self.path)
        if self.air_scan_file is not None:
            print('air_scan_file = ', self.air_scan_file)
        if self.dark_scan_file is not None:
            print('dark_scan_file = ', self.dark_scan_file)
        if self.raw_scan_file is not None:
            print('raw_scan_file = ', self.raw_scan_file)
        if self.reconstruction_file is not None:
            print('reconstruction_file = ', self.reconstruction_file)
        print('')
        
        print('======== Physics ========')
        if self.source_spectra_file is not None and len(self.source_spectra_file) > 0:
            print('source_spectra_file = ', self.source_spectra_file)
        elif self.kV is not None and self.kV > 0.0:
            print('kV = ', self.kV)
            print('anode_material = ', self.anode_material)
            print('takeoff_angle = ', self.takeoff_angle)
            if self.anode_normal is not None:
                print('anode_normal = ', self.anode_normal)
        if self.source_filters is not None:
            print('source_filters = ', self.source_filters)
        if self.detector_filters is not None:
            print('detector_filters = ', self.detector_filters)
        if self.detector_response_model is not None:
            print('detector_response_model = ', self.detector_response_model)
        if self.object_model is not None:
            print('object_model = ', self.object_model)
        if self.reference_energy > 0.0:
            print('reference_energy = ', self.reference_energy)
        
        self.leapct.print_parameters()
        """
        self.dark_scan_file = None
        self.raw_scan_file = None
        self.projection_file = None
        self.reconstruction_file = None
        """
    
    ###################################################################################################################
    ###################################################################################################################
    # FILE I/O
    ###################################################################################################################
    ###################################################################################################################
    def set_path(self, path):
        if os.path.exists(fullPath):
            self.path = path
        else:
            print('Error: specified path does not exist')
        
    def clear_path(self):
        self.path = None
        
    def create_outputDir(self):
        fullPath = os.path.join(self.path, self.outputDir)
        if not os.path.exists(fullPath):
            os.makedirs(fullPath)
        
    def save_image_file(self, fileName, x, use_outputDir=True):
        """Save 2D data to file (tif sequence, nrrd, or npy)"""
        if x is None:
            return False
        if len(x.shape) != 2:
            return False
        if use_outputDir:
            fullPath = os.path.join(self.path, self.outputDir, fileName)
            self.create_outputDir()
        else:
            fullPath = os.path.join(self.path, fileName)
        volFilePath, dontCare = os.path.split(fullPath)
        if os.path.isdir(volFilePath) == False or os.access(volFilePath, os.W_OK) == False:
            print('Folder to save data either does not exist or not accessible!')
            return False
            
        if has_torch == True and type(x) is torch.Tensor:
            x = x.cpu().detach().numpy()
            
        if fullPath.endswith('.npy'):
            np.save(fullPath, x)
            return True
        elif fullPath.endswith('.nrrd'):
            try:
                import nrrd
                
                # https://pynrrd.readthedocs.io/en/latest/examples.html
                #header = {'units': ['mm', 'mm', 'mm'], 'spacings': [T, T, T], 'axismins': [offset_0, offset_1, offset_2], 'thicknesses': [T, T, T],}
                #nrrd.write(fileName, x, header)
                nrrd.write(fullPath, x)
                return True
            except:
                print('Error: Failed to load nrrd library!')
                print('To install this package do: pip install pynrrd')
                return False
        elif fullPath.endswith('.tif') or fullPath.endswith('.tiff'):
            return self.leapct.save_tif(fullPath, x)
            """
            try:
                #from PIL import Image
                import imageio
                
                imageio.imwrite(fullPath, x)
                return True
                
            except:
                #print('Error: Failed to load PIL library!')
                #print('To install this package do: pip install Pillow')
                print('Error: Failed to load imageio library!')
                print('To install PIL do: pip install imageio')
                return False
            """
        else:
            print('Error: must be a tif, npy, or nrrd file!')
            return False
        
    def read_1D(self, fileName):
        if fileName is None:
            return None
        fullPath = os.path.join(self.path, fileName)
        if fullPath.endswith('.npy'):
            if os.path.isfile(fullPath) == False:
                print('file does not exist')
                return None
            else:
                x = np.array(np.load(fullPath), dtype=np.float32)
                return x
        elif fullPath.endswith('.txt'):
            data = []
            try:
                with open(fullPath, 'r') as file:
                    for line in file:
                        if line[0] != '#':
                            data.append(float(line))
                x = np.array(data, dtype=np.float32)
                return x
            except FileNotFoundError:
                print('Error: failed to open the file ', fullPath)
                return None
            except Exception as e:
                print('Error occured while loading the data', str(e))
                return None
        else:
            print('Error: read_1D currently only works for npy and txt files')
            return None
        
    def read_image_file(self, fileName, rowRange=None, colRange=None, shape=None, dtype=np.float32):
        if fileName is None:
            return None
        if isinstance(fileName, int) or isinstance(fileName, float):
            return float(fileName)
        fullPath = os.path.join(self.path, fileName)
        if rowRange is not None:
            if len(rowRange) != 2 or rowRange[0] > rowRange[1] or rowRange[0] < 0 or rowRange[1] < 0:
                print('Error: rowRange must be a list of two positive numbers')
                return None
        if colRange is not None:
            if len(colRange) != 2 or colRange[0] > colRange[1] or colRange[0] < 0 or colRange[1] < 0:
                print('Error: colRange must be a list of two positive numbers')
                return None
        
        if fullPath.endswith('.npy'):
            if os.path.isfile(fullPath) == False:
                print('file does not exist')
                return None
            else:
                x = np.load(fullPath)
        elif fullPath.endswith('.nrrd'):
            if os.path.isfile(fullPath) == False:
                print('file does not exist')
                return None
            try:
                import nrrd
                x, header = nrrd.read(fullPath)
            except:
                print('Error: Failed to load nrrd library!')
                print('To install this package do: pip install pynrrd')
                return None
        elif fullPath.endswith('.tif') or fullPath.endswith('.tiff'):
            
            try:
                #from PIL import Image
                import imageio
                hasPIL = True
            except:
                #print('Error: Failed to load PIL or glob library!')
                #print('To install PIL do: pip install Pillow')
                print('Error: Failed to load imageio or glob library!')
                print('To install PIL do: pip install imageio')
                return None
            if hasPIL == True:
                if os.path.isfile(fullPath) == False:
                    print('file does not exist')
                    return None
                else:
                    x = np.array(imageio.imread(fullPath), dtype=np.float32)
        elif fullPath.endswith('.raw') or fullPath.endswith('.sdt'):
            if shape is None or dtype is None:
                print('Error: must specify shape and dtype for raw file types')
                return None
            else:
                x = self.read_raw_file(fullPath, shape, dtype, rowRange, colRange)
        else:
            try:
                x = float(fileName)
                return x
            except:
                print('Error: must be a tif, tiff, std, raw, npy, or nrrd file!')
                return None
            
        return self.crop_image(x, rowRange, colRange)
        
    def read_raw_file(self, fileName, shape=None, dtype=np.float32, rowRange=None, colRange=None):
        x = np.fromfile(fileName, dtype)
        if shape is not None and len(shape) == 2:
            x = x.reshape((shape[0], shape[1]))
            x = self.crop_image(x, rowRange, colRange)
        return x
        
    def crop_image(self, x, rowRange=None, colRange=None):
        if x is None:
            return None
        if len(x.shape) != 2:
            x = np.ascontiguousarray(x, dtype=np.float32)
        else:
            if rowRange is not None and len(rowRange) == 2 and rowRange[1] < x.shape[0]:
                x = x[rowRange[0]:rowRange[1]+1,:]
            if colRange is not None and len(colRange) == 2 and colRange[1] < x.shape[1]:
                x = x[:,colRange[0]:colRange[1]+1]
            x = np.ascontiguousarray(x, dtype=np.float32)
        return x
        
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
        self.create_outputDir()
        if self.geometry_file is None or len(self.geometry_file) == 0:
            self.geometry_file = os.path.join(self.outputDir, 'geometry.txt')
        elif self.geometry_file.startswith(self.outputDir) == False:
            self.geometry_file = os.path.join(self.outputDir, self.geometry_file)
        fullPath = os.path.join(self.path, self.geometry_file)
        return self.leapct.save_parameters(fullPath)
    
    def load_geometry_file(self, inputFile=None):
        if inputFile is not None:
            self.geometry_file = inputFile
        if self.geometry_file is not None and len(self.geometry_file) > 0:
            if self.geometry_file.startswith(self.path):
                fullPath = self.geometry_file
            else:
                fullPath = os.path.join(self.path, self.geometry_file)
            return self.leapct.load_parameters(fullPath)
        else:
            return False
    
    def save_spectra_model(self):
        if has_physics:
            if self.source_spectra_defined():
                Es, s = self.source_spectra()
                self.physics.save_spectra(self.source_spectra_file, s, Es)
                if self.detector_response_defined():
                    Es, d = self.detector_response(Es)
                    self.physics.save_spectra(self.detector_response_file, d, Es)
        
    def save_parameters(self, fileName=None):
        """Saves CT geometry, CT volume, and all spectra parameters to file"""
        self.create_outputDir()
        if fileName is None or len(fileName) == 0:
            fileName = os.path.join(self.path, self.outputDir, 'leapct_params.txt')
        elif fileName.startswith(self.path) == False and os.path.isabs(fileName) == False:
            fileName = os.path.join(self.path, self.outputDir, fileName)
        self.save_geometry_file()
        
        """
        self.path
        self.air_scan_file
        self.dark_scan_file
        self.raw_scan_file
        self.projection_file
        self.reconstruction_file
        self.geometry_file
        """
        f = open(fileName, "w")
        if self.path is not None and len(self.path) > 0:
            f.write('path = ' + self.path + '\n')
        if self.air_scan_file is not None and len(self.air_scan_file) > 0:
            f.write('air_scan_file = ' + self.air_scan_file + '\n')
        if self.dark_scan_file is not None and len(self.dark_scan_file) > 0:
            f.write('dark_scan_file = ' + self.dark_scan_file + '\n')
        if self.raw_scan_file is not None and len(self.raw_scan_file) > 0:
            f.write('raw_scan_file = ' + self.raw_scan_file + '\n')
        if self.projection_file is not None and len(self.projection_file) > 0:
            f.write('projection_file = ' + self.projection_file + '\n')
        if self.reconstruction_file is not None and len(self.reconstruction_file) > 0:
            f.write('reconstruction_file = ' + self.reconstruction_file + '\n')
        if self.geometry_file is not None and len(self.geometry_file) > 0:
            f.write('geometry_file = ' + self.geometry_file + '\n')
        
        if self.data_type == self.RAW:
            f.write('data_type = RAW\n')
        elif self.data_type == self.RAW_DARK_SUBTRACTED:
            f.write('data_type = RAW_DARK_SUBTRACTED\n')
        elif self.data_type == self.TRANSMISSION:
            f.write('data_type = TRANSMISSION\n')
        elif self.data_type == self.ATTENUATION:
            f.write('data_type = ATTENUATION\n')
            
        #"""
        if self.source_spectra_file is not None and len(self.source_spectra_file) > 0:
            f.write('source_spectra_file = ' + self.source_spectra_file + '\n')
        elif self.kV is not None and self.kV > 0.0:
            f.write('kV = ' + str(self.kV) + '\n')
            f.write('anode_material = ' + str(self.anode_material) + '\n')
            f.write('takeoff_angle = ' + str(self.takeoff_angle) + '\n')
            if self.anode_normal is not None:
                f.write('anode_normal = ' + str(self.anode_normal) + '\n')
        if self.source_filters is not None:
            f.write('source_filters = ' + str(self.source_filters) + '\n')
        if self.detector_filters is not None:
            f.write('detector_filters = ' + str(self.detector_filters) + '\n')
        if self.detector_response_model is not None:
            f.write('detector_response_model = ' + str(self.detector_response_model) + '\n')
        if self.object_model is not None:
            f.write('object_model = ' + str(self.object_model) + '\n')
        if self.reference_energy > 0.0:
            f.write('reference_energy = ' + str(self.reference_energy) + '\n')
        #"""
        
        #f.write('max_CPU_memory_usage = ' + str(self.max_CPU_memory_usage) + '\n')
        #f.write('GPUs = ' + str(self.leapct.get_gpus()) + '\n')
        
        f.close()
        #self.save_spectra_model()
    
    def load_projections_into_memory(self):
        if self.data_type == self.TRANSMISSION or self.data_type == self.ATTENUATION:
            if self.projection_file is not None and len(self.projection_file) > 0:
                self.g = self.load_projections(self.projection_file)
        else:
            if self.raw_scan_file is not None and len(self.raw_scan_file) > 0:
                self.g = self.load_projections(self.raw_scan_file)

    def load_dark_scan_into_memory(self):
        if self.dark_scan_file is not None and len(self.dark_scan_file) > 0:
            return self.read_image_file(self.dark_scan_file)
        else:
            return None
            
    def load_air_scan_into_memory(self):
        if self.air_scan_file is not None and len(self.air_scan_file) > 0:
            return self.read_image_file(self.air_scan_file)
        else:
            return None
        
    def load_volume_into_memory(self):
        if self.reconstruction_file is not None and len(self.reconstruction_file) > 0:
            self.f = self.load_volume(self.reconstruction_file)
    
    def load_projections(self, fileName=None):
        return self.load_projection_angles(fileName)
    
    def load_projection_angles(self, fileName=None, inds=None):
        """load selected angles of projections
        
        Args:
            fileName (string): full path
            inds (list of two integers): specifies the range of projections to load
            
        Returns:
            3D numpy of the projections loaded from file
        """
        if fileName is None:
            if self.data_type == self.RAW or self.data_type == self.RAW_DARK_SUBTRACTED:
                if self.raw_scan_file is None:
                    print('Error: data_type is raw, but raw_scan_file is not specified!')
                    return None
                fileName = self.raw_scan_file
            else:
                if self.projection_file is None:
                    print('Error: projection_file is not specified!')
                    return None
                fileName = self.projection_file
        fullPath = os.path.join(self.path, fileName)
        #if os.path.isfile(fullPath) == False:
        #    print('Error: ' + str(fullPath) + ' does not exist!')
        #    return None
        dataFolder, baseFileName = os.path.split(fullPath)
        if "sino" in baseFileName:
            if inds is not None:
                g = np.zeros((inds[1]-inds[0]+1, self.leapct.get_numRows(), self.leapct.get_numCols()),dtype=np.float32)
            else:
                g = np.zeros((self.leapct.get_numAngles(), self.leapct.get_numRows(), self.leapct.get_numCols()),dtype=np.float32)
            #g = np.swapaxes(g, 0, 1)
            self.leapct.load_data(fullPath, x=g, fileRange=None, rowRange=inds, colRange=None, axis_split=1)
            #g = np.swapaxes(g, 0, 1)
            #g = np.ascontiguousarray(g, dtype=np.float32)
            """
            elif baseFileName.find('*') != -1:
                import imageio
                files = glob.glob(fullPath)
                if inds is None:
                    inds = [0, len(files)-1]
                else:
                    inds[0] = max(0, inds[0])
                    inds[1] = min(len(files)-1, inds[1])
                for n in range(inds[0], inds[1]+1):
                    file = files[n]
                    anImage = np.array(imageio.imread(files[n]))
                    if n == inds[0]:
                        g = np.zeros((len(files), anImage.shape[0], anImage.shape[1]), dtype=np.float32)
                    g[n,:,:] = anImage[:,:]
            """
        else:
            g = self.leapct.load_data(fullPath, x=None, fileRange=inds, rowRange=None, colRange=None)
        #self.g = g # ?
        return g
        
    def load_projection_rows(self, fileName=None, inds=None):
        """load selected rows of projections
        
        Args:
            fileName (string): full path
            inds (list of two integers): specifies the range of detector rows to load
            
        Returns:
            3D numpy of the sinograms loaded from file
        """
        if fileName is None:
            if self.data_type == self.RAW or self.data_type == self.RAW_DARK_SUBTRACTED:
                if self.raw_scan_file is None:
                    print('Error: data_type is raw, but raw_scan_file is not specified!')
                    return None
                fileName = self.raw_scan_file
            else:
                if self.projection_file is None:
                    print('Error: projection_file is not specified!')
                    return None
                fileName = self.projection_file
        fullPath = os.path.join(self.path, fileName)
        #if os.path.isfile(fullPath) == False:
        #    print('Error: ' + str(fullPath) + ' does not exist!')
        #    return None
        dataFolder, baseFileName = os.path.split(fullPath)
        if "sino" in baseFileName:
            if inds is not None:
                g = np.zeros((self.leapct.get_numAngles(), inds[1]-inds[0]+1, self.leapct.get_numCols()),dtype=np.float32)
            else:
                g = np.zeros((self.leapct.get_numAngles(), self.leapct.get_numRows(), self.leapct.get_numCols()),dtype=np.float32)
            #g = np.swapaxes(g, 0, 1)
            g = self.leapct.load_data(fullPath, x=g, fileRange=inds, rowRange=None, colRange=None, axis_split=1)
            #g = np.swapaxes(g, 0, 1)
            #g = np.ascontiguousarray(g, dtype=np.float32)
        else:
            g = self.leapct.load_data(fullPath, x=None, fileRange=None, rowRange=inds, colRange=None)
        #self.g = g # ?
        return g
    
    def save_projection_angles(self, g=None, seq_offset=0, update_params=False):
        """Saves the projection data in a sequence of tif files, one file for each projection angle
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data
            seq_offset (int): the file sequence number for the first file
            
        Returns:
            The base file name of the saved data, if failed to write to file returns None
        """
        if g is None:
            g = self.g
        if g is None:
            print('Error: no projection data exists to save')
            return None
        self.create_outputDir()
        #if self.data_type == self.RAW or self.data_type == self.RAW_DARK_SUBTRACTED:
        #    fileName = self.raw_scan_file
        #else:
        #    fileName = self.projection_file
        fileName = self.get_default_projection_file_name()
        
        if self.outputDir in fileName:
            newFileName = fileName
        else:
            newFileName = os.path.join(self.outputDir, fileName)
        fullPath = os.path.join(self.path, newFileName)
        
        if self.leapct.save_projections(fullPath, g, seq_offset) == True:
            if update_params:
                if self.data_type == self.TRANSMISSION or self.data_type == self.ATTENUATION:
                    self.projection_file = newFileName
                else:
                    self.raw_scan_file = newFileName
            return newFileName
        else:
            return None
            
    def save_volume(self, f=None, seq_offset=0, update_params=False):
        """Saves the volume data in a sequence of tif files, one file for each z-slice
        
        Args:
            f (C contiguous float32 numpy array or torch tensor): volume data
            seq_offset (int): the file sequence number for the first file
            
        Returns:
            The base file name of the saved data, if failed to write to file returns None
        """
        if f is None:
            f = self.f
        if f is None:
            print('Error: no volume data exists to save')
            return None
        self.create_outputDir()
        fileName = 'zslice.tif'
        if self.outputDir in fileName:
            newFileName = fileName
        else:
            newFileName = os.path.join(self.outputDir, fileName)
        fullPath = os.path.join(self.path, newFileName)
        
        if self.leapct.save_volume(fullPath, f, seq_offset) == True:
            if update_params:
                self.reconstruction_file = newFileName
            return newFileName
        else:
            return None
            
    def load_volume(self, fileName=None, inds=None):
        if fileName is None:
            if self.reconstruction_file is None or len(self.reconstruction_file) == 0:
                print('Error: reconstruction_file is not defined!')
                return None
            fileName = self.reconstruction_file
        fullPath = os.path.join(self.path, fileName)
        #if os.path.isfile(fullPath) == False:
        #    print('Error: ' + str(fullPath) + ' does not exist!')
        #    return None
        f = self.leapct.load_data(fullPath, x=None, fileRange=inds, rowRange=None, colRange=None)
        return f
        
    def get_default_projection_file_name(self):
        if self.outName is not None and len(self.outName) > 0:
            return self.outName
        elif self.data_type == self.RAW:
            return 'raw.tif'
        elif self.data_type == self.RAW_DARK_SUBTRACTED:
            return 'rawDarkSub.tif'
        elif self.data_type == self.TRANSMISSION:
            return 'transRad.tif'
        elif self.data_type == self.ATTENUATION:
            return 'attenRad.tif'
        else:
            return 'image.tif'
    
    def save_projection_rows(self, g, seq_offset=0, update_params=False):
        """Saves the projection data in a sequence of tif files, one file for each detector row
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data
            seq_offset (int): the file sequence number for the first file
            
        Returns:
            The base file name of the saved data, if failed to write to file returns None
        """
        self.create_outputDir()
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
            
        if self.outputDir in fileName:
            newFileName = fileName
        else:
            newFileName = os.path.join(self.outputDir, fileName)
        fullPath = os.path.join(self.path, newFileName)
        
        #g = np.swapaxes(g, 0, 1)
        #isSuccessful = self.leapct.save_projections(fullPath, g, seq_offset, axis_split=1)
        #g = np.swapaxes(g, 0, 1)
        #g = np.ascontiguousarray(g, dtype=np.float32)
        
        if self.leapct.save_projections(fullPath, g, seq_offset, axis_split=1) == True:
            if update_params:
                if self.data_type == self.TRANSMISSION or self.data_type == self.ATTENUATION:
                    self.projection_file = newFileName
                else:
                    self.raw_scan_file = newFileName
            return newFileName
        else:
            return None
    
    def get_zslice(self, iz, thickness=1):
        # TODO: read from file if not loaded in memory
        if self.f is None:
            return None
        elif self.leapct.ct_volume_defined() == False:
            return None
        else:
            return self.get_2Dsubset(self.f, iz, 0, thickness)
            
    def get_yslice(self, iy, thickness=1):
        # TODO: read from file if not loaded in memory
        if self.f is None:
            return None
        elif self.leapct.ct_volume_defined() == False:
            return None
        else:
            return self.get_2Dsubset(self.f, iy, 1, thickness)
            
    def get_xslice(self, ix, thickness=1):
        # TODO: read from file if not loaded in memory
        if self.f is None:
            return None
        elif self.leapct.ct_volume_defined() == False:
            return None
        else:
            return self.get_2Dsubset(self.f, ix, 2, thickness)
            
    def get_projection(self, iphi, thickness=1):
        # TODO: read from file if not loaded in memory
        if self.g is None:
            return None
        elif self.leapct.ct_geometry_defined() == False:
            return None
        else:
            return self.get_2Dsubset(self.g, iphi, 0, thickness)
            
    def get_sinogram(self, irow, thickness=1):
        # TODO: read from file if not loaded in memory
        if self.g is None:
            return None
        elif self.leapct.ct_geometry_defined() == False:
            return None
        else:
            return self.get_2Dsubset(self.g, irow, 1, thickness)
            
    def get_2Dsubset(self, x, ind, axis, thickness=1):
        if x is None:
            return None
        elif axis < 0 or axis > 3:
            return None
        elif ind < 0 or ind >= x.shape[axis]:
            return None
        else:
            thickness = min(max(1,thickness), ind+1, x.shape[axis]-ind)
            if axis == 0:
                slice = np.empty((x.shape[1], x.shape[2]), dtype=np.float32)
            elif axis == 1:
                slice = np.empty((x.shape[0], x.shape[2]), dtype=np.float32)
            else:
                slice = np.empty((x.shape[0], x.shape[1]), dtype=np.float32)
            if thickness == 1:
                if axis == 0:
                    slice[:,:] = x[ind,:,:]
                elif axis == 1:
                    slice[:,:] = x[:,ind,:]
                else:
                    slice[:,:] = x[:,:,ind]
            else:
                ind_min = ind - thickness//2
                ind_max = ind + thickness//2
                if thickness % 2 == 0:
                    w = np.ones(thickness+1, dtype=np.float32)
                    w[0] = 0.5
                    w[-1] = 0.5
                    w = w / float(thickness)
                    if axis == 0:
                        #slice[:,:] = np.tensordot(x[ind_min:ind_max+1,:,:], w, axes=axis)
                        slice[:,:] = np.sum(x[ind_min:ind_max+1,:,:] * w[:,None,None], axis=axis)
                    elif axis == 1:
                        slice[:,:] = np.sum(x[:,ind_min:ind_max+1,:] * w[None,:,None], axis=axis)
                    else:
                        slice[:,:] = np.sum(x[:,:,ind_min:ind_max+1] * w[None,None,:], axis=axis)
                else:
                    if axis == 0:
                        slice[:,:] = np.sum(x[ind_min:ind_max+1,:,:], axes=axis) / float(thickness)
                    elif axis == 1:
                        slice[:,:] = np.sum(x[:,ind_min:ind_max+1,:], axes=axis) / float(thickness)
                    else:
                        slice[:,:] = np.sum(x[:,:,ind_min:ind_max+1], axes=axis) / float(thickness)
            return slice
    
    def extrema(self, x):
        """ Returns (min, max) of x.

        Uses the fast C++ leapct.extrema routine, which is significantly faster
        than numpy for large 3D volumes / projection stacks, and falls back to
        numpy if it cannot be applied (e.g. non-float32 or non-contiguous data).
        """
        try:
            minValue, maxValue = self.leapct.extrema(x)
            if minValue is not None and maxValue is not None:
                return minValue, maxValue
        except Exception:
            pass
        return np.min(x), np.max(x)

    def basic_stats(self, x):
        if x is None:
            return None, None, None, None, None
        else:
            #numpy.histogram(a, bins=10, range=None, density=None, weights=None)
            mu = np.mean(x)
            sigma = np.std(x)
            minValue, maxValue = self.extrema(x)
            return minValue, maxValue, mu, sigma, mu/sigma
    
    ###################################################################################################################
    ###################################################################################################################
    # DATA MANAGEMENT
    ###################################################################################################################
    ###################################################################################################################
    def set_projection_data(self, g):
        self.g = g
        
    def clear_projection_data(self):
        if self.g is not None:
            del self.g
        self.g = None
        
    def set_volume_data(self, f):
        self.f = f
        
    def clear_volume_data(self):
        if self.f is not None:
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
            return float(x.nbytes) / 2.0**30
        
    def memory_usage(self):
        return self.memory_used_by_array(self.g) + self.memory_used_by_array(self.f)
        
        
    def projection_memory(self):
        if self.leapct.ct_geometry_defined():
            N_phis = self.leapct.get_numAngles()
            N_rows = self.leapct.get_numRows()
            N_cols = self.leapct.get_numCols()
            return 4.0 * float(N_phis) * float(N_rows) * float(N_cols) / 2.0**30
        else:
            return 0.0
            
    def volume_memory(self):
        if self.leapct.ct_volume_defined():
            numX = self.leapct.get_numX()
            numY = self.leapct.get_numY()
            numZ = self.leapct.get_numZ()
            return 4.0 * float(numX) * float(numY) * float(numZ) / 2.0**30
        else:
            return 0.0
        
    """
    [self.PROJECTION, self.DETECTOR_ROW, self.Z_SLICE] = [0, 1, 2]
    self.chunking_type = self.PROJECTION
    self.num_proj = 0
    self.num_vol = 0
    self.scratch_space = 0.0 # extra memory reserved
    self.chunk_size = 0
    """
    
    def set_chunk_size(self):
        """Sets the size of the largest chunk that can be used to perform a specific algorithm
        
        BEFORE running this function, one should set the following:
            self.chunking_type
            self.num_proj
            self.num_vol
        
        """
        
        if self.chunking_type == self.PROJECTION:
            if self.leapct.ct_geometry_defined() == False:
                print('Error: CT geometry not defined!')
                return False
        
            self.num_proj = max(1, self.num_proj)
            if self.num_proj * self.projection_memory() < self.max_CPU_memory_usage - self.scratch_space:
                self.chunk_size = self.leapct.get_numAngles()
            else:
                numAngles = self.leapct.get_numAngles()
                #chunk_size * self.num_proj * self.projection_memory() / float(numAngles) = self.max_CPU_memory_usage - self.scratch_space
                self.chunk_size = int(float((self.max_CPU_memory_usage - self.scratch_space) * float(numAngles) / (self.num_proj * self.projection_memory())))
                
                numChunks = int(np.ceil(float(numAngles)/float(self.chunk_size)))
                self.chunk_size = int(np.ceil(float(numAngles)/float(numChunks)))
        
        elif self.chunking_type == self.DETECTOR_ROW:
            if self.leapct.ct_geometry_defined() == False:
                print('Error: CT geometry not defined!')
                return False
        
            self.num_proj = max(1, self.num_proj)
            if self.num_proj * self.projection_memory() < self.max_CPU_memory_usage - self.scratch_space:
                self.chunk_size = self.leapct.get_numRows()
            else:
                numRows = self.leapct.get_numRows()
                self.chunk_size = int(float((self.max_CPU_memory_usage - self.scratch_space) * float(numRows) / (self.num_proj * self.projection_memory())))
                
                numChunks = int(np.ceil(float(numRows)/float(self.chunk_size)))
                self.chunk_size = int(np.ceil(float(numRows)/float(numChunks)))
        
        elif self.chunking_type == self.Z_SLICE:
            self.chunk_size = 1
            self.num_vol = max(1, self.num_vol)
            if self.leapct.ct_volume_defined() == False:
                print('Error: CT volume not defined!')
                return False

            memory_remaining = self.max_CPU_memory_usage - self.scratch_space - self.memory_usage()
            if memory_remaining <= 0.0:
                return False
                
            numZ = float(self.leapct.get_numZ())
            
            if self.num_proj <= 0:
                # Postprocessing Algorithm
                if self.num_vol * self.volume_memory() < self.max_CPU_memory_usage - self.scratch_space:
                    self.chunk_size = numZ
                else:
                    self.chunk_size = int(float((self.max_CPU_memory_usage - self.scratch_space) * float(numZ) / (self.num_vol * self.volume_memory())))
                    
                    numChunks = int(np.ceil(float(numZ)/float(self.chunk_size)))
                    self.chunk_size = int(np.ceil(float(numZ)/float(numChunks)))
            else:
                # Reconstruction Algorithm
                numRows = float(self.leapct.get_numRows())
                while self.chunk_size < numZ:
                    numRows_needed = float(self.leapct.numRowsRequiredForBackprojectingSlab(self.chunk_size))
                    mem_needed = self.num_vol*self.volume_memory()*self.chunk_size/numZ + self.num_proj*self.projection_memory()*numRows_needed/numRows
                    if mem_needed > memory_remaining:
                        self.chunk_size = self.chunk_size-1
                        break
                    self.chunk_size = self.chunk_size + 1
                self.chunk_size = min(self.chunk_size, numZ)
                numChunks = int(np.ceil(float(numZ)/float(self.chunk_size)))
                self.chunk_size = int(np.ceil(float(numZ)/float(numChunks)))
                return True
                
        else:
            print('Error: chunking_type value is invalid')
            self.chunk_size = 0
            
        if self.chunk_size > 0:
            return True
    
    ###################################################################################################################
    ###################################################################################################################
    # SPECTRA
    ###################################################################################################################
    ###################################################################################################################
    def source_spectra_defined(self):
        if self.kV is not None and self.kV >= 1.0:
            return True
        elif self.is_number(self.source_spectra_file):
            return True
        elif self.source_spectra_file is not None and os.path.isfile(self.source_spectra_file):
            return True
        else:
            return False
    
    def detector_response_defined(self):
        if self.detector_response_file is not None and os.path.isfile(self.detector_response_file):
            return True
        elif self.detector_response_model is not None:
            return True
        else:
            return False
    
    def set_source_spectra(self, kV, takeOffAngle=11.0, Z=74):
        self.kV = kV
        self.takeoff_angle = takeOffAngle
        self.anode_material = Z
    
    def add_source_filter(self, material, mass_density, thickness):
        if mass_density is None:
            mass_density = self.physics.massDensity(material)
        if self.source_filters is None:
            self.source_filters = [(material, mass_density, thickness)]
        else:
            self.source_filters.append((material, mass_density, thickness))
    
    def clear_source_filters(self):
        self.source_filters = None
    
    def add_detector_filter(self, material, mass_density, thickness):
        if mass_density is None:
            mass_density = self.physics.massDensity(material)
        if self.detector_filters is None:
            self.detector_filters = [(material, mass_density, thickness)]
        else:
            self.detector_filters.append((material, mass_density, thickness))
    
    def clear_detector_filters(self):
        self.detector_filters = None
    
    # Backward-compatible aliases (x-ray filters were historically stored as a
    # single list; they now map onto the source filters).
    def add_filter(self, material, mass_density, thickness):
        self.add_source_filter(material, mass_density, thickness)
    
    def clear_filters(self):
        self.clear_source_filters()
        
    def set_detector_response(self, material, mass_density, thickness):
        if mass_density is None:
            mass_density = self.physics.massDensity(material)
        self.detector_response_model = [material, mass_density, thickness]
        
    def clear_detector_response(self):
        self.detector_response_model = None
        
    def set_object_model(self, material, mass_density=None):
        """ Sets the object model to a single material (replaces any existing list). """
        if mass_density is None or mass_density == 0.0:
            mass_density = self.physics.massDensity(material)
        self.object_model = [(material, mass_density)]

    def add_object_model(self, material, mass_density=None):
        """ Appends a material to the object model list.

        The object model is stored as a list of (material, mass_density) tuples
        (mirroring source_filters), which allows multiple materials to be specified.
        """
        if mass_density is None or mass_density == 0.0:
            mass_density = self.physics.massDensity(material)
        if self.object_model is None:
            self.object_model = [(material, mass_density)]
        else:
            self._normalize_object_model()
            self.object_model.append((material, mass_density))

    def clear_object_model(self):
        self.object_model = None

    def object_model_materials(self):
        """ Returns the list of object-model material names, or None if unset. """
        if self.object_model is None:
            return None
        self._normalize_object_model()
        return [entry[0] for entry in self.object_model]

    def _normalize_object_model(self):
        """ Normalize self.object_model to a list of (material, mass_density) tuples.

        Handles backward compatibility with the legacy flat [material, density]
        format as well as a bare material-name string.
        """
        om = self.object_model
        if om is None:
            return
        if isinstance(om, str):
            self.object_model = [(om, self.physics.massDensity(om))]
            return
        # Legacy flat format: a single [material, density] pair
        if isinstance(om, (list, tuple)) and len(om) == 2 and isinstance(om[0], str) and isinstance(om[1], (int, float)):
            self.object_model = [(om[0], om[1])]
            return
        normalized = []
        for entry in om:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                normalized.append((entry[0], entry[1]))
            elif isinstance(entry, str):
                normalized.append((entry, self.physics.massDensity(entry)))
            else:
                normalized.append(entry)
        self.object_model = normalized
    
    def source_spectra(self, do_normalize=False):
        if has_physics == False:
            print('Error: XrayPhysics library not found!')
            return None, None
        elif self.source_spectra_defined() == False:
            print('Error: spectra not defined!')
            return None, None
        else:
            # Set source spectra
            if self.is_number(self.source_spectra_file):
                Es = np.array([float(self.source_spectra_file)], dtype=np.float32)
                s = Es.copy()
                s[:] = 1.0
            elif self.source_spectra_file is not None and os.path.isfile(self.source_spectra_file):
                Es, s = self.physics.load_spectra(self.source_spectra_file)
                if Es is None or s is None:
                    return None, None
            else:
                Es, s = self.physics.simulateSpectra(self.kV, self.takeoff_angle, self.anode_material)
                
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
                    Es = np.array(range(N_E), dtype=np.float32)*energy_bin_width + lowest_energy
                    Es, s = self.physics.simulateSpectra(self.kV, self.takeoff_angle, self.anode_material, Es)
                    
            
            # Set filter response (source-side filters attenuate the source spectrum)
            if self.source_filters is not None:
                for n in range(len(self.source_filters)):
                    s *= self.physics.filterResponse(self.source_filters[n][0], self.source_filters[n][1], self.source_filters[n][2], Es)
            if do_normalize:
                self.physics.normalizeSpectrum(s, Es)
            return Es, s
            
    def detector_response(self, Es):
        if has_physics == False:
            print('Error: XrayPhysics library not found!')
            return None, None
        elif self.detector_response_file is not None and os.path.isfile(self.detector_response_file):
            Es_new, s = self.physics.load_spectra(self.detector_response_file)
            return Es_new, s
        elif Es is None:
            print('Error: energy bins not defined!')
            return None, None
        elif self.detector_response_model is not None:
            s = self.physics.detectorResponse(self.detector_response_model[0], self.detector_response_model[1], self.detector_response_model[2], Es)
            self._apply_detector_filters(s, Es)
            return Es, s
        else:
            s = Es.copy()
            s[:] = 1.0
            self._apply_detector_filters(s, Es)
            return Es, s
    
    def _apply_detector_filters(self, s, Es):
        """ Attenuate the detector response in-place by the detector-side filters. """
        if self.detector_filters is not None:
            for n in range(len(self.detector_filters)):
                s *= self.physics.filterResponse(self.detector_filters[n][0], self.detector_filters[n][1], self.detector_filters[n][2], Es)
    
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
            if d is not None:
                s *= d
            
            if do_normalize:
                self.physics.normalizeSpectrum(s, Es)
            return Es, s
            
            
    ###################################################################################################################
    ###################################################################################################################
    # PREPROCESSING ALGORITHMS
    ###################################################################################################################
    ###################################################################################################################
    def grab_single_projection(self, iProj):
        iProj = max(0, min(self.leapct.get_numAngles()-1, iProj))
        if self.g is None:
            aProj = self.load_projection_angles(inds=[iProj, iProj])
            if aProj is None:
                print('Error: failed to load data')
                return None
        else:
            aProj = np.zeros((1, self.g.shape[1], self.g.shape[2]), dtype=np.float32)
            aProj[0,:,:] = self.g[iProj, :, :]
        return aProj
        
    def grab_necessary_sinograms_for_reconstruction(self, iz, numPad=0):
        if self.leapct.ct_geometry_defined() == False or self.leapct.ct_volume_defined() == False:
            return None, None
        if iz < 0 or iz >= self.leapct.get_numZ():
            iz = self.leapct.get_numZ()//2
        rowRange = self.leapct.rowRangeNeededForBackprojection(iz)
        if numPad > 0:
            rowRange[0] = max(0, rowRange[0]-numPad)
            rowRange[1] = min(self.leapct.get_numRows()-1, rowRange[1]+numPad)
        #print(rowRange)
        if rowRange is None:
            return None, None
        
        if self.g is None:
            g_ROI = self.load_projection_rows(inds=rowRange)
        else:
            g_ROI = np.zeros((self.g.shape[0], rowRange[1]-rowRange[0]+1, self.g.shape[2]), dtype=np.float32)
            g_ROI[:,:,:] = self.g[:,rowRange[0]:rowRange[1]+1,:]
        return g_ROI, rowRange
    
    def grab_slices(self, sliceRange):
        if self.f is None:
            f_ROI = self.load_volume(inds=sliceRange)
        else:
            f_ROI = np.zeros((sliceRange[1]-sliceRange[0]+1, self.f.shape[1], self.f.shape[2]), dtype=np.float32)
            f_ROI[:,:,:] = self.f[sliceRange[0]:sliceRange[1]+1,:,:]
        return f_ROI
    
    def stacked_projection(self):
        if self.data_type == self.UNSPECIFIED:
            print('Error: must specify data_type')
            return None
        self.num_proj = 1
        self.numOverlap = 0
        if self.projection_processing_setup() == False:
            return None
        if self.g is not None:
            if self.data_type == self.ATTENUATION:
                if has_torch == True and type(self.g) is torch.Tensor:
                    g_stack = torch.max(self.g,axis=0)
                else:
                    g_stack = np.max(self.g,axis=0)
            else:
                if has_torch == True and type(self.g) is torch.Tensor:
                    g_stack = torch.min(self.g,axis=0)
                else:
                    g_stack = np.min(self.g,axis=0)
            return g_stack
        else:
            g_stack = None
            if self.data_type == self.TRANSMISSION or self.data_type == self.ATTENUATION:
                input_file = self.projection_file
            else:
                input_file = self.raw_scan_file
            
            # Need to process the entire set of projections
            numAngles = self.leapct.get_numAngles()
            numChunks = int(np.ceil(float(numAngles)/float(self.chunk_size)))
            for n in range(numChunks):
                angleStart = n*self.chunk_size
                angleEnd = min(numAngles-1, angleStart + self.chunk_size - 1)
                
                g_chunk = self.load_projection_angles(input_file, [angleStart, angleEnd])
                if g_chunk is None:
                    print('failed to load projections!')
                    return False
                
                if self.data_type == self.ATTENUATION:
                    if has_torch == True and type(self.g) is torch.Tensor:
                        g_stack_cur = torch.max(g_chunk, axis=0)
                    else:
                        g_stack_cur = np.max(g_chunk, axis=0)
                else:
                    if has_torch == True and type(self.g) is torch.Tensor:
                        g_stack_cur = torch.min(g_chunk, axis=0)
                    else:
                        g_stack_cur = np.min(g_chunk, axis=0)
                
                if g_stack is None:
                    g_stack = g_stack_cur
                else:
                    if self.data_type == self.ATTENUATION:
                        g_stack = self.leapct.maximum(g_stack, g_stack_cur)
                    else:
                        g_stack = self.leapct.minimum(g_stack, g_stack_cur)
                
                del g_chunk
            
            return g_stack
    
    def gain_correction(self, calibration_scans=None, ROI=None, badPixelFile=None):
        if self.leapct.ct_geometry_defined() == False:
            print('Error: CT geometry not defined!')
            return False
            
        if ROI is not None:
            if ROI[0] < 0 or ROI[2] < 0 or ROI[1] < ROI[0] or ROI[3] < ROI[2] or ROI[1] >= self.leapct.get_numRows() or ROI[3] >= self.leapct.get_numCols():
                print('Error: invalid ROI')
                return False
            
        air_scan = None
        dark_scan = None
        
        # FIXME: load bad pixel map from file
        badPixelMap = self.read_image_file(badPixelFile)
        
        #Read in air and dark scan images if necessary
        if self.data_type <= self.UNSPECIFIED or self.data_type > self.ATTENUATION:
            print('Error: must specify data_type')
            return False
        
        if self.data_type == self.RAW:
            dark_scan = self.read_image_file(self.dark_scan_file)
            if dark_scan is None:
                print('Error: failed to load dark scan file')
                return False
        else:
            print('Nothing to do; this function is only for processing raw data')
            return True
            
        if self.data_type == self.RAW or self.data_type == self.RAW_DARK_SUBTRACTED:
            air_scan = self.read_image_file(self.air_scan_file)
            if air_scan is None:
                print('Error: failed to load air scan file')
                return False
            
        if self.g is None:
            self.g = self.load_projections()
            if self.g is None:
                print('Error: failed to load data')
                return False
        
        if self.data_type == self.ATTENUATION:
            self.g = self.leapct.expNeg(self.g)
        
        func = lambda g: leap_preprocessing_algorithms.gain_correction(self.leapct, g, air_scan, dark_scan, calibration_scans, ROI, badPixelMap)
        if func(self.g) == True:
            self.data_type = self.RAW_DARK_SUBTRACTED

            # need to save air scan file
            baseFileName, fileExtension = os.path.splitext(os.path.basename(self.air_scan_file))
            self.air_scan_file = baseFileName + '_gain' + fileExtension
            self.save_image_file(self.air_scan_file, air_scan, use_outputDir=False)
            
            return True
        else:
            return False
    
    def makeAttenuationRadiographs(self, ROI=None, tryIndex=None):
        if self.leapct.ct_geometry_defined() == False:
            print('Error: CT geometry not defined!')
            return False
            
        if ROI is not None:
            if ROI[0] < 0 or ROI[2] < 0 or ROI[1] < ROI[0] or ROI[3] < ROI[2] or ROI[1] >= self.leapct.get_numRows() or ROI[3] >= self.leapct.get_numCols():
                print('Error: invalid ROI')
                return False
            
        air_scan = None
        dark_scan = None
            
        #Read in air and dark scan images if necessary
        if self.data_type <= self.UNSPECIFIED or self.data_type > self.ATTENUATION:
            print('Error: must specify data_type')
            return False
        
        if self.data_type == self.RAW:
            dark_scan = self.read_image_file(self.dark_scan_file)
            if dark_scan is None:
                print('Error: failed to load dark scan file')
                return False
            
        if self.data_type == self.RAW or self.data_type == self.RAW_DARK_SUBTRACTED:
            air_scan = self.read_image_file(self.air_scan_file)
            if air_scan is None:
                print('Error: failed to load air scan file')
                return False
            
        algorithm = lambda g: leap_preprocessing_algorithms.makeAttenuationRadiographs(self.leapct, g, air_scan, dark_scan, ROI)
        self.numOverlap = 0
        self.num_proj = 1

        self.outName = 'attenRad.tif'        
        retVal = self.projection_processing(algorithm, tryIndex)
        self.outName = None
        if retVal and tryIndex is None:
            self.data_type = self.ATTENUATION
        return retVal
        
        """
        if tryIndex is None:
        
            if self.g is None:
                self.g = self.load_projections()
                if self.g is None:
                    print('Error: failed to load data')
                    return False
        
            if self.data_type == self.ATTENUATION:
                self.leapct.expNeg(self.g)
            
            if algorithm(self.g) == True:
                self.data_type = self.ATTENUATION
                return True
            else:
                return False
        else:
            aProj = self.grab_single_projection(tryIndex)
            if aProj is None:
                return False
                
            if self.data_type == self.ATTENUATION:
                self.leapct.expNeg(aProj)
            
            if algorithm(aProj) == True:
                self.lastImage = np.squeeze(aProj)
                return True
            else:
                self.lastImage = None
                return False
        #"""
                
    def crop_projections(self, rowRange=None, colRange=None):
    
        if self.data_type != self.ATTENUATION:
            print('Error: this algorithm currently only implemented for attenuation data')
            return False
        if rowRange is not None:
            if rowRange[0] < 0 or rowRange[0] >= self.leapct.get_numRows() or rowRange[1] < 0 or rowRange[1] >= self.leapct.get_numRows() or rowRange[0] > rowRange[1]:
                print('Error: invalid cropping region')
                return False
            numRows = rowRange[1] - rowRange[0] + 1
        else:
            numRows = self.leapct.get_numRows()
        if colRange is not None:
            if colRange[0] < 0 or colRange[0] >= self.leapct.get_numCols() or colRange[1] < 0 or colRange[1] >= self.leapct.get_numCols() or colRange[0] > colRange[1]:
                print('Error: invalid cropping region')
                return False
            numCols = colRange[1] - colRange[0] + 1
        else:
            numCols = self.leapct.get_numCols()
        self.chunking_type = self.PROJECTION
        self.numOverlap = 0
        self.num_vol = 0
        self.num_proj = 1 + numRows*numCols / (self.get_numRows()*self.get_numCols())
        algorithm = lambda g: self.leapct.crop_projections(rowRange, colRange, g)
            
        return self.projection_processing(algorithm, tryIndex)
    
        """
        if self.leapct.ct_geometry_defined() == False:
            print('Error: CT geometry not defined!')
            return False
        if self.data_type == self.ATTENUATION:
            if self.g is None:
                self.g = self.load_projections()
                if self.g is None:
                    print('Error: failed to load data')
                    return False
            if rowRange is not None or colRange is not None:
                self.g = self.leapct.crop_projections(rowRange, colRange, self.g)
            return True
        else:
            print('Error: crop projections current only implemented for attenuation data')
            return False
        #"""
        
    def badPixelCorrection(self, badPixelFile=None, windowSize=5):
        if self.leapct.ct_geometry_defined() == False:
            print('Error: CT geometry not defined!')
            return False
            
        # FIXME: load bad pixel map from file
        badPixelMap = self.read_image_file(badPixelFile)
        
        air_scan = None
        dark_scan = None
            
        #Read in air and dark scan images if necessary
        if self.data_type <= self.UNSPECIFIED or self.data_type > self.ATTENUATION:
            print('Error: must specify data_type')
            return False
        
        if self.data_type == self.RAW:
            dark_scan = self.read_image_file(self.dark_scan_file)
            if dark_scan is None:
                print('Error: failed to load dark scan file')
                return False
            
        if self.data_type == self.RAW or self.data_type == self.RAW_DARK_SUBTRACTED:
            air_scan = self.read_image_file(self.air_scan_file)
            if air_scan is None:
                print('Error: failed to load air scan file')
                return False
        
        if self.g is None:
            self.g = self.load_projections()
            if self.g is None:
                print('Error: failed to load data')
                return False
        
        if leap_preprocessing_algorithms.badPixelCorrection(self.leapct, self.g, air_scan, dark_scan, badPixelMap, windowSize, self.data_type == self.ATTENUATION) == True:
            if air_scan is not None:
                # need to save air scan file
                baseFileName, fileExtension = os.path.splitext(os.path.basename(self.air_scan_file))
                self.air_scan_file = baseFileName + '_badpix' + fileExtension
                self.save_image_file(self.air_scan_file, air_scan, use_outputDir=False)
                #plt.imshow(air_scan)
                #plt.show()
            if dark_scan is not None:
                # need to save dark scan file
                baseFileName, fileExtension = os.path.splitext(os.path.basename(self.dark_scan_file))
                self.dark_scan_file = baseFileName + '_badpix' + fileExtension
                self.save_image_file(self.dark_scan_file, dark_scan, use_outputDir=False)
                #plt.imshow(dark_scan)
                #plt.show()

            return True
        else:
            return False
        
    def outlierCorrection(self, threshold=0.03, windowSize=3, tryIndex=None):
        if self.data_type != self.ATTENUATION:
            print('Error: this algorithm currently only implemented for attenuation data')
            return False
        self.chunking_type = self.PROJECTION
        self.numOverlap = 0
        self.num_vol = 0
        self.num_proj = 1
        algorithm = lambda g: leap_preprocessing_algorithms.outlierCorrection(self.leapct, g, threshold, windowSize, isAttenuationData=True)
            
        return self.projection_processing(algorithm, tryIndex)
        
        
    def outlierCorrection_highEnergy(self, tryIndex=None):
        if self.data_type != self.ATTENUATION:
            print('Error: this algorithm currently only implemented for attenuation data')
            return False
        self.chunking_type = self.PROJECTION
        self.numOverlap = 0
        self.num_vol = 0
        self.num_proj = 1
        algorithm = lambda g: leap_preprocessing_algorithms.outlierCorrection_highEnergy(self.leapct, g, isAttenuationData=True)
        
        return self.projection_processing(algorithm, tryIndex)
        
    def detectorDeblur_FourierDeconv(self, H, WienerParam=0.0):
        #leap_preprocessing_algorithms.detectorDeblur_FourierDeconv(self.leapct, ...)
        pass
        
    def detectorDeblur_RichardsonLucy(self, H, numIter=10):
        #leap_preprocessing_algorithms.detectorDeblur_RichardsonLucy(self.leapct, ...)
        pass
        
    def find_centerCol(self, iRow=-1):
        if self.leapct.ct_geometry_defined() == False:
            print('Error: CT geometry not defined!')
            return False
        if self.data_type != self.ATTENUATION:
            print('Error: find_centerCol current only implemented for attenuation data')
            return False
        
        if self.g is None and self.projection_memory() >= self.max_CPU_memory_usage:
            if iRow < 0 or iRow >= self.leapct.get_numRows():
                if self.leapct.get_geometry() == 'CONE' or self.leapct.get_geometry() == 'CONE-PARALLEL':
                    iRow = int(np.floor(0.5 + self.leapct.get_centerRow()))
                else:
                    iRow = self.leapct.get_numRows()//2
            rowRange = [iRow, iRow]
            g_chunk = self.load_projection_rows(self.projection_file, rowRange)
            if g_chunk is None:
                print('Error: failed to load data!')
                return False
            self.leapct_backup.copy_parameters(self.leapct)
            self.leapct.crop_rows(rowRange)
            self.leapct.find_centerCol(g_chunk)
            centerCol = self.leapct.get_centerCol()
            self.leapct.copy_parameters(self.leapct_backup)
            self.leapct.set_centerCol(centerCol)
            return True
        else:
            if self.g is None:
                self.g = self.load_projections()
                if self.g is None:
                    print('Error: failed to load data')
                    return False
            self.leapct.find_centerCol(self.g, iRow)
            return True
            
    def conjugate_difference(self, alpha=0.0, centerCol=None):
        if self.leapct.ct_geometry_defined() == False:
            print('Error: CT geometry not defined!')
            return False
        if self.projection_memory() >= self.max_CPU_memory_usage:
            print('Error: not enough memory for this operation!')
            return False
        if self.data_type == self.ATTENUATION:
            if self.g is None:
                self.g = self.load_projections()
                if self.g is None:
                    print('Error: failed to load data')
                    return False
            self.lastImage = self.leapct.conjugate_difference(self.g, alpha, centerCol)
            return True
        else:
            print('Error: conjugate_difference current only implemented for attenuation data')
            return False
    
    def estimate_tilt(self):
        if self.leapct.ct_geometry_defined() == False:
            print('Error: CT geometry not defined!')
            return 0.0
        if self.projection_memory() >= self.max_CPU_memory_usage:
            print('Error: not enough memory for this operation!')
            return False
        if self.data_type == self.ATTENUATION:
            if self.g is None:
                self.g = self.load_projections()
                if self.g is None:
                    print('Error: failed to load data')
                    return 0.0
            return self.leapct.estimate_tilt(self.g)
        else:
            print('Error: estimate_tilt current only implemented for attenuation data')
            return 0.0
    
    def ringRemoval(self, delta=0.01, beta=1.0e3, numIter=30, maxChange=0.05, which='fast', tryIndex=None):
        self.chunking_type = self.DETECTOR_ROW
        self.numOverlap = 3
        self.num_vol = 0
        
        #algorithm = lambda f: self.leapct.MedianFilter(f, threshold, windowSize)
        if which == 'fast':
            self.num_proj = 1
            algorithm = lambda g: leap_preprocessing_algorithms.ringRemoval_fast(self.leapct, g, delta, beta, numIter, maxChange, True)
        else:
            self.num_proj = 3
            algorithm = lambda g: leap_preprocessing_algorithms.ringRemoval(self.leapct, g, delta, beta, numIter, maxChange)
            
        return self.sinogram_processing(algorithm, tryIndex)
    
        """
        if self.leapct.ct_geometry_defined() == False:
            print('Error: CT geometry not defined!')
            return False
        if self.data_type != self.ATTENUATION:
            print('Error: ring removal current only implemented for attenuation data')
            return False
        if tryIndex is None:
            if self.g is None:
                self.g = self.load_projections()
                if self.g is None:
                    print('Error: failed to load data')
                    return False
            if which == 'fast':
                return leap_preprocessing_algorithms.ringRemoval_fast(self.leapct, self.g, delta, beta, numIter, maxChange)
            else:
                return leap_preprocessing_algorithms.ringRemoval(self.leapct, self.g, delta, beta, numIter, maxChange)
        else:
            iz = tryIndex
            if iz < 0 or iz >= self.leapct.get_numZ():
                iz = self.leapct.get_numZ()//2
            #g_ROI = self.g.copy()
            #rowRange = [0, g_ROI.shape[1]-1]
            g_ROI, rowRange = self.grab_necessary_sinograms_for_reconstruction(iz, 3)
            print(rowRange)
            if g_ROI is None:
                print('Error: failed to load data')
                return False
            
            #g_copy = g_ROI.copy()
            if which == 'fast':
                leap_preprocessing_algorithms.ringRemoval_fast(self.leapct, g_ROI, delta, beta, numIter, maxChange)
            else:
                leap_preprocessing_algorithms.ringRemoval(self.leapct, g_ROI, delta, beta, numIter, maxChange)
            #self.lastImage = np.squeeze(g_copy[:,g_copy.shape[1]//2,:] - g_ROI[:,g_copy.shape[1]//2,:])
            self.leapct_backup.copy_parameters(self.leapct)
            self.leapct_backup.crop_projections(rowRange)
            f_slice = self.leapct_backup.FBP_slice(g_ROI, iz)
            del g_ROI
            self.lastImage = np.squeeze(f_slice)
            
            return True
            """
        
    def ringRemoval_median(self, threshold=0.0, windowSize=5, numIter=1, tryIndex=None):
        self.chunking_type = self.DETECTOR_ROW
        self.numOverlap = 0
        self.num_vol = 0
        self.num_proj = 2
        algorithm = lambda g: leap_preprocessing_algorithms.ringRemoval_median(self.leapct, g, threshold, windowSize, numIter)
        return self.sinogram_processing(algorithm, tryIndex)
        
    def parameter_sweep(self, values, param='centerCol', iz=None, algorithmName='FBP'):
        if self.leapct.all_defined() == False:
            print('Error: CT geometry and CT volume must be defined before running this algorithm!')
            return False
        if self.data_type != self.ATTENUATION:
            print('Error: parameter_sweep current only implemented for attenuation data')
            return False
        if self.projection_memory() >= self.max_CPU_memory_usage:
            print('Error: not enough memory for this operation!')
            return False
        if self.g is None:
            self.g = self.load_projections()
            if self.g is None:
                print('Error: failed to load data')
                return False
        f_stack = leap_preprocessing_algorithms.parameter_sweep(self.leapct, self.g, values, param, iz, algorithmName)
        if f_stack is None:
            return False
        else:
            if f_stack.shape[0] == 1 or len(f_stack.shape) == 2:
                plt.imshow(np.squeeze(f_stack), cmap='gray', interpolation='nearest')
                plt.show()
            else:
                self.leapct.display(f_stack)
            return True
    
    def polynomialBHC(self, coeffs, tryIndex=None):
        if coeffs is None:
            print('Error: must define the polynomial coefficients for BHC')
        self.chunking_type = self.DETECTOR_ROW
        self.numOverlap = 0
        self.num_vol = 0
        self.num_proj = 1
        algorithm = lambda g: self.apply_polynomial(g, coeffs)
        return self.sinogram_processing(algorithm, tryIndex)
    
    def apply_polynomial(self, g, coeffs):
    
        if coeffs.size == 1:
            if coeffs[0] != 1.0:
                g[:] = coeffs[0]*g[:]
        elif coeffs.size == 2:
            if coeffs[0] != 1.0 or coeffs[1] != 0.0:
                g[:] = coeffs[0]*g[:] + coeffs[1]*g[:]**2
        elif coeffs.size == 3:
            if coeffs[0] != 1.0 or coeffs[1] != 0.0 or coeffs[2] != 0.0:
                g[:] = coeffs[0]*g[:] + coeffs[1]*g[:]**2 + coeffs[2]*g[:]**3
        elif coeffs.size == 4:
            if coeffs[0] != 1.0 or coeffs[1] != 0.0 or coeffs[2] != 0.0 or coeffs[3] != 0.0:
                g[:] = coeffs[0]*g[:] + coeffs[1]*g[:]**2 + coeffs[2]*g[:]**3 + coeffs[3]*g[:]**4
        else:
            if coeffs[0] != 1.0 or coeffs[1] != 0.0 or coeffs[2] != 0.0 or coeffs[3] != 0.0 or coeffs[4] != 0.0:
                g[:] = coeffs[0]*g[:] + coeffs[1]*g[:]**2 + coeffs[2]*g[:]**3 + coeffs[3]*g[:]**4 + coeffs[4]*g[:]**5
    
    def singleMaterialBHC(self, material=None, tryIndex=None):
        if has_physics == False:
            print('Error: BHC requires the XrayPhysics package!')
            return False
        if self.source_spectra_defined() == False:
            print('Error: spectra not defined!')
        if material is None:
            if self.object_model is not None:
                self._normalize_object_model()
                material = self.object_model[0][0]
            else:
                print('Error: must define material for BHC')
                return False
                
        Es, s_total = self.totalSystemSpectralResponse()
        if self.reference_energy is None or self.reference_energy < Es[0] or self.reference_energy > Es[-1]:
            self.reference_energy = self.physics.meanEnergy(s_total, Es)
        BHC_LUT, T_lut = self.physics.setBHClookupTable(s_total, Es, material, self.reference_energy)
        if BHC_LUT is None:
            return False        
        
        self.chunking_type = self.DETECTOR_ROW
        self.numOverlap = 0
        self.num_vol = 0
        self.num_proj = 1
        algorithm = lambda g: self.leapct.applyTransferFunction(g, BHC_LUT, T_lut)
        return self.sinogram_processing(algorithm, tryIndex)

    def _load_attenuation_projections(self):
        """ Ensure the full attenuation projection data is loaded into self.g.

        Returns True when attenuation projection data is available in memory.
        """
        if self.leapct.ct_geometry_defined() == False:
            print('Error: CT geometry not defined!')
            return False
        if self.data_type != self.ATTENUATION:
            print('Error: this algorithm currently only implemented for attenuation data')
            return False
        if self.g is None:
            self.g = self.load_projections()
            if self.g is None:
                print('Error: failed to load data')
                return False
        return True

    def down_sample_projections(self, factor=2.0):
        """ Down-sample the projection data (and CT volume) by the given factor.

        An anti-aliasing filter is applied to the rows and columns of the
        projections and the CT geometry / volume parameters are updated.
        """
        if self._load_attenuation_projections() == False:
            return False
        self.g = self.leapct.down_sample_projections(factor, self.g)
        self.leapct.down_sample_volume(factor)
        return self.g is not None

    def projection_processing_setup(self, tryIndex=None):
        if self.leapct.ct_geometry_defined() == False:
            print('Error: CT geometry must be defined before running this algorithm!')
            return False
            
        self.chunking_type = self.PROJECTION
        self.num_vol = 0
        self.num_proj = max(1, self.num_proj)
        self.numOverlap = max(0, self.numOverlap)
        
        self.chunk_size = self.leapct.get_numAngles()
        
        if tryIndex is None:
            # Need to process the entire set of projections
            if self.num_proj*self.projection_memory() >= self.max_CPU_memory_usage:
                # not enough memory for this operation, so clear any memory currently being used
                self.clear_volume_data()
                if self.g is not None:
                    # save projection data first
                    print('Saving projection data to disk...')
                    self.save_projection_angles(self.g, update_params=True)
                    self.clear_projection_data_data()
                    
                self.set_chunk_size()
                self.create_outputDir() # do I really need to do this?
                return True
            else:
                # there is enough memory to perform operation in one chunk
                if self.memory_used_by_array(self.f) + self.num_proj*self.projection_memory() >= self.max_CPU_memory_usage:
                    # clear volume data memory because it pushes us past the limit
                    self.clear_volume_data()
                if self.g is None:
                    if self.data_type == self.TRANSMISSION or self.data_type == self.ATTENUATION:
                        input_file = self.projection_file
                    else:
                        input_file = self.raw_scan_file
                    self.g = self.load_projection_angles(input_file)
                if self.g is None:
                    print('Error: failed to load data')
                    return False
        else:
            return True
    
    def projection_processing(self, algorithm, tryIndex=None):
        if self.projection_processing_setup(tryIndex) == False:
            return False
        
        if self.data_type == self.TRANSMISSION or self.data_type == self.ATTENUATION:
            input_file = self.projection_file
        else:
            input_file = self.raw_scan_file
        
        if tryIndex is None:
            # Need to process the entire set of projections
            numAngles = self.leapct.get_numAngles()
            numChunks = int(np.ceil(float(numAngles)/float(self.chunk_size)))
            if self.chunk_size < numAngles:
                
                print('Performing algorithm in ' + str(numChunks) + ' chunks of ' + str(self.chunk_size) + ' slices...')
                
                for n in range(numChunks):
                    print('processing chunk ' + str(n+1) + ' of ' + str(numChunks))
                    
                    angleStart = n*self.chunk_size
                    angleEnd = min(numAngles-1, angleStart + self.chunk_size - 1)
                    
                    #print('reading ' + str(input_file) + '...')
                    g_chunk = self.load_projection_angles(input_file, [angleStart, angleEnd])
                    if g_chunk is None:
                        print('failed to load projections!')
                        return False
                    
                    self.leapct_backup.copy_parameters(self.leapct)
                    if algorithm(g_chunk) == False:
                        self.leapct.copy_parameters(self.leapct_backup)
                        return False
                    
                    if n == numChunks-1:
                        update_params = True
                    else:
                        update_params = False
                    
                    self.save_projection_angles(g_chunk, angleStart, update_params=update_params)
                    if update_params:
                        self.save_parameters()
                    else:
                        self.leapct.copy_parameters(self.leapct_backup)
                    del g_chunk
                
                return True
            else:
                if self.g is None:
                    # data should have been loaded by projection_processing_setup
                    print('Error: failed to load data')
                    return False
                return algorithm(self.g)
        else:
            iAngle = tryIndex
            if iAngle < 0 or iAngle >= self.leapct.get_numAngles():
                iAngle = 0
                
            aProj = self.grab_single_projection(iAngle)
            if aProj is None:
                print('Error: failed to load data')
                return False
            self.leapct_backup.copy_parameters(self.leapct)
            data_type_save = self.data_type
            algorithm(aProj)
            self.data_type = data_type_save
            self.leapct.copy_parameters(self.leapct_backup)
            self.lastImage = np.squeeze(aProj)
            
            return True
        
    def sinogram_processing(self, algorithm, tryIndex=None):
        if self.leapct.ct_geometry_defined() == False:
            print('Error: CT geometry must be defined before running this algorithm!')
            return False
        if self.data_type != self.ATTENUATION:
            print('Error: data_type must be ATTENUATION for this algorithm')
            return False
        
        self.chunking_type = self.DETECTOR_ROW
        self.num_vol = 0
        self.num_proj = max(1, self.num_proj)
        self.numOverlap = max(0, self.numOverlap)
        
        self.chunk_size = self.leapct.get_numRows()
        
        if tryIndex is None:
            # Need to process all detector rows
            if self.num_proj*self.projection_memory() >= self.max_CPU_memory_usage:
                # not enough memory for this operation, so clear any memory currently being used
                self.clear_volume_data()
                if self.g is not None:
                    # save projection data first
                    print('Saving projection data to disk...')
                    self.save_projection_rows(self.g, update_params=True)
                    self.clear_projection_data_data()
                    
                ############################################################################################
                self.set_chunk_size()
                self.create_outputDir() # do I really need to do this?
                
                numRows = self.leapct.get_numRows()
                numChunks = int(np.ceil(float(numRows)/float(self.chunk_size)))
                
                print('Performing algorithm in ' + str(numChunks) + ' chunks of ' + str(self.chunk_size) + ' slices...')
                
                if self.numOverlap > 0:
                    g_lastRows = np.zeros((self.leapct.get_numAngles(), self.numOverlap, self.leapct.get_numCols()), dtype=np.float32)
                else:
                    g_lastRows = None
                    
                last_row = None
                for n in range(numChunks):
                    print('processing chunk ' + str(n+1) + ' of ' + str(numChunks))
                    
                    rowStart = n*self.chunk_size
                    rowEnd = min(numRows-1, rowStart + self.chunk_size - 1)
                    
                    rowStart_pad = max(0, rowStart - self.numOverlap)
                    rowEnd_pad = min(numRows-1, rowEnd + self.numOverlap)
                    
                    padded_left_rows = []
                    padded_right_rows = []
                    if rowStart_pad < rowStart:
                        padded_left_rows = list(range(rowStart_pad-rowStart_pad, rowStart-rowStart_pad))
                    if rowEnd_pad > rowEnd:
                        padded_right_rows = list(range(rowEnd+1-rowStart_pad, rowEnd_pad+1-rowStart_pad))
                    padded_rows = padded_left_rows + padded_right_rows
                    
                    print('reading ' + str(self.projection_file) + '...')
                    g_chunk = self.load_projection_rows(self.projection_file, [rowStart_pad, rowEnd_pad])
                    if g_chunk is None:
                        print('failed to load rows!')
                        
                    if self.numOverlap >= 1:
                        if n > 0:
                            g_chunk[:,0:self.numOverlap,:] = g_lastRows[:]
                        if n < numChunks-1:
                            g_lastRows[:] = g_chunk[:,g_chunk.shape[1]-self.numOverlap:g_chunk.shape[1],:]
                        
                    algorithm(g_chunk)
                    
                    #if n == 1:
                    #    self.leapct.display(g_chunk)
                    
                    # Perform single-slice feathering between slabs
                    if self.numOverlap >= 1:
                        if last_row is not None:
                            g_chunk[:,self.numOverlap,:] = 0.5*(last_row[:,:] + g_chunk[:,self.numOverlap,:])
                        
                        last_row = np.zeros((g_chunk.shape[0], g_chunk.shape[2]), dtype=np.float32)
                        last_row[:,:] = g_chunk[:,g_chunk.shape[1]-self.numOverlap,:]
                            
                    if len(padded_rows) > 0:
                        g_chunk = np.delete(g_chunk, padded_rows, axis=1)
                    
                    if n == numChunks-1:
                        update_params = True
                    else:
                        update_params = False
                    
                    self.save_projection_rows(g_chunk, rowStart, update_params=update_params)
                    if update_params:
                        self.save_parameters()
                    del g_chunk
                
                return True
            else:
                # there is enough memory to perform operation in one chunk
                if self.memory_used_by_array(self.f) + self.num_proj*self.projection_memory() >= self.max_CPU_memory_usage:
                    # clear volume data memory because it pushes us past the limit
                    self.clear_volume_data()
                if self.g is None:
                    self.g = self.load_projection_angles(self.projection_file)
                if self.g is None:
                    print('Error: failed to load data')
                    return False
                return algorithm(self.g)
        else:
            if self.leapct.ct_volume_defined() == False:
                print('Error: CT volume parameters must be defined!')
                return False
            iz = tryIndex
            if iz < 0 or iz >= self.leapct.get_numZ():
                iz = self.leapct.get_numZ()//2
            #g_ROI = self.g.copy()
            #rowRange = [0, g_ROI.shape[1]-1]
            g_ROI, rowRange = self.grab_necessary_sinograms_for_reconstruction(iz, self.numOverlap)
            #print(rowRange)
            if g_ROI is None:
                print('Error: failed to load data')
                return False
            
            #g_copy = g_ROI.copy()
            algorithm(g_ROI)
            #self.lastImage = np.squeeze(g_copy[:,g_copy.shape[1]//2,:] - g_ROI[:,g_copy.shape[1]//2,:])
            self.leapct_backup.copy_parameters(self.leapct)
            self.leapct_backup.crop_projections(rowRange)
            f_slice = self.leapct_backup.FBP_slice(g_ROI, iz)
            del g_ROI
            self.lastImage = np.squeeze(f_slice)
            
            return True
        
    def reconstruction_slab_processing(self):
        pass
        
    def zslice_processing(self, algorithm, tryIndex=None):
        if self.leapct.ct_volume_defined() == False:
            print('Error: CT volume must be defined before running this algorithm!')
            return False
            
        self.chunking_type = self.Z_SLICE
        self.num_proj = 0
        self.num_vol = max(1, self.num_vol)
        self.numOverlap = max(0, self.numOverlap)
        
        self.chunk_size = self.leapct.get_numZ()
        
        if tryIndex is None:
            # Need to process the whole volume
            if self.num_vol*self.volume_memory() >= self.max_CPU_memory_usage:
                # not enough memory for this operation, so clear any memory currently being used
                self.clear_projection_data()
                if self.f is not None:
                    # save volume data first
                    print('Saving volume to disk...')
                    self.save_volume(self.f, update_params=True)
                    self.clear_volume_data()
                    
                ############################################################################################
                self.set_chunk_size()
                self.create_outputDir() # do I really need to do this?
                
                numZ = self.leapct.get_numZ()
                numChunks = int(np.ceil(float(numZ)/float(self.chunk_size)))
                
                print('Performing algorithm in ' + str(numChunks) + ' chunks of ' + str(self.chunk_size) + ' slices...')
                
                if self.numOverlap > 0:
                    f_lastSlices = np.zeros((self.numOverlap, self.leapct.get_numY(), self.leapct.get_numX()), dtype=np.float32)
                else:
                    f_lastSlices = None
                    
                last_slice = None
                for n in range(numChunks):
                    print('processing chunk ' + str(n+1) + ' of ' + str(numChunks))
                    
                    sliceStart = n*self.chunk_size
                    sliceEnd = min(numZ-1, sliceStart + self.chunk_size - 1)
                    
                    sliceStart_pad = max(0, sliceStart - self.numOverlap)
                    sliceEnd_pad = min(numZ-1, sliceEnd + self.numOverlap)
                    
                    padded_left_slices = []
                    padded_right_slices = []
                    if sliceStart_pad < sliceStart:
                        padded_left_slices = list(range(sliceStart_pad-sliceStart_pad, sliceStart-sliceStart_pad))
                    if sliceEnd_pad > sliceEnd:
                        padded_right_slices = list(range(sliceEnd+1-sliceStart_pad, sliceEnd_pad+1-sliceStart_pad))
                    padded_slices = padded_left_slices + padded_right_slices
                    
                    print('reading ' + str(self.reconstruction_file) + '...')
                    f_chunk = self.load_volume(self.reconstruction_file, [sliceStart_pad, sliceEnd_pad])
                    if f_chunk is None:
                        print('failed to load slices!')
                        
                    if self.numOverlap >= 1:
                        if n > 0:
                            f_chunk[0:self.numOverlap,:,:] = f_lastSlices[:]
                        if n < numChunks-1:
                            f_lastSlices[:] = f_chunk[f_chunk.shape[0]-self.numOverlap:f_chunk.shape[0],:,:]
                        
                    algorithm(f_chunk)
                    
                    # Perform single-slice feathering between slabs
                    if self.numOverlap >= 1:
                        if last_slice is not None:
                            f_chunk[self.numOverlap,:,:] = 0.5*(last_slice[:,:] + f_chunk[self.numOverlap,:,:])
                        
                        last_slice = np.zeros((f_chunk.shape[1], f_chunk.shape[2]), dtype=np.float32)
                        last_slice[:,:] = f_chunk[f_chunk.shape[0]-self.numOverlap,:,:]
                            
                    if len(padded_slices) > 0:
                        f_chunk = np.delete(f_chunk, padded_slices, axis=0)
                    
                    if n == numChunks-1:
                        update_params = True
                    else:
                        update_params = False
                    
                    self.save_volume(f_chunk, sliceStart, update_params=update_params)
                    if update_params:
                        self.save_parameters()
                    del f_chunk
                
                return True
            else:
                # there is enough memory to perform operation in one chunk
                if self.memory_used_by_array(self.g) + self.num_vol*self.volume_memory() >= self.max_CPU_memory_usage:
                    # clear projection data memory because it pushes us past the limit
                    self.clear_projection_data()
                if self.f is None:
                    self.f = self.load_volume(self.reconstruction_file)
                if self.f is None:
                    print('Error: failed to load data')
                    return False
                return algorithm(self.f)
        else:
            # just trying this algorithm for a single slice
            iz = tryIndex
            numZ = self.leapct.get_numZ()
            if iz < 0 or iz >= numZ:
                iz = numZ//2
            sliceRange = [max(0, min(iz-self.numOverlap, numZ-1)), max(0, min(iz+self.numOverlap, numZ-1))]
            f_ROI = self.grab_slices(sliceRange) # will grab from self.f if it exists, otherwise will read from file
            if f_ROI is None:
                print('Error: failed to load data')
                return False
            else:
                algorithm(f_ROI)
                self.lastImage = np.squeeze(f_ROI[f_ROI.shape[0]//2,:,:])
                del f_ROI
                return True
        
    
    ###################################################################################################################
    ###################################################################################################################
    # RECONSTRUCTION ALGORITHMS
    ###################################################################################################################
    ###################################################################################################################
    def project(self):
        if self.leapct.all_defined() == False:
            print('Error: CT geometry and CT volume must be defined before running this algorithm!')
            return False
        if self.data_type != self.ATTENUATION:
            print('Error: data_type must be ATTENUATION for projection')
            return False
        if self.f is None:
            self.f = self.load_volume(self.reconstruction_file)
            if self.f is None:
                print('Error: failed to load volume data')
                return False
        if self.leapct.project(self.g, self.f) is not None:
            return True
        else:
            return False
        
    def backproject(self):
        if self.leapct.all_defined() == False:
            print('Error: CT geometry and CT volume must be defined before running this algorithm!')
            return False
        if self.data_type != self.ATTENUATION:
            print('Error: data_type must be ATTENUATION for reconstruction')
            return False
        if self.g is None:
            self.g = self.load_projections()
            if self.g is None:
                print('Error: failed to load data')
                return False
        if self.leapct.backproject(self.g, self.f) is not None:
            return True
        else:
            return False
        
    def FBP(self):
        if self.leapct.all_defined() == False:
            print('Error: CT geometry and CT volume must be defined before running this algorithm!')
            return False
        if self.data_type != self.ATTENUATION:
            print('Error: data_type must be ATTENUATION for reconstruction')
            return False
        
        if self.projection_memory() + self.volume_memory() < self.max_CPU_memory_usage:
            if self.g is None:
                self.g = self.load_projections()
                if self.g is None:
                    print('Error: failed to load data')
                    return False

            if self.f is not None:
                del self.f
            self.f = self.leapct.allocate_volume()
            if self.leapct.FBP(self.g, self.f) is not None:
                minValue, maxValue = self.extrema(self.f)
                print('range of values: ' + str(minValue) + ', ' + str(maxValue))
                if self.leapct.wmax is None:
                    self.leapct.wmax = maxValue
                return True
            else:
                return False
        else:
            # some bit of chunk needs to be performed
            if self.projection_memory() >= self.max_CPU_memory_usage:
                # not even enough memory to hold projections, so save data to disk just in case it is there
                if self.g is not None:
                    # save volume data first
                    print('Saving projection data to disk...')
                    self.save_projection_angles(self.g, update_params=True)
                    self.clear_projection_data_data()
        
            # chunking!
            output_file = os.path.join(self.outputDir, 'zslice.tif')
            output_full_path = os.path.join(self.path, output_file)
            self.create_outputDir()
            self.chunking_type = self.Z_SLICE
            self.num_vol = 1
            self.num_proj = 1
            self.set_chunk_size()
            if self.chunk_size < 1:
                print('Error: insufficient memory!')
                return False
                
            numChunks = int(np.ceil(float(self.leapct.get_numZ())/float(self.chunk_size)))
            z = self.leapct.z_samples()
            
            print('Performing FBP in ' + str(numChunks) + ' chunks of ' + str(self.chunk_size) + ' slices...')
            
            minValue = None
            maxValue = None
            for n in range(numChunks):
                print('processing chunk ' + str(n+1) + ' of ' + str(numChunks))
                self.leapct_backup.copy_parameters(self.leapct)
                
                sliceStart = n*self.chunk_size
                sliceEnd = min(z.size-1, sliceStart + self.chunk_size - 1)
                numZ = sliceEnd - sliceStart + 1
                
                self.leapct_backup.set_numZ(numZ)
                self.leapct_backup.set_offsetZ(self.leapct_backup.get_offsetZ() + z[sliceStart]-self.leapct_backup.get_z0())
                rowRange = self.leapct_backup.rowRangeNeededForBackprojection()

                if self.g is not None:
                    g_chunk = self.leapct_backup.cropProjections(rowRange, None, self.g)
                else:
                    g_chunk = self.load_projection_rows(self.projection_file, rowRange)
                    if g_chunk is None:
                        print('Error: failed to load projection data!')
                        return False
                    self.leapct_backup.cropProjections(rowRange, None)
                
                f_chunk = self.leapct_backup.FBP(g_chunk)
                del g_chunk
                
                minValue_cur, maxValue_cur = self.extrema(f_chunk)
                if minValue is None:
                    minValue = minValue_cur
                    maxValue = maxValue_cur
                else:
                    minValue = min(minValue, minValue_cur)
                    maxValue = max(maxValue, maxValue_cur)
                
                self.leapct_backup.save_volume(output_full_path, f_chunk, sliceStart)
                del f_chunk
            self.reconstruction_file = output_file
            print('range of values: ' + str(minValue) + ', ' + str(maxValue))
            if self.leapct.wmax is None:
                self.leapct.wmax = maxValue
            return True
            
    def FBP_slice(self, islice=None, coord='z'):
        if self.leapct.all_defined() == False:
            print('Error: CT geometry and CT volume must be defined before running this algorithm!')
            return None

        # FBP requires attenuation data.  For convenience (e.g. the GUI preview
        # button) this also supports reconstructing directly from transmission
        # data or dark-subtracted raw data: the data is converted to attenuation
        # just for the reconstruction and converted back afterwards so the stored
        # projection data is left unchanged.  Dark-subtracted raw data requires a
        # numeric air scan value so the flat-field normalization can be applied.
        # gray_value is the value passed to negLog/expNeg (their flat-field
        # divisor); None means the data is already attenuation and no transform
        # is needed.
        if self.data_type == self.ATTENUATION:
            gray_value = None
        elif self.data_type == self.TRANSMISSION:
            # Transmission data is already flat-field normalized.
            gray_value = 1.0
        elif self.data_type == self.RAW_DARK_SUBTRACTED and self.is_number(self.air_scan_file):
            gray_value = float(self.air_scan_file)
        else:
            print('Error: data_type must be ATTENUATION, TRANSMISSION, or dark-subtracted raw (with a numeric air scan) for reconstruction')
            return None

        if self.g is not None:
            # The full projection data set is already in memory: convert it to
            # attenuation in place, reconstruct, then convert it back so the
            # stored projection data is left exactly as we found it.
            if gray_value is not None:
                self.leapct.negLog(self.g, gray_value)
            f_slice = self.leapct.FBP_slice(self.g, islice, coord)
            if gray_value is not None:
                self.leapct.expNeg(self.g, gray_value)
            return f_slice

        # The projection data is not in memory, so only load the data actually
        # needed to reconstruct the requested slice rather than the whole data
        # set.  For a z-slice only the detector rows that backproject into that
        # slice are required; x- and y-slices need every detector row.
        if coord == 'z':
            iz = islice
            if iz is None or iz < 0 or iz >= self.leapct.get_numZ():
                iz = self.leapct.get_numZ() // 2
            g_ROI, rowRange = self.grab_necessary_sinograms_for_reconstruction(iz)
            if g_ROI is None:
                print('Error: failed to load data')
                return None
            if gray_value is not None:
                self.leapct.negLog(g_ROI, gray_value)
            # Reconstruct with a backup geometry cropped to the loaded rows so the
            # primary geometry (self.leapct) is left untouched.
            self.leapct_backup.copy_parameters(self.leapct)
            self.leapct_backup.crop_projections(rowRange)
            f_slice = self.leapct_backup.FBP_slice(g_ROI, iz)
            del g_ROI
            return f_slice
        else:
            # x- and y-slices need all detector rows, so the full data set must
            # be loaded.  Load it and keep it in memory; convert to attenuation
            # in place, reconstruct, then convert it back so the stored
            # projection data is left in its original form.
            self.g = self.load_projections()
            if self.g is None:
                print('Error: failed to load data')
                return None
            if gray_value is not None:
                self.leapct.negLog(self.g, gray_value)
            f_slice = self.leapct.FBP_slice(self.g, islice, coord)
            if gray_value is not None:
                self.leapct.expNeg(self.g, gray_value)
            return f_slice

    def inconsistencyReconstruction(self):
        if self.leapct.all_defined() == False:
            print('Error: CT geometry and CT volume must be defined before running this algorithm!')
            return False
        if self.data_type != self.ATTENUATION:
            print('Error: data_type must be ATTENUATION for reconstruction')
            return False
        if self.g is None:
            self.g = self.load_projections()
            if self.g is None:
                print('Error: failed to load data')
                return False
        if self.leapct.inconsistencyReconstruction(self.g, self.f) is not None:
            return True
        else:
            return False
        
    def tight_volume(self, threshold, L=8, tryIndex=None):
        if self.leapct.all_defined() == False:
            print('Error: CT geometry and CT volume must be defined before running this algorithm!')
            return False
        if self.data_type != self.ATTENUATION:
            print('Error: data_type must be ATTENUATION for reconstruction')
            return False
        
        if self.projection_memory() >= self.max_CPU_memory_usage:
            print('Error: insufficient memory!')
            return False
        
        if self.g is None:
            self.g = self.load_projections()
            if self.g is None:
                print('Error: failed to load data')
                return False
    
        self.leapct_backup.copy_parameters(self.leapct)
        
        self.leapct.set_default_volume()
        self.leapct.set_diameterFOV(self.leapct.get_numX()*self.leapct.get_voxelWidth())
        self.leapct.set_default_volume(float(L))
        self.leapct.set_projector('SF')
        self.leapct.set_rampFilter(0)
        f = self.leapct.FBP(self.g)
        
        if threshold > self.extrema(f)[1]:
            self.leapct.copy_parameters(self.leapct_backup)
            print('Error: threshold exceeds maximum value of reconstruction!')
            return False
        
        M_yx = np.amax(np.amax(f,axis=2),axis=1)
        M_zx = np.amax(np.amax(f,axis=2),axis=0)
        M_zy = np.amax(np.amax(f,axis=1),axis=0)
        
        ind_x = np.squeeze(np.argwhere(M_zy > threshold))
        ind_y = np.squeeze(np.argwhere(M_zx > threshold))
        ind_z = np.squeeze(np.argwhere(M_yx > threshold))
        x = self.leapct.x_samples()
        y = self.leapct.y_samples()
        z = self.leapct.z_samples()
        AABB = [x[ind_x[0]], x[ind_x[-1]], y[ind_y[0]], y[ind_y[-1]], z[ind_z[0]], z[ind_z[-1]]]

        self.leapct.set_default_volume()
        T_x = self.leapct.get_voxelWidth()
        T_z = self.leapct.get_voxelHeight()
        numX_full = self.leapct.get_numX()
        numY_full = self.leapct.get_numY()
        numZ_full = self.leapct.get_numZ()
        offsetZ_full = self.leapct.get_offsetZ()
        numX = int((AABB[1]-AABB[0])/T_x)+3*int(L)
        numY = int((AABB[3]-AABB[2])/T_x)+3*int(L)
        numZ = int((AABB[5]-AABB[4])/T_z)+2*int(L)
        offsetX = 0.5*(AABB[0]+AABB[1])
        offsetY = 0.5*(AABB[2]+AABB[3])
        offsetZ = 0.5*(AABB[4]+AABB[5])
        
        if numX > numX_full:
            numX = numX_full
            offsetX = 0.0
        if numY > numY_full:
            numY = numY_full
            offsetY = 0.0
        if numZ > numZ_full:
            numZ = numZ_full
            offsetZ = offsetZ_full
        
        #self.leapct.print_parameters()
        self.leapct.copy_parameters(self.leapct_backup)
        self.leapct.set_volume(numX, numY, numZ, T_x, T_z, offsetX, offsetY, offsetZ)
        #self.leapct.print_parameters()
        reductionFactor = (float(numX_full)*float(numY_full)*float(numZ_full)) / (float(numX)*float(numY)*float(numZ))
        print('Reduced volume size by a factor of: ' + str(reductionFactor))
        
        if tryIndex is not None:
            iz = tryIndex
            if iz < 0 or iz >= self.leapct.get_numZ():
                iz = self.leapct.get_numZ()//2
            g_ROI, rowRange = self.grab_necessary_sinograms_for_reconstruction(iz)
            if g_ROI is None:
                print('Error: failed to load data')
                return False
            
            self.leapct.crop_projections(rowRange)
            f_slice = self.leapct.FBP_slice(g_ROI, iz)
            del g_ROI
            self.lastImage = np.squeeze(f_slice)
            self.leapct.copy_parameters(self.leapct_backup)
            return True
        else:
            return True
    
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

    
    ###################################################################################################################
    ###################################################################################################################
    # VOLUME DENOISING
    ###################################################################################################################
    ###################################################################################################################
    def MedianFilter(self, threshold=0.0, windowSize=3, tryIndex=None):
        self.chunking_type = self.Z_SLICE
        self.numOverlap = 1
        self.num_proj = 0
        self.num_vol = 2
        
        algorithm = lambda f: self.leapct.MedianFilter(f, threshold, windowSize)
        return self.zslice_processing(algorithm, tryIndex)
        
    def MedianFilter2D(self, threshold=0.0, windowSize=3, tryIndex=None):
        self.chunking_type = self.Z_SLICE
        self.numOverlap = 0
        self.num_proj = 0
        self.num_vol = 1
        
        algorithm = lambda f: self.leapct.MedianFilter2D(f, threshold, windowSize)
        return self.zslice_processing(algorithm, tryIndex)
        
    def TVdenoising(self, delta=0.001, beta=1.0e1, numIter=20, p=1.2, tryIndex=None):
        self.chunking_type = self.Z_SLICE
        self.numOverlap = 3 # 4?
        self.num_proj = 0
        self.num_vol = 3
        
        algorithm = lambda f: self.leapct.TV_denoise(f, delta, beta, numIter, p)
        return self.zslice_processing(algorithm, tryIndex)
        
    def GuidedFilter(self, r=1, epsilon=0.02, numIter=1, tryIndex=None):
        self.chunking_type = self.Z_SLICE
        self.numOverlap = max(1, int(min(r, 10)) * max(1, int(numIter)))
        self.num_proj = 0
        self.num_vol = 3
        
        algorithm = lambda f: self.leapct.GuidedFilter(f, r, epsilon, numIter)
        return self.zslice_processing(algorithm, tryIndex)
        
    def BilateralFilter(self, spatialFWHM=2.0, intensityFWHM=0.1, scale=1.0, tryIndex=None):
        self.chunking_type = self.Z_SLICE
        self.numOverlap = max(1, int(np.ceil(3.0 * max(spatialFWHM, scale))))
        self.num_proj = 0
        self.num_vol = 3
        
        algorithm = lambda f: self.leapct.BilateralFilter(f, spatialFWHM, intensityFWHM, scale)
        return self.zslice_processing(algorithm, tryIndex)
    
    def compress_volume(self, dtype=np.uint16, wmin=0.0, wmax=None):
        if self.reconstruction_file is None or len(self.reconstruction_file) == 0:
            print('Error: reconstruction_file not set!')
            return False
            
        fileList = self.leapct.get_file_list(os.path.join(self.path, self.reconstruction_file))
        if fileList is not None and len(fileList) > 0:
            self.leapct.set_fileIO_parameters(dtype, wmin, wmax)
            for n in range(len(fileList)):
                x = self.leapct.load_tif(fileList[n])
                if x is None:
                    self.leapct.file_dtype = np.float32
                    return False
                else:
                    self.leapct.save_tif(fileList[n], x)
            self.leapct.file_dtype = np.float32
            return True
        else:
            return False
    
    ###################################################################################################################
    ###################################################################################################################
    # GUI UTILITY FUNCTIONS
    ###################################################################################################################
    ###################################################################################################################
    def run(self, text, optional=None):
        return self.cmd(text)
    
    def cmd(self, text, optional=None):
        if "=" in text:
            return self.set_cmd(text)
        elif text.startswith("clear"):
            return self.clear_cmd(text)
        elif text == "trackHistory":
            return False
        else:
            print("Error cmd (" + str(text) + ") failed!")
            return False
            
    def clear_cmd(self, text):
        key = text.split(' ')[1].strip()
        #print("Error: clear command not yet implemented")
        match key:
            case "archdir" | "path":
                self.path = ""
            case "outputdir":
                self.outputDir = ""
            case "dataType":
                #[self.UNSPECIFIED, self.RAW, self.RAW_DARK_SUBTRACTED, self.TRANSMISSION, self.ATTENUATION]
                self.dataType = self.UNSPECIFIED
            case "backgroundFile":
                self.air_scan_file = ""
            case "darkCurrentFile":
                self.dark_scan_file = ""
            case "sfile":
                self.raw_scan_file = ""
            case "pfile":
                self.projection_file = ""
            case "rfile":
                self.reconstruction_file = ""
            case "systemGeometryFile":
                self.geometry_file = ""
            case "lengthUnits":
                pass
            case "bgeometry":
                self.leapct.set_geometry(0)
            case "geometry":
                self.leapct.set_geometry(0)
            case "sod":
                self.leapct.set_sod(0.0)
            case "sdd":
                self.leapct.set_sdd(0.0)
            case "odd":
                pass
            case "helicalpitch":
                self.leapct.set_helicalPitch(0.0)
            case "nangles":
                self.num_angles = 0
                self.leapct.set_numAngles(0)
            case "initangle":
                self.init_angle = 0.0
                phis = self.leapct.get_angles()
                if phis is not None:
                    phis -= phi[0]
                    self.leapct.set_angles(phis)
            case "arange":
                self.angular_range = 0.0
                self.angular_step = 0.0
                self.leapct.set_numAngles(0)
            case "rotationDirection":
                phis = self.leapct.get_angles()
                if phis is not None and phis.size > 1:
                    if phis[1] < phis[0]:
                        phis *= -1.0
                        self.leapct.set_angles(phis)
            case "rotationdirection":
                phis = self.leapct.get_angles()
                if phis is not None and phis.size > 1:
                    if phis[1] < phis[0]:
                        phis *= -1.0
                        self.leapct.set_angles(phis)
            case "nrays":
                self.leapct.set_numCols(0)
            case "nslices":
                self.leapct.set_numRows(0)
            case "pxcenter":
                self.leapct.set_centerCol(0.0)
            case "pzcenter":
                self.leapct.set_centerRow(0.0)
            case "pxmidoff":
                leapct.set_tau(0.0)
            case "pxsize":
                self.leapct.set_pixelWidth(0.0)
            case "pzsize":
                self.leapct.set_pixelHeight(0.0)
            case "detectorShape":
                self.leapct.set_flatDetector()
            case "detectorResponseFile":
                self.detector_response_file = ""
            case "kV":
                self.kV = -1.0
            case "takeOffAngle":
                self.takeoff_angle = 11.0
            case "anodeNormal":
                self.anode_normal = None # np.array([0.0, 1.0, 0.0], dtype=np.float32)
            case "anodeMaterial":
                self.anode_material = 74
            case "filterMaterials" | "xray_filters":
                self.source_filters = None
                self.detector_filters = None
            case "sourceFilters" | "source_filters":
                self.source_filters = None
            case "detectorFilters" | "detector_filters":
                self.detector_filters = None
            case "spectraFile":
                #self.spectra_model_file = ""
                self.source_spectra_file = ""
            case "referenceEnergy":
                self.reference_energy = -1.0
            case "numTVneighbors":
                self.leapct.set_numTVneighbors(26)
            case "projector" | "backprojector":
                self.leapct.set_projector('SF')
            case "rfilter":
                self.leapct.set_rampFilter(2)
            case "rampFWHM":
                self.leapct.set_FBPlowpass(1.0)
            case "rxsize":
                self.leapct.set_voxelWidth(0.0)
            case "rysize":
                self.leapct.set_voxelWidth(0.0)
            case "rzsize":
                self.leapct.set_voxelHeigh(0.0)
            case "rxref":
                self.leapct.set_offsetX(0.0)
            case "ryref":
                self.leapct.set_offsetY(0.0)
            case "rzref":
                self.leapct.set_offsetZ(0.0)
            case "rxoffset":
                pass
            case "ryoffset":
                pass
            case "rzoffset":
                pass
            case "rxelements":
                self.leapct.set_numX(0)
            case "ryelements":
                self.leapct.set_numY(0)
            case "rzelements":
                self.leapct.set_numZ(0)
            case "axisOfSymmetry":
                self.leapct.clear_axisOfSymmetry()
            case "halfscan":
                self.leapct.set_offsetScan(False)
            case _:
                print("Error: cmd keyword " + str(key) + " not yet implemented!")
                return False
        return True
    
    def _parse_material_density_thickness(self, value):
        """ Parse a [material, mass_density, thickness] triple from a stored string.

        Accepts a Python list/tuple literal (e.g. "['GOS', 0.00732, 0.14]") or a
        plain comma-separated string (e.g. "GOS, 0.00732, 0.14"), with or without
        surrounding quotes.  The mass density and thickness are coerced to floats
        so downstream consumers (physics models, the GUI) always receive numbers.
        Returns None when the value cannot be parsed into a valid triple.
        """
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            parts = list(value)
        else:
            text = str(value).strip()
            if len(text) == 0:
                return None
            if text[0] in '[(':
                try:
                    parts = list(eval(text))
                except Exception:
                    return None
            else:
                parts = [p.strip().strip('\'"') for p in text.split(',')]
        if len(parts) < 3:
            return None
        try:
            material = str(parts[0]).strip().strip('\'"')
            mass_density = float(parts[1])
            thickness = float(parts[2])
        except (TypeError, ValueError):
            return None
        return [material, mass_density, thickness]

    def set_cmd(self, text, printError=True):
        key = text.split('=')[0].strip()
        value = text.split('=')[1].strip()
        if len(key) > 0 and len(value) > 0:
            self.set_key_value_pairs(key, value, printError)
        
    def set_key_value_pairs(self, key, value, printError=True):
        match key:
            case "archdir" | "path":
                self.path = value
            case "outputdir":
                self.outputDir = value
            case "dataType" | "data_type":
                #[self.UNSPECIFIED, self.RAW, self.RAW_DARK_SUBTRACTED, self.TRANSMISSION, self.ATTENUATION]
                if value.upper() == "RAW_UNCALIB" or value.upper() == "RAW":
                    self.data_type = self.RAW
                elif value.upper() == "RAW_CALIB" or value.upper() == "RAW_DARKSUB" or value.upper() == "RAW_DARK_SUBTRACTED":
                    self.data_type = self.RAW_DARK_SUBTRACTED
                elif value.upper() == "TRANS_RAD" or value.upper() == "TRANSMISSION":
                    self.data_type = self.TRANSMISSION
                elif value.upper() == "ATTEN_RAD" or value.upper() == "ATTENUATION" or value.upper() == "SINOGRAM":
                    self.data_type = self.ATTENUATION
                elif value.upper() == "RECXY":
                    self.data_type = self.UNSPECIFIED
            case "backgroundFile" | "air_scan_file":
                self.air_scan_file = value
            case "darkCurrentFile" | "dark_scan_file":
                self.dark_scan_file = value
            case "sfile" | "scan_file" | "raw_scan_file":
                self.raw_scan_file = value
            case "Filename Prefix":
                self.raw_scan_file = value + str("*[0-9].tif")
            case "pfile" | "projection_file":
                self.projection_file = value
            case "rfile" | "reconstruction_file":
                self.reconstruction_file = value
            case "systemGeometryFile" | "system_geometry_file" | "geometry_file":
                self.geometry_file = value
            case "lengthUnits":
                pass
            case "max_CPU_memory_usage" | "maxMemoryUsage":
                self.max_CPU_memory_usage = float(value)
            case "GPUs" | "gpus":
                self.leapct.set_gpus(list(eval(value)))
            case "bgeometry" | "geometry":
                self.leapct.set_geometry(value)
            case "geometry":
                self.leapct.set_geometry(value)
            case "sod" | "Object to Source (mm)":
                self.leapct.set_sod(float(value))
            case "sdd" | "Camera to Source (mm)":
                self.leapct.set_sdd(float(value))
            case "odd":
                pass
            case "helicalpitch" | "helicalPitch" | "helical_pitch":
                self.leapct.set_helicalPitch(float(value))
            case "nangles" | "numAngles" | "Number of Files":
                self.num_angles = int(value)
                if key == "Number of Files":
                    self.num_angles = self.num_angles - 1
                self.leapct.set_numAngles(int(value))
                if self.num_angles > 0 and self.angular_range != 0.0:
                    phis = self.init_angle + self.leapct.setAngleArray(self.num_angles, self.angular_range)
                    self.leapct.set_angles(phis)
                elif self.num_angles > 0 and self.angular_step != 0.0:
                    phis = self.init_angle + self.leapct.setAngleArray(self.num_angles, self.angular_step*self.num_angles)
                    self.leapct.set_angles(phis)
            case "initangle" | "init_angle":
                self.init_angle = float(value)
                if self.num_angles > 0 and self.angular_range != 0.0:
                    phis = self.init_angle + self.leapct.setAngleArray(self.num_angles, self.angular_range)
                    self.leapct.set_angles(phis)
                elif self.num_angles > 0 and self.angular_step != 0.0:
                    phis = self.init_angle + self.leapct.setAngleArray(self.num_angles, self.angular_step*self.num_angles)
                    self.leapct.set_angles(phis)
            case "arange" | "angularRange" | "angular_range":
                self.angular_range = float(value)
                if self.num_angles > 0 and self.angular_range != 0.0:
                    phis = self.init_angle + self.leapct.setAngleArray(self.num_angles, self.angular_range)
                    self.leapct.set_angles(phis)
            case "Rotation Step (deg)":
                self.angular_step = float(value)
                if self.num_angles > 0 and self.angular_step != 0.0:
                    phis = self.init_angle + self.leapct.setAngleArray(self.num_angles, self.angular_step*self.num_angles)
                    self.leapct.set_angles(phis)
            case "rotationDirection":
                print("Set rotationDirection not yet implemented!")
            case "nrays" | "numCols" | "Number of Columns":
                self.leapct.set_numCols(int(value))
            case "nslices" | "numRows" | "Number of Rows":
                self.leapct.set_numRows(int(value))
            case "pxcenter" | "centerCol":
                self.leapct.set_centerCol(float(value))
            case "pzcenter" | "centerRow" | "Optical Axis (line)":
                self.leapct.set_centerRow(float(value))
            case "pxmidoff":
                #print("Set pxmidoff not yet implemented!")
                self.leapct.set_tau(float(value))
            case "tau":
                self.leapct.set_tau(float(value))
            case "pxsize" | "pixelWidth":
                self.leapct.set_pixelWidth(float(value))
            case "pzsize" | "pixelHeight":
                self.leapct.set_pixelHeight(float(value))
            case "detectorShape" | "detector_shape" | "detectorType" | "detector_type":
                if value == "FLAT":
                    self.leapct.set_flatDetector()
                else:
                    self.leapct.set_curvedDetector()
            case "detectorResponseFile" | "detector_response_file":
                self.detector_response_file = value
            case "kV" | "Source Voltage (kV)":
                self.kV = float(value)
            case "takeOffAngle" | "takeoff_angle":
                self.takeoff_angle = float(value)
            case "anode_normal" | "anodeNormal":
                self.anode_normal = eval(value)
            case "anodeMaterial" | "anode_material":
                self.anode_material = int(value)
            case "filterMaterials" | "xray_filters":
                # Backward compatibility: the legacy single x-ray filter list maps
                # onto the source filters.
                if value[0] == '[':
                    self.source_filters = eval(value)
                else:
                    self.source_filters = value
            case "sourceFilters" | "source_filters":
                if value[0] == '[':
                    self.source_filters = eval(value)
                else:
                    self.source_filters = value
            case "detectorFilters" | "detector_filters":
                if value[0] == '[':
                    self.detector_filters = eval(value)
                else:
                    self.detector_filters = value
            case "object_model":
                if value[0] == '[':
                    self.object_model = eval(value)
                else:
                    self.object_model = value
                self._normalize_object_model()
            case "detector_response_model":
                self.detector_response_model = self._parse_material_density_thickness(value)
            case "spectraFile" | "source_spectra_file":
                #self.spectra_model_file = value
                self.source_spectra_file = value
            case "referenceEnergy" | "reference_energy":
                self.reference_energy = float(value)
            case "numTVneighbors":
                self.leapct.set_numTVneighbors(int(value))
            case "projector" | "backprojector":
                self.leapct.set_projector(value)
            case "default_algorithms":
                self.default_algorithms = eval(value)
            case "rfilter":
                self.leapct.set_rampFilter(int(value))
            case "rampID" | "rampFilter":
                self.leapct.set_rampFilter(int(value))
            case "rampFWHM" | "FBPlowpass":
                self.leapct.set_FBPlowpass(float(value))
            case "rxsize" | "voxelWidth":
                self.leapct.set_voxelWidth(float(value))
            case "rysize" | "voxelWidth":
                self.leapct.set_voxelWidth(float(value))
            case "rzsize" | "voxelHeight":
                self.leapct.set_voxelHeight(float(value))
            case "rxref":
                #rxref = 0.5*(self.leapct.get_numX()-1) - self.leapct.get_offsetX()/self.leapct.get_voxelWidth()
                rref = float(value)
                offsetX = 0.5*(self.leapct.get_numX()-1)*self.leapct.get_voxelWidth() - rref*self.leapct.get_voxelWidth()
                self.leapct.set_offsetX(offsetX)
            case "ryref":
                rref = float(value)
                offsetY = 0.5*(self.leapct.get_numY()-1)*self.leapct.get_voxelWidth() - rref*self.leapct.get_voxelWidth()
                self.leapct.set_offsetY(offsetY)
            case "rzref":
                rref = float(value)
                offsetZ = 0.5*(self.leapct.get_numZ()-1)*self.leapct.get_voxelHeight() - rref*self.leapct.get_voxelHeight()
                self.leapct.set_offsetZ(offsetZ)
            case "offsetX":
                self.leapct.set_offsetX(float(value))
            case "offsetY":
                self.leapct.set_offsetY(float(value))
            case "offsetZ":
                self.leapct.set_offsetZ(float(value))
            case "rxoffset":
                pass
            case "ryoffset":
                pass
            case "rzoffset":
                pass
            case "rxelements" | "numX":
                self.leapct.set_numX(int(value))
            case "ryelements" | "numY":
                self.leapct.set_numY(int(value))
            case "rzelements" | "numZ":
                self.leapct.set_numZ(int(value))
            case "halfscan" | "offsetScan":
                if value.lower() == "true":
                    self.leapct.set_offsetScan(True)
                elif value.lower() == "false":
                    self.leapct.set_offsetScan(False)
                else:
                    print("Error setting offsetScan")
            case "truncatedScan":
                if value.lower() == "true":
                    self.leapct.set_truncatedScan(True)
                elif value.lower() == "false":
                    self.leapct.set_truncatedScan(False)
                else:
                    print("Error setting truncatedScan")
            case "trackHistory":
                pass
            case "Camera Pixel Size (um)":
                self.leapct.set_pixelWidth(float(value)/1000.0)
                self.leapct.set_pixelHeight(float(value)/1000.0)
            case _:
                if printError:
                    print("Error: cmd keyword " + str(key) + " not yet implemented!")
                return False
        return True
        
    def getParam(self, text):
        match text:
            case "archdir":
                return self.path
            case "outputdir":
                return self.outputDir
            case "dataType":
                #[self.UNSPECIFIED, self.RAW, self.RAW_DARK_SUBTRACTED, self.TRANSMISSION, self.ATTENUATION]
                if self.data_type == 0:
                    return "UNKNOWN"
                elif self.data_type == 1:
                    return "RAW_UNCALIB"
                elif self.data_type == 2:
                    return "RAW_DARKSUB"
                elif self.data_type == 3:
                    return "TRANS_RAD"
                else: #if self.data_type == 4:
                    return "ATTEN_RAD"
            case "datatype":
                if self.data_type == 0:
                    return "UNKNOWN"
                elif self.data_type == 1:
                    return "RAW_UNCALIB"
                elif self.data_type == 2:
                    return "RAW_DARKSUB"
                elif self.data_type == 3:
                    return "TRANS_RAD"
                else: #if self.data_type == 4:
                    return "ATTEN_RAD"
            case "backgroundFile":
                return self.air_scan_file
            case "backgroundfile":
                return self.air_scan_file
            case "darkCurrentFile":
                return self.dark_scan_file
            case "darkcurrentfile":
                return self.dark_scan_file
            case "sfile":
                return self.raw_scan_file
            case "pfile":
                return self.projection_file
            case "rfile":
                return self.reconstruction_file
            case "systemGeometryFile":
                return self.geometry_file
            case "lengthUnits":
                return "mm"
            case "bgeometry":
                return self.leapct.get_geometry()
            case "geometry":
                return self.leapct.get_geometry()
            case "sod":
                return str(self.leapct.get_sod())
            case "sdd":
                return str(self.leapct.get_sdd())
            case "odd":
                return str(self.leapct.get_sdd() - self.leapct.get_sod())
            case "helicalpitch" | "helicalPitch":
                return str(self.leapct.get_helicalPitch())
            case "normalizedHelicalPitch":
                return str(self.leapct.get_normalizedHelicalPitch())
            case "axisOfSymmetry":
                axisOfSymmetry = self.leapct.get_axisOfSymmetry()
                if np.abs(self.leapct.get_axisOfSymmetry()) <= 30.0:
                    return str(axisOfSymmetry)
                else:
                    return ""
            case "nangles":
                return str(self.leapct.get_numAngles())
            case "initangle":
                phis = self.leapct.get_angles()
                if phis is None or len(phis) == 0:
                    return str(0.0)
                else:
                    return str(phis[0])
            case "initAngle":
                phis = self.leapct.get_angles()
                if phis is None or len(phis) == 0:
                    return str(0.0)
                else:
                    return str(phis[0])
            case "arange":
                return str(self.leapct.get_angularRange())
            case "rotationDirection":
                if self.leapct.get_angularRange() >= 0.0:
                    return "COUNTERCLOCKWISE"
                else:
                    return "CLOCKWISE"
            case "rotationdirection":
                if self.leapct.get_angularRange() >= 0.0:
                    return "COUNTERCLOCKWISE"
                else:
                    return "CLOCKWISE"
            case "nrays":
                return str(self.leapct.get_numCols())
            case "nslices":
                return str(self.leapct.get_numRows())
            case "pxcenter":
                return str(self.leapct.get_centerCol())
            case "pzcenter":
                return str(self.leapct.get_centerRow())
            case "pxmidoff":
                #print("Error: getParams pxmidoff not yet implemented!")
                #return str(0.0)
                return str(self.leapct.get_tau())
            case "tau":
                return str(self.leapct.get_tau())
            case "pxsize":
                return str(self.leapct.get_pixelWidth())
            case "pzsize":
                return str(self.leapct.get_pixelHeight())
            case "detectorShape":
                return self.leapct.get_detectorType()
            case "detectorResponseFile":
                return self.detector_response_file
            case "kV":
                return str(self.kV)
            case "takeOffAngle":
                return str(self.takeoff_angle)
            case "anode_normal":
                if self.anode_normal is not None:
                    return str(self.anode_normal)
                else:
                    return ""
            case "anodeMaterial":
                return str(self.anode_material)
            case "filterMaterials" | "sourceFilters" | "source_filters":
                return str(self.source_filters)
            case "detectorFilters" | "detector_filters":
                return str(self.detector_filters)
            case "spectraFile":
                #return self.spectra_model_file
                return self.source_spectra_file
            case "referenceEnergy":
                return str(self.reference_energy)
            case "numTVneighbors":
                return str(self.leapct.get_numTVneighbors())
            case "projector" | "backprojector":
                return self.leapct.get_projector()
            case "rfilter":
                return str(self.leapct.get_rampFilter())
            case "rampID":
                return str(self.leapct.get_rampFilter())
            case "rampFWHM":
                return str(self.leapct.get_FBPlowpass())
            case "rxsize":
                return str(self.leapct.get_voxelWidth())
            case "rysize":
                return str(self.leapct.get_voxelWidth())
            case "rzsize":
                return str(self.leapct.get_voxelHeight())
            case "rxref":
                if self.leapct.get_voxelWidth() <= 0.0:
                    return str(0.0)
                else:
                    return str(0.5*(self.leapct.get_numX()-1) - self.leapct.get_offsetX()/self.leapct.get_voxelWidth())
            case "ryref":
                if self.leapct.get_voxelWidth() <= 0.0:
                    return str(0.0)
                else:
                    return str(0.5*(self.leapct.get_numY()-1) - self.leapct.get_offsetY()/self.leapct.get_voxelWidth())
            case "rzref":
                if self.leapct.get_voxelHeight() <= 0.0:
                    return str(0.0)
                else:
                    return str(0.5*(self.leapct.get_numZ()-1) - self.leapct.get_offsetZ()/self.leapct.get_voxelHeight())
            case "rxoffset":
                return str(0)
            case "ryoffset":
                return str(0)
            case "rzoffset":
                return str(0)
            case "rxelements":
                return str(self.leapct.get_numX())
            case "ryelements":
                return str(self.leapct.get_numY())
            case "rzelements":
                return str(self.leapct.get_numZ())
            case "halfscan":
                return str(self.leapct.get_offsetScan())
            case "ImageJpath":
                return ""
            case "LTTcmd":
                return ""
            case "wmin":
                return ""
            case "wmax":
                return ""
            case "compressFile":
                return "False"
            case "LTTwCmd":
                return ""
            case "trackHistory":
                return "False"
            case "fileType":
                return "tif"
            case "untruncatedProjection":
                return "0"
            case _:
                print("Error: getParam keyword " + str(text) +  " not yet implemented!")
                return ""
        
    def unknown(self, text):
        match text:
            case "archdir":
                if len(self.path) >  0:
                    return False
                else:
                    return True
            case "outputdir":
                if len(self.outputDir) >  0:
                    return False
                else:
                    return True
            case "dataType":
                if self.data_type == self.UNSPECIFIED:
                    return True
                else:
                    return False
            case "datatype":
                if self.data_type == self.UNSPECIFIED:
                    return True
                else:
                    return False
            case "backgroundFile":
                if len(self.air_scan_file) >  0:
                    return False
                else:
                    return True
            case "backgroundfile":
                if len(self.air_scan_file) >  0:
                    return False
                else:
                    return True
                return self.air_scan_file
            case "darkCurrentFile":
                if len(self.dark_scan_file) >  0:
                    return False
                else:
                    return True
            case "darkcurrentfile":
                if len(self.dark_scan_file) >  0:
                    return False
                else:
                    return True
            case "sfile":
                if len(self.raw_scan_file) >  0:
                    return False
                else:
                    return True
            case "pfile":
                if len(self.projection_file) >  0:
                    return False
                else:
                    return True
            case "rfile":
                if len(self.reconstruction_file) >  0:
                    return False
                else:
                    return True
            case "systemGeometryFile":
                if len(self.geometry_file) >  0:
                    return False
                else:
                    return True
            case "lengthUnits":
                return False
            case "bgeometry":
                return False
            case "geometry":
                return False
            case "sod":
                if self.leapct.get_sod() > 0.0:
                    return False
                else:
                    return True
            case "sdd":
                if self.leapct.get_sdd() > 0.0:
                    return False
                else:
                    return True
            case "odd":
                if self.leapct.get_sod() > 0.0 and self.leapct.get_sdd() > 0.0:
                    return False
                else:
                    return True
            case "helicalpitch":
                return False
            case "helicalPitch":
                return False
            case "normalizedHelicalPitch":
                return False
            case "axisOfSymmetry":
                axisOfSymmetry = self.leapct.get_axisOfSymmetry()
                if np.abs(self.leapct.get_axisOfSymmetry()) <= 30.0:
                    return False
                else:
                    return True
            case "nangles":
                if self.leapct.get_numAngles() > 0:
                    return False
                else:
                    return True
            case "initangle":
                phis = self.leapct.get_angles()
                if phis is None or len(phis) == 0:
                    return True
                else:
                    return False
            case "initAngle":
                phis = self.leapct.get_angles()
                if phis is None or len(phis) == 0:
                    return True
                else:
                    return False
            case "arange":
                if self.leapct.get_angularRange() == 0.0:
                    return True
                else:
                    return False
            case "rotationDirection":
                if self.leapct.get_angularRange() == 0.0:
                    return True
                else:
                    return False
            case "rotationdirection":
                if self.leapct.get_angularRange() == 0.0:
                    return True
                else:
                    return False
            case "nrays":
                if self.leapct.get_numCols() > 0:
                    return False
                else:
                    return True
            case "nslices":
                if self.leapct.get_numRows() > 0:
                    return False
                else:
                    return True
            case "pxcenter":
                return False
            case "pzcenter":
                return False
            case "pxmidoff":
                return False
            case "pxsize":
                if self.leapct.get_pixelWidth() > 0.0:
                    return False
                else:
                    return True
            case "pzsize":
                if self.leapct.get_pixelHeight() > 0.0:
                    return False
                else:
                    return True
            case "detectorShape":
                return False
            case "detectorResponseFile":
                if len(self.detector_response_file) == 0:
                    return True
                else:
                    return False
            case "kV":
                if self.kV > 0.0:
                    return False
                else:
                    return True
            case "takeOffAngle":
                if self.takeoff_angle > 0.0:
                    return False
                else:
                    return True
            case "anode_normal":
                if self.anode_normal is not None:
                    return False
                else:
                    return True
            case "anodeMaterial":
                return False
            case "filterMaterials":
                if self.source_filters is None and self.detector_filters is None:
                    return True
                else:
                    return False
            case "sourceFilters" | "source_filters":
                if self.source_filters is None:
                    return True
                else:
                    return False
            case "detectorFilters" | "detector_filters":
                if self.detector_filters is None:
                    return True
                else:
                    return False
            case "spectraFile":
                #if len(self.spectra_model_file) > 0:
                if len(self.source_spectra_file) > 0:
                    return False
                else:
                    return True
            case "referenceEnergy":
                if self.reference_energy > 0.0:
                    return False
                else:
                    return True
            case "numTVneighbors":
                return False
            case "projector" | "backprojector":
                return False
            case "rfilter":
                return False
            case "rxsize":
                if self.leapct.get_voxelWidth() > 0.0:
                    return False
                else:
                    return True
            case "rysize":
                if self.leapct.get_voxelWidth() > 0.0:
                    return False
                else:
                    return True
            case "rzsize":
                if self.leapct.get_voxelHeight() > 0.0:
                    return False
                else:
                    return True
            case "rxref":
                if self.leapct.get_voxelWidth() <= 0.0:
                    return True
                else:
                    return False
            case "ryref":
                if self.leapct.get_voxelWidth() <= 0.0:
                    return True
                else:
                    return False
            case "rzref":
                if self.leapct.get_voxelHeight() <= 0.0:
                    return True
                else:
                    return False
            case "rxoffset":
                return False
            case "ryoffset":
                return False
            case "rzoffset":
                return False
            case "rxelements":
                if self.leapct.get_numX() > 0:
                    return False
                else:
                    return True
            case "ryelements":
                if self.leapct.get_numY() > 0:
                    return False
                else:
                    return True
            case "rzelements":
                if self.leapct.get_numZ() > 0:
                    return False
                else:
                    return True
            case "halfscan":
                return False
            case "ImageJpath":
                return True
            case "LTTcmd":
                return True
            case "wmin":
                return True
            case "wmax":
                return True
            case "compressFile":
                return True
            case "LTTwCmd":
                return True
            case "trackHistory":
                return True
            case "fileType":
                return False
            case _:
                print("Error: getParam keyword " + str(text) +  " not yet implemented!")
                return ""

    def loadsct(self, fileName):
        if fileName.endswith('.sct'):
            fdes = open(fileName, 'r')
            Lines = fdes.readlines()
            for line in Lines:
                if line[0] == '-':
                    line = line[1:].strip()
                    x = line.split(' ', 1)
                    if len(x) == 2:
                        self.set_key_value_pairs(x[0], x[1])
        else:
            print('This is not an sct file')
            
    def load_skyscan(self, fileName):
        if fileName.endswith('.log'):
            fdes = open(fileName, 'r')
            Lines = fdes.readlines()
            for line in Lines:
                if "=" in line:
                    self.set_cmd(line, False)
            self.path = os.path.split(fileName)[0]
            self.leapct.set_geometry("CONE")
            self.leapct.set_flatDetector()
            self.air_scan_file = "57363.766"
            self.data_type = self.RAW_DARK_SUBTRACTED
            self.leapct.set_centerCol((self.leapct.get_numCols()-1)/2.0)
            self.leapct.set_centerRow(self.leapct.get_numRows()-1-self.leapct.get_centerRow())
            self.takeoff_angle = 38.0
            self.set_detector_response('Gd2O2S', 7.32e-3, 0.02)
        else:
            print('This is not a Skyscan/ Bruker log file')
    
    def load_parameters(self, fileName):
        self.load_key_equal_value(fileName)
        self.load_geometry_file()
        self.geometry_file = None
    
    def load_key_equal_value(self, fileName):
        if os.path.isfile(fileName) == False:
            fileName = os.path.join(self.path, fileName)
            if os.path.isfile(fileName) == False:
                print('Error: meta-data file does not exist!')
                return
        fdes = open(fileName, 'r')
        Lines = fdes.readlines()
        for line in Lines:
            line = line.lstrip()
            if len(line) == 0 or line[0] == '#':
                pass
            elif "=" in line:
                #print(line)
                self.set_cmd(line, False)
            else:
                line = line.rstrip()
                if line.endswith('.txt'):
                    self.load_key_equal_value(line)
        if self.path is None or len(self.path) == 0:
            self.path = os.path.split(fileName)[0]
        
        
    def getHelpText(self, text, length=0):
        return "---"
        
    def get_nangles(self):
        return self.leapct.get_numAngles()
        
    def get_nrays(self):
        return self.leapct.get_numCols()
        
    def get_rxoffset(self):
        return 0
        
    def get_ryoffset(self):
        return 0
        
    def get_rzoffset(self):
        return 0
        
    def get_rxelements(self):
        return self.leapct.get_numX()
        
    def get_ryelements(self):
        return self.leapct.get_numY()
        
    def get_rzelements(self):
        return self.leapct.get_numZ()
        
    def projectionsAllocated(self):
        if self.g is None:
            return False
        else:
            return True
            
    def volumeAllocated(self):
        if self.f is None:
            return False
        else:
            return True
            
    def projectionDataExists(self):
        # FIXME: need to check for file
        return self.projectionsAllocated()
    
    def reconstructionDataExists(self):
        # FIXME: need to check for file
        return self.volumeAllocated()
        
    def getLengthUnits(self):
        return "mm"
    
"""
match text:
case "archdir":
case "dataType":
case "backgroundFile":
case "darkCurrentFile":
case "sfile":
case "pfile":
case "rfile":
case "systemGeometryFile":
case "lengthUnits":
case "bgeometry":
case "geometry":
case "sod":
case "sdd":
case "odd":
case "helicalpitch":
case "nangles":
case "initangle":
case "arange":
case "rotationDirection":
case "nrays":
case "nslices":
case "pxcenter":
case "pzcenter":
case "pxmidoff":
case "pxsize":
case "pzsize":
case "detectorShape":
case "detectorResponseFile":
case "kV":
case "TakeOffAngle":
case "AnodeMaterial":
case "filterMaterials":
case "spectraFile":
case "referenceEnergy":
case "rfilter":
case "rxsize":
case "rysize":
case "rzsize":
case "rxref":
case "ryref":
case "rzref":
case "rxoffset":
case "ryoffset":
case "rzoffset":
case "rxelements":
case "ryelements":
case "rzelements":
case "halfscan":
case _:
    """
     
""" TESTING
lctserver=leapctserver()
lctserver.set_source_spectra(100)
lctserver.physics.use_mm()
lctserver.add_filter('Al',None,2.0)
lctserver.set_detector_response('GOS',None,0.1)
Es,s=lctserver.totalSystemSpectralResponse()

import matplotlib.pyplot as plt
plt.plot(Es,s)
plt.show()
#"""
     
