"""
This file contains functions to convert between LEAP and TIGRE geometry definitions.

I do not gaurantee that this works correctly in all cases and some features may be missing!!!
"""
from leapctype import *
import tigre

def set_leap_from_tigre(geo, leapct=None):
    r""" Set LEAP CT geometry and CT volume parameters from TIGRE geometry object
    
    Args:
        geo: TIGRE geometry class object
        leapct: LEAP tomographicModels class object or None
        
    Returns:
        LEAP tomographicModels class object with CT geometry and CT volumes specified to match TIGRE geometry specifications
    """
    if leapct is None:
        leapct = tomographicModels()

    # Set CT geometry
    numAngles = geo.angles.size
    numRows = geo.nDetector[0]
    numCols = geo.nDetector[1]
    pixelHeight = geo.dDetector[0]
    pixelWidth = geo.dDetector[1]
    centerRow = 0.5*(numRows-1) - geo.offDetector[0]/pixelHeight
    centerCol = 0.5*(numCols-1) - geo.offDetector[1]/pixelWidth
    phis = (geo.angles + 0.5*np.pi)*180.0/np.pi
    sod = geo.DSO
    sdd = geo.DSD
    leapct.set_conebeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd)
    
    if np.any(geo.rotDetector != 0.0):
        leapct.convert_to_modularbeam()
        dRoll = geo.rotDetector[2]
        dPitch = geo.rotDetector[0]
        dYaw = geo.rotDetector[1]
        from scipy.spatial.transform import Rotation as R
        A = R.from_euler('zxy', [dRoll, dYaw, dPitch], degrees=False).as_matrix()
        
        leapct.rotate_detector(A)
    
    # Set CT volume
    stageShift = (centerRow-0.5*(numRows-1))*pixelHeight*sod/sdd
    offsetX = 0.0
    offsetY = 0.0
    offsetZ = 0.0
    if hasattr(geo, 'offOrigin'):
        offsetX = geo.offOrigin[2]
        offsetY = geo.offOrigin[1]
        offsetZ = geo.offOrigin[0]#-stageShift
    leapct.set_volume(geo.nVoxel[2], geo.nVoxel[1], geo.nVoxel[0], geo.dVoxel[1], geo.dVoxel[1], offsetX, offsetY, offsetZ)
    
    return leapct
    
def set_tigre_from_leap(leapct, geo=None):
    r""" Set TIGRE geometry object from LEAP CT geometry and CT volume parameters
    
    Args:
        leapct: LEAP tomographicModels class object
        geo: TIGRE geometry class object or None
        
    Returns:
        TIGRE geometry class object parameters specified to match LEAP CT geometry and CT volume specifications
    """
    if leapct.get_geometry() != 'CONE':
        print('Error: only conversion from LEAP cone-beam geometry has been implemented!')
        return None
    if geo is None:
        geo = tigre.geometry()
    geo.mode = "cone"
    geo.nVoxel = np.array([leapct.get_numZ(), leapct.get_numY(), leapct.get_numX()])
    
    phis = leapct.get_angles()*np.pi/180.0 - 0.5*np.pi
    numAngles = phis.size
    numRows = leapct.get_numRows()
    numCols = leapct.get_numCols()
    pixelHeight = leapct.get_pixelHeight()
    pixelWidth = leapct.get_pixelWidth()
    centerRow = leapct.get_centerRow()
    centerCol = leapct.get_centerCol()
    sod = leapct.get_sod()
    sdd = leapct.get_sdd()
    
    stageShift = (centerRow-0.5*(numRows-1))*pixelHeight*sod/sdd
    
    geo.dDetector = np.array([pixelHeight, pixelWidth])  # size of each pixel            (mm)
    geo.nDetector = np.array([numRows, numCols])
    geo.sDetector = geo.dDetector * geo.nDetector
    geo.DSO = sod
    geo.DSD = sdd
    geo.offDetector = np.array([0.0, 0.0])
    geo.offDetector[0] = -(centerRow - 0.5*(numRows-1)) * pixelHeight
    geo.offDetector[1] = -(centerCol - 0.5*(numCols-1)) * pixelWidth
    geo.dVoxel = np.array([leapct.get_voxelHeight(), leapct.get_voxelWidth(), leapct.get_voxelWidth()])
    geo.nVoxel = np.array([leapct.get_numZ(), leapct.get_numY(), leapct.get_numX()])
    geo.sVoxel = geo.dVoxel * geo.nVoxel
    geo.offOrigin = np.array([0.0, 0.0, 0.0])
    geo.offOrigin[2] = leapct.get_offsetX()
    geo.offOrigin[1] = leapct.get_offsetY()
    geo.offOrigin[0] = leapct.get_offsetZ()
    #geo.offOrigin[0] = leapct.get_offsetZ()+stageShift
    geo.angles = phis
        
    return geo
    