"""
This file contains functions to convert between LEAP and TIGRE geometry definitions.

I do not gaurantee that this works correctly in all cases and some features may be missing!!!
"""
from leapctype import *
import tigre
from scipy.spatial.transform import Rotation as R

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
    row_offs = geo.offDetector[0]
    col_offs = geo.offDetector[1]
    phis = (geo.angles + 0.5*np.pi)*180.0/np.pi
    DSO = geo.DSO
    DSD = geo.DSD
    yaw = geo.rotDetector[2]
    pitch = geo.rotDetector[1]
    roll = geo.rotDetector[0]

    # Calculate rotation matrix
    A = R.from_euler('xyz', [roll, pitch, yaw], degrees=False).as_matrix()

    # Remove rotation to transfer it to tau
    R_yaw_inv = R.from_euler('xyz', [0.0, 0.0, -yaw], degrees=False).as_matrix()

    detectorPosition = np.array([DSO-DSD, col_offs, row_offs], dtype=np.float32)
    sourcePosition = np.array([DSO, 0.0, 0.0], dtype=np.float32)

    sourcePosition_new = np.matmul(R_yaw_inv, sourcePosition)
    detectorPosition_new = np.matmul(R_yaw_inv, detectorPosition)
    tau = -sourcePosition_new[1]
    sod = sourcePosition_new[0]

    n_vec = A[:,0] # from detector to source
    u_vec = A[:,1]
    v_vec = A[:,2]

    n_vec_new = np.matmul(R_yaw_inv, n_vec)
    u_vec_new = np.matmul(R_yaw_inv, u_vec)
    v_vec_new = np.matmul(R_yaw_inv, v_vec)

    sdd = np.abs(np.sum((sourcePosition_new-detectorPosition_new)*n_vec_new))

    temp = detectorPosition_new - sourcePosition_new
    u_offs = np.sum(temp*u_vec_new)
    v_offs = np.sum(temp*v_vec_new)

    phis += yaw*180.0/np.pi
    
    centerRow = 0.5*(numRows-1) - v_offs/pixelHeight
    centerCol = 0.5*(numCols-1) - u_offs/pixelWidth
    
    leapct.set_conebeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau, 0.0, roll*180.0/np.pi, pitch*180.0/np.pi)
    
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
    tau = leapct.get_tau()

    yaw = np.arctan(tau/sod)
    roll = leapct.get_tiltAngle()*np.pi/180.0
    pitch = leapct.get_pitchAngle()*np.pi/180.0

    # Remove rotation to transfer it to tau
    R_yaw = R.from_euler('xyz', [0.0, 0.0, yaw], degrees=False).as_matrix()
    
    u_offs = (0.5*(numCols-1) - centerCol)*pixelWidth
    v_offs = (0.5*(numRows-1) - centerRow)*pixelHeight
    
    phis -= yaw

    A_noyaw = R.from_euler('xyz', [roll, pitch, 0.0], degrees=False).as_matrix()
    u_vec_new = A_noyaw[:,1]
    v_vec_new = A_noyaw[:,2]
    
    sourcePosition = np.array([sod, -tau, 0.0], dtype=np.float32)
    detectorPosition = sourcePosition - sdd*np.array([np.cos(pitch), 0.0, -np.sin(pitch)], dtype=np.float32)
    detectorPosition += u_offs*u_vec_new + v_offs*v_vec_new

    sourcePosition_new = np.matmul(R_yaw, sourcePosition)
    detectorPosition_new = np.matmul(R_yaw, detectorPosition)

    col_offs = detectorPosition_new[1]
    row_offs = detectorPosition_new[2]

    # Set TIGRE geometry parameters
    geo.mode = "cone"
    geo.nVoxel = np.array([leapct.get_numZ(), leapct.get_numY(), leapct.get_numX()])
    geo.DSD = sourcePosition_new[0] - detectorPosition_new[0]
    geo.DSO = np.sqrt(sod**2 + tau**2)
    geo.rotDetector = np.array([roll, pitch, yaw])

    geo.dDetector = np.array([pixelHeight, pixelWidth])
    geo.nDetector = np.array([numRows, numCols])
    geo.sDetector = geo.dDetector * geo.nDetector
    geo.offDetector = np.array([row_offs, col_offs])
     
    geo.dVoxel = np.array([leapct.get_voxelHeight(), leapct.get_voxelWidth(), leapct.get_voxelWidth()])
    geo.nVoxel = np.array([leapct.get_numZ(), leapct.get_numY(), leapct.get_numX()])
    geo.sVoxel = geo.dVoxel * geo.nVoxel
    geo.offOrigin = np.array([0.0, 0.0, 0.0])
    geo.offOrigin[2] = leapct.get_offsetX()
    geo.offOrigin[1] = leapct.get_offsetY()
    geo.offOrigin[0] = leapct.get_offsetZ()
    geo.angles = phis
    
    return geo
    
