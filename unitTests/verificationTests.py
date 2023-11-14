# This script performs regression tests for the non-AI portions of LEAP
# It requires Livermore Tomography Tools (LTT) to run
import sys
import os
import time
import numpy as np
from LTTserver import LTTserver
from leapctype import *

#objfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'FORBILD_head.pd')
objfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'FORBILD_head_noEar.pd')

LTT = LTTserver()
leapct = tomographicModels()

def setLEAPfromLTT():
    numAngles = int(LTT.getParam('nangles'))
    numRows = int(LTT.getParam('numRows'))
    numCols = int(LTT.getParam('numCols'))
    
    arange = float(LTT.getParam('arange'))
    pixelHeight = float(LTT.getParam('pixelHeight'))
    pixelWidth = float(LTT.getParam('pixelWidth'))
    
    centerRow = float(LTT.getParam('centerRow'))
    centerCol = float(LTT.getParam('centerCol'))
    
    geometry = LTT.getParam('geometry')
    if geometry == 'CONE':
        sod = float(LTT.getParam('sod'))
        sdd = float(LTT.getParam('sdd'))
        leapct.setConeBeamParams(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, leapct.setAngleArray(numAngles, arange), sod, sdd)
    elif geometry == 'FAN':
        sod = float(LTT.getParam('sod'))
        sdd = float(LTT.getParam('sdd'))
        leapct.setFanBeamParams(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, leapct.setAngleArray(numAngles, arange), sod, sdd)
    elif geometry == 'PARALLEL':
        leapct.setParallelBeamParams(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, leapct.setAngleArray(numAngles, arange))
    
    if LTT.unknown('axisOfSymmetry') == False:
        leapct.set_axisOfSymmetry(float(LTT.getParam('axisOfSymmetry')))
        
    numX = int(LTT.getParam('rxelements'))
    numY = int(LTT.getParam('ryelements'))
    numZ = int(LTT.getParam('rzelements'))
    
    voxelWidth = float(LTT.getParam('rxsize'))
    voxelHeight = float(LTT.getParam('rzsize'))
    
    offsetX = float(LTT.getParam('rxref')) - 0.5*float(numX-1)
    offsetY = float(LTT.getParam('ryref')) - 0.5*float(numY-1)
    offsetZ = float(LTT.getParam('rzref')) - 0.5*float(numZ-1)
    
    leapct.setVolumeParams(numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ)
    
for n in range(3):
    LTT.cmd('clearAll')
    LTT.cmd('diskIO=off')
    LTT.cmd('archdir=pwd')
    LTT.cmd('objfile = ' + str(objfile))
    #LTT.cmd('axisOfSymmetry = 0.0')
    pixelSize = 1.0
    numAngles = int(720.0/pixelSize)
    if LTT.unknown('axisOfSymmetry') == False:
        numAngles = 1
    if n == 0:
        print('********* CONE-BEAM *********')
        LTT.cmd(['geometry=cone','detectorShape=flat','sdd=1400','sod=1100','numAngles = ' + str(numAngles),'arange=360','pixelSize='+str(pixelSize),'numRows=340/'+str((pixelSize)),'numCols=320/'+str((pixelSize)),'centerRow=(numRows-1)/2','centerCol=(numCols-1)/2'])
    elif n == 1:
        if LTT.unknown('axisOfSymmetry') == False:
            continue
        print('********* FAN-BEAM *********')
        LTT.cmd(['geometry=fan','detectorShape=flat','sdd=1400','sod=1100','numAngles = ' + str(numAngles),'arange=360','pixelWidth='+str(pixelSize),'pixelHeight=pixelWidth*11/14','numRows=340/'+str((pixelSize)),'numCols=320/'+str((pixelSize)),'centerRow=(numRows-1)/2','centerCol=(numCols-1)/2'])
    elif n == 2:
        print('********* PARALLEL-BEAM *********')
        LTT.cmd(['geometry=parallel','numAngles = ' + str(numAngles),'arange=360','pixelSize='+str(pixelSize)+'*11/14','numRows=340/'+str((pixelSize)),'numCols=320/'+str((pixelSize)),'centerRow=(numRows-1)/2','centerCol=(numCols-1)/2'])
    LTT.cmd('defaultVolume')
    LTT.cmd('spectraFile =63.817')
    LTT.cmd('dataType=atten')
    LTT.cmd('projectorType=SF')

    setLEAPfromLTT()
    #leapct.setVolumeDimensionOrder(0) # XYZ
    leapct.setVolumeDimensionOrder(1) # ZYX

    #'''
    # Test Forward Projection on CPU & GPU    
    LTT.cmd('voxelizePhantom #{overSampling=3}')
    f_true = LTT.getAllReconSlicesZ()
    if leapct.getVolumeDimensionOrder() == 1: # leap is ZYX
        f_true = np.ascontiguousarray(np.flip(f_true, 1), dtype=np.float32) # LTT is ZYX, but Y is flipped
    else: # leap is XYZ
        f_true = np.ascontiguousarray(np.flip(np.swapaxes(f_true, 0, 2),axis=1), dtype=np.float32)
    
    LTT.cmd('project')
    g_LTT = LTT.getAllProjections()

    leapct.setGPU(0)
    g_leap_GPU = leapct.allocateProjections()
    startTime = time.time()
    leapct.project(g_leap_GPU,f_true)
    print('project GPU elapsed time: ' + str(time.time()-startTime))

    leapct.setGPU(-1)
    g_leap_CPU = leapct.allocateProjections()
    startTime = time.time()
    leapct.project(g_leap_CPU,f_true)
    print('project CPU elapsed time: ' + str(time.time()-startTime))

    #leapct.displayVolume(g_LTT)
    leapct.displayVolume((g_LTT-g_leap_GPU)/np.max(g_LTT))
    leapct.displayVolume((g_LTT-g_leap_CPU)/np.max(g_LTT))
    #leapct.displayVolume(g_leap_GPU)
    #leapct.displayVolume(g_leap_CPU)
    #'''
    
    quit()
    
    #'''
    # Test Backprojection on CPU & GPU
    LTT.cmd('simulate #{overSampling=3}')
    g_true = LTT.getAllProjections()
    
    LTT.cmd('backproject')
    f_LTT = LTT.getAllReconSlicesZ()
    if leapct.getVolumeDimensionOrder() == 1: # leap is ZYX
        f_LTT = np.ascontiguousarray(np.flip(f_LTT, 1), dtype=np.float32) # LTT is ZYX, but Y is flipped
    else: # leap is XYZ
        f_LTT = np.ascontiguousarray(np.flip(np.swapaxes(f_LTT, 0, 2),axis=1), dtype=np.float32)
    
    leapct.setGPU(0)
    f_leap_GPU = leapct.allocateVolume()
    startTime = time.time()
    leapct.backproject(g_true, f_leap_GPU)
    print('backproject GPU elapsed time: ' + str(time.time()-startTime))
    
    leapct.setGPU(-1)
    f_leap_CPU = leapct.allocateVolume()
    startTime = time.time()
    leapct.backproject(g_true, f_leap_CPU)
    print('backproject CPU elapsed time: ' + str(time.time()-startTime))
    
    #leapct.displayVolume(f_LTT)
    leapct.displayVolume((f_LTT-f_leap_GPU)/np.max(f_LTT))
    leapct.displayVolume((f_LTT-f_leap_CPU)/np.max(f_LTT))
    #leapct.displayVolume(f_leap_GPU)
    #leapct.displayVolume(f_leap_CPU)
    #'''
    
    #'''
    # Test FBP
    LTT.cmd('simulate #{overSampling=3}')
    g_true = LTT.getAllProjections()
    
    LTT.cmd('FBP')
    f_LTT = LTT.getAllReconSlicesZ()
    if leapct.getVolumeDimensionOrder() == 1: # leap is ZYX
        f_LTT = np.ascontiguousarray(np.flip(f_LTT, 1), dtype=np.float32) # LTT is ZYX, but Y is flipped
    else: # leap is XYZ
        f_LTT = np.ascontiguousarray(np.flip(np.swapaxes(f_LTT, 0, 2),axis=1), dtype=np.float32)
    
    leapct.setGPU(0)
    f_leap_GPU = leapct.allocateVolume()
    startTime = time.time()
    leapct.FBP(g_true, f_leap_GPU)
    print('FBP GPU elapsed time: ' + str(time.time()-startTime))

    leapct.setGPU(-1)
    #leapct.setGPUs(np.array([0,1]))
    f_leap_CPU = leapct.allocateVolume()
    startTime = time.time()
    leapct.FBP(g_true, f_leap_CPU)
    print('FBP CPU elapsed time: ' + str(time.time()-startTime))
    
    #leapct.displayVolume(f_LTT)
    leapct.displayVolume((f_LTT-f_leap_GPU)/np.max(f_LTT))
    leapct.displayVolume((f_LTT-f_leap_CPU)/np.max(f_LTT))
    #leapct.displayVolume(f_leap_CPU)
    #leapct.displayVolume(f_leap_GPU)
    #'''
    
    '''
    # Test SART/ RWLS
    LTT.cmd('simulate #{overSampling=3}')
    g_true = LTT.getAllProjections()
    
    LTT.cmd('allocateVolume')
    #LTT.cmd('FBP')
    f_LTT = LTT.getAllReconSlicesZ()
    if leapct.getVolumeDimensionOrder() == 1: # leap is ZYX
        f_LTT = np.ascontiguousarray(np.flip(f_LTT, 1), dtype=np.float32) # LTT is ZYX, but Y is flipped
    else: # leap is XYZ
        f_LTT = np.ascontiguousarray(np.flip(np.swapaxes(f_LTT, 0, 2),axis=1), dtype=np.float32)
    
    leapct.setGPU(0)
    leapct.setGPUs([0,1])
    f_leap_GPU = leapct.allocateVolume()
    startTime = time.time()
    #leapct.SART(g_true, f_leap_GPU, 10)
    leapct.RWLS(g_true, f_leap_GPU, 10, delta=0.0, beta=0.0, W=None, SQS=True)
    print('SART elapsed time: ' + str(time.time()-startTime))
    
    #leapct.displayVolume(f_LTT)
    #leapct.displayVolume((f_LTT-f_leap_GPU)/np.max(f_LTT))
    #leapct.displayVolume((f_LTT-f_leap_CPU)/np.max(f_LTT))
    leapct.displayVolume(f_leap_GPU)
    #'''
    
    #quit()
    