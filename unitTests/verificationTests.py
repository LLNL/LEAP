import sys
import os
import time
import numpy as np
from LTTserver import LTTserver
from leapctype import *
#sys.path.append(r'..\utils')
sys.path.append(r'C:\Users\champley\Documents\git_leap\LEAP\utils')
from bridgeToLTT import *

objfile = r'C:\Users\champley\Documents\tools\LTT\sampleScripts\FORBILD_head_noEar.pd'

LTT = LTTserver()
leapct = tomographicModels()

test_cpu_methods = False

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
        #LTT.cmd('detectorShape=curved')
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

    setLEAPfromLTT(LTT,leapct)
    #leapct.set_volumeDimensionOrder(0) # XYZ
    leapct.set_volumeDimensionOrder(1) # ZYX

    #'''
    # Test Forward Projection on CPU & GPU    
    LTT.cmd('voxelizePhantom #{overSampling=3}')
    f_true = LTT.getAllReconSlicesZ()
    if leapct.get_volumeDimensionOrder() == 1: # leap is ZYX
        f_true = np.ascontiguousarray(np.flip(f_true, 1), dtype=np.float32) # LTT is ZYX, but Y is flipped
    else: # leap is XYZ
        f_true = np.ascontiguousarray(np.flip(np.swapaxes(f_true, 0, 2),axis=1), dtype=np.float32)
    
    LTT.cmd('project')
    g_LTT = LTT.getAllProjections()

    leapct.set_GPU(0)
    g_leap_GPU = leapct.allocateProjections()
    startTime = time.time()
    leapct.project(g_leap_GPU,f_true)
    print('project GPU elapsed time: ' + str(time.time()-startTime))

    if test_cpu_methods:
        leapct.set_GPU(-1)
        g_leap_CPU = leapct.allocateProjections()
        startTime = time.time()
        leapct.project(g_leap_CPU,f_true)
        print('project CPU elapsed time: ' + str(time.time()-startTime))
        leapct.displayVolume((g_LTT-g_leap_CPU)/np.max(g_LTT))

    leapct.displayVolume((g_LTT-g_leap_GPU)/np.max(g_LTT))
    #'''
    
    #'''
    # Test Backprojection on CPU & GPU
    LTT.cmd('simulate #{overSampling=3}')
    g_true = LTT.getAllProjections()
    
    LTT.cmd('backproject')
    f_LTT = LTT.getAllReconSlicesZ()
    if leapct.get_volumeDimensionOrder() == 1: # leap is ZYX
        f_LTT = np.ascontiguousarray(np.flip(f_LTT, 1), dtype=np.float32) # LTT is ZYX, but Y is flipped
    else: # leap is XYZ
        f_LTT = np.ascontiguousarray(np.flip(np.swapaxes(f_LTT, 0, 2),axis=1), dtype=np.float32)
    
    leapct.set_GPU(0)
    f_leap_GPU = leapct.allocateVolume()
    startTime = time.time()
    leapct.backproject(g_true, f_leap_GPU)
    print('backproject GPU elapsed time: ' + str(time.time()-startTime))
    
    if test_cpu_methods:
        leapct.set_GPU(-1)
        f_leap_CPU = leapct.allocateVolume()
        startTime = time.time()
        leapct.backproject(g_true, f_leap_CPU)
        print('backproject CPU elapsed time: ' + str(time.time()-startTime))
        leapct.displayVolume((f_LTT-f_leap_CPU)/np.max(f_LTT))
    
    leapct.displayVolume((f_LTT-f_leap_GPU)/np.max(f_LTT))
    #'''
    
    #'''
    # Test FBP
    LTT.cmd('simulate #{overSampling=3}')
    g_true = LTT.getAllProjections()
    
    LTT.cmd('FBP')
    f_LTT = LTT.getAllReconSlicesZ()
    if leapct.get_volumeDimensionOrder() == 1: # leap is ZYX
        f_LTT = np.ascontiguousarray(np.flip(f_LTT, 1), dtype=np.float32) # LTT is ZYX, but Y is flipped
    else: # leap is XYZ
        f_LTT = np.ascontiguousarray(np.flip(np.swapaxes(f_LTT, 0, 2),axis=1), dtype=np.float32)
    
    leapct.set_GPU(0)
    f_leap_GPU = leapct.allocateVolume()
    startTime = time.time()
    leapct.FBP(g_true, f_leap_GPU)
    print('FBP GPU elapsed time: ' + str(time.time()-startTime))

    if test_cpu_methods:
        leapct.set_GPU(-1)
        f_leap_CPU = leapct.allocateVolume()
        startTime = time.time()
        leapct.FBP(g_true, f_leap_CPU)
        print('FBP CPU elapsed time: ' + str(time.time()-startTime))
        leapct.displayVolume((f_LTT-f_leap_CPU)/np.max(f_LTT))
    
    leapct.displayVolume((f_LTT-f_leap_GPU)/np.max(f_LTT))
    #'''
    
    #quit()
    