################################################################################
# Copyright 2022-2024 Lawrence Livermore National Security, LLC and other 
# LEAP project developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# LivermorE AI Projector for tomographic reconstruction (LEAP)
# The functions here set LEAP parameters from LTT parameter settings and vice-versa.
# Some parameters mappings are not yet implemented, such as non equi-spaced
# projection angles
################################################################################
import sys
import os
import numpy as np
from LTTserver import LTTserver
from leapctype import *


def setLEAPfromLTT(LTT,leapct):

    geometry = LTT.getParam('geometry')

    numAngles = int(LTT.getParam('nangles'))
    numRows = int(LTT.getParam('numRows'))
    numCols = int(LTT.getParam('numCols'))
    
    if numAngles <= 0 or numRows <= 0 or numCols <= 0:
        print('Error: LTT CT geometry not set')
        return
    
    arange = float(LTT.getParam('arange'))
    pixelHeight = float(LTT.getParam('pixelHeight'))
    pixelWidth = float(LTT.getParam('pixelWidth'))
    
    centerRow = float(LTT.getParam('centerRow'))
    centerCol = float(LTT.getParam('centerCol'))
    
    if geometry == 'CONE':
        sod = float(LTT.getParam('sod'))
        sdd = float(LTT.getParam('sdd'))
        tau = 0.0
        if LTT.unknown('helicalPitch') == False:
            helicalPitch = float(LTT.getParam('helicalPitch'))
        else:
            helicalPitch = 0.0
        leapct.set_conebeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, leapct.setAngleArray(numAngles, arange), sod, sdd, tau, helicalPitch)
        if LTT.getParam('detectorShape') == 'FLAT':
            leapct.set_flatDetector()
        else:
            leapct.set_curvedDetector()
    elif geometry == 'FAN':
        sod = float(LTT.getParam('sod'))
        sdd = float(LTT.getParam('sdd'))
        tau = 0.0
        leapct.set_fanbeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, leapct.setAngleArray(numAngles, arange), sod, sdd, tau)
    elif geometry == 'PARALLEL':
        leapct.set_parallelbeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, leapct.setAngleArray(numAngles, arange))
    elif geometry == 'MODULAR':
        geometryDesc = LTT.getModularBeamGeometry()
        if len(geometryDesc) > 0:
            sourcePositions = np.zeros((numAngles,3),dtype=np.float32)
            moduleCenters = np.zeros((numAngles,3),dtype=np.float32)
            colVectors = np.zeros((numAngles,3),dtype=np.float32)
            rowVectors = np.zeros((numAngles,3),dtype=np.float32)
            for i in range(0,numAngles):
                offset = 12*i
                sourcePositions[i,:] = geometryDesc[offset:offset+3]
                moduleCenters[i,:] = geometryDesc[offset+3:offset+6]
                colVectors[i,:] = geometryDesc[offset+6:offset+9]
                rowVectors[i,:] = geometryDesc[offset+9:offset+12]
            leapct.set_modularbeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, sourcePositions, moduleCenters, rowVectors, colVectors)
    else:
        print('Error: unknown geometry')
        return
    
    if LTT.unknown('axisOfSymmetry') == False:
        leapct.set_axisOfSymmetry(float(LTT.getParam('axisOfSymmetry')))
        
    numX = int(LTT.getParam('rxelements'))
    numY = int(LTT.getParam('ryelements'))
    numZ = int(LTT.getParam('rzelements'))
    
    voxelWidth = float(LTT.getParam('rxsize'))
    voxelHeight = float(LTT.getParam('rzsize'))
    
    #x samples: x[i] = (i + rxoffset - rxref)*rxsize, for i = 0, 1, ..., rxelements-1
    #y samples: y[j] = (j + ryoffset - ryref)*rysize, for j = 0, 1, ..., ryelements-1
    #z samples: z[k] = (k + rzoffset - rzref)*rzsize, for k = 0, 1, ..., rzelements-1
    offsetX = (float(LTT.getParam('rxoffset')) - float(LTT.getParam('rxref')) + 0.5*float(numX-1))*float(LTT.getParam('rxsize'))
    offsetY = (float(LTT.getParam('ryoffset')) - float(LTT.getParam('ryref')) + 0.5*float(numY-1))*float(LTT.getParam('rysize'))
    if geometry == 'PARALLEL' or geometry == 'FAN':
        offsetZ = (float(LTT.getParam('rzoffset')) - float(LTT.getParam('rzref')))*float(LTT.getParam('rzsize')) + float(centerRow)*float(LTT.getParam('pzsize'))
    elif geometry == 'CONE':
        #offsetZ = centerRow * sod/sdd * pixelHeight + (float(LTT.getParam('rzoffset')) - float(LTT.getParam('rzref'))) * float(LTT.getParam('rzsize'))
        offsetZ = 0.5*float(numZ-1)*float(LTT.getParam('rzsize')) + (float(LTT.getParam('rzoffset')) - float(LTT.getParam('rzref'))) * float(LTT.getParam('rzsize'))
    else:
        offsetZ = 0.0
    
    leapct.set_volume(numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ)

    
def setLTTfromLEAP(leapct,LTT):
    geometry = leapct.get_geometry()
    
    sod = leapct.get_sod()
    sdd = leapct.get_sdd()
    numAngles = leapct.get_numAngles()
    numRows = leapct.get_numRows()
    numCols = leapct.get_numCols()
    pixelHeight = leapct.get_pixelHeight()
    pixelWidth = leapct.get_pixelWidth()
    centerRow = leapct.get_centerRow()
    centerCol = leapct.get_centerCol()
    helicalPitch = leapct.get_helicalPitch()
    phis = leapct.get_angles()
    #angularRange = (phis[1]-phis[0])*phis.size
    if phis.size > 1:
        angularRange = (phis[-1]-phis[0]) + phis[1]-phis[0]
    else:
        angularRange = 360.0
    
    numX = leapct.get_numX()
    numY = leapct.get_numY()
    numZ = leapct.get_numZ()
    voxelHeight = leapct.get_voxelHeight()
    voxelWidth = leapct.get_voxelWidth()
    offsetX = leapct.get_offsetX()
    offsetY = leapct.get_offsetY()
    offsetZ = leapct.get_offsetZ()
    rxref = 0.5*float(numX-1) - offsetX/voxelWidth
    ryref = 0.5*float(numY-1) - offsetY/voxelWidth
    if geometry == 'PARALLEL' or geometry == 'FAN':
        rzref = centerRow - offsetZ/voxelHeight
    else:
        #rzref = (centerRow*(sod/sdd)*pixelHeight) / voxelHeight
        rzref = (0.5*float(numZ-1)*voxelHeight - offsetZ) / voxelHeight
    LTT.cmd(['rxelements = ' + str(numX), 'ryelements = ' + str(numY), 'rzelements = ' + str(numZ)])
    LTT.cmd(['rxsize = ' + str(voxelWidth), 'rysize = ' + str(voxelWidth), 'rzsize = ' + str(voxelHeight)])
    LTT.cmd(['rxref = ' + str(rxref), 'ryref = ' + str(ryref), 'rzref = ' + str(rzref)])
    
    if geometry == 'MODULAR':
        print('ERROR: Conversion of modular-beam not yet implemented!')
        return
    
    LTT.cmd('geometry = ' + str(geometry))
    if geometry == 'CONE':
        detectorType = leapct.get_detectorType()
        LTT.cmd(['sod = ' + str(sod), 'sdd = ' + str(sdd), 'detectorShape = ' + str(detectorType)])
        LTT.cmd('helicalPitch = ' + str(helicalPitch))
    elif geometry == 'FAN':
        LTT.cmd(['sod = ' + str(sod), 'sdd = ' + str(sdd), 'detectorShape = flat'])
    LTT.cmd(['numAngles = ' + str(numAngles), 'numRows = ' + str(numRows), 'numCols = ' + str(numCols)])
    LTT.cmd(['angularRange = ' + str(angularRange), 'pixelHeight = ' + str(pixelHeight), 'pixelWidth = ' + str(pixelWidth)])
    LTT.cmd(['centerRow = ' + str(centerRow), 'centerCol = ' + str(centerCol)])

def parseGeomFile(infile):
    import re
    sources=[]
    dets=[]
    rowVector=[]
    colVector=[]
    with open(infile) as fp:
        for result in re.findall('sourcePosition=\((.*?)\)', fp.read(), re.S):
            sources.append(list(eval(result)))
    with open(infile) as fp:
        for result in re.findall('moduleCenter=\((.*?)\)', fp.read(), re.S):
            dets.append(list(eval(result)))
    with open(infile) as fp:
        for result in re.findall('rowVector=\((.*?)\)', fp.read(), re.S):
             rowVector.append(list(eval(result)))
    with open(infile) as fp:
        for result in re.findall('columnVector=\((.*?)\)', fp.read(), re.S):
             colVector.append(list(eval(result)))
    return sources, dets, rowVector, colVector

''' Example Usage
leapct = tomographicModels()
LTT = LTTserver()

# Set the scanner geometry
numCols = 512
numAngles = 2*2*int(360*numCols/1024)
pixelSize = 0.65*512/numCols
numRows = numCols
leapct.set_conebeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0), 1100, 1400)
leapct.set_default_volume()

setLTTfromLEAP(leapct,LTT)

# Now print the LTT and LEAP parameters to see that they match
LTT.cmd('printgeometry')
LTT.cmd('printvolume')
leapct.print_parameters()
#'''
