import sys
import os
import time
import numpy as np
from leapctype import *
leapct = tomographicModels()


# Specify the number of detector columns which is used below
# Scale the number of angles and the detector pixel size with N
numCols = 512
numAngles = 2*2*int(360*numCols/1024)
pixelSize = 0.65*512/numCols

# Set the number of detector rows
numRows = 64

# Set the scanner geometry
# Let's start with a standard cone-beam geometry so make it easier to specify,
# then convert it to modular beam for we can perform detector dithering
leapct.set_conebeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0), 1100, 1400)
leapct.convert_conebeam_to_modularbeam()

# Set the volume parameters.
# It is best to do this after the CT geometry is set
leapct.set_default_volume()

# If you want to specify the volume yourself, use this function:
#leapct.set_volume(numX, numY, numZ, voxelWidth=None, voxelHeight=None, offsetX=None, offsetY=None, offsetZ=None):

# This is not necessary, but let's set the FOV mask,
# so we mask out those voxels outside the field of view
leapct.set_diameterFOV(leapct.get_numX()*leapct.get_voxelWidth())

#'''
# These steps randonly shift the detector for each view.
# This "detector dithering" method remove ring artifacts
# If you comment out this section, no dithering will be done
# and the reconstruction will have ring artifacts
sourcePositions = leapct.get_sourcePositions()
moduleCenters = leapct.get_moduleCenters()
rowVectors = leapct.get_rowVectors()
colVectors = leapct.get_colVectors()
for i in range(numAngles):
    colShift = pixelSize*np.random.uniform(-4.0,4.0,1)
    rowShift = pixelSize*np.random.uniform(-4.0,4.0,1)
    
    moduleCenters[i,:] = colShift*colVectors[i,:] + rowShift*rowVectors[i,:] 

leapct.set_modularbeam(numAngles, numRows, numCols, pixelSize, pixelSize, sourcePositions, moduleCenters, rowVectors, colVectors)
#'''


# Allocate space for the projections and the volume
g = leapct.allocateProjections()
f_true = leapct.allocateVolume()


# Specify simplified FORBILD head phantom
leapct.set_FORBILD(f_true,True)


# "Simulate" projection data
leapct.project(g,f_true)


# Add random detector gain to each pixel
detectorGain = np.random.uniform(1.0-0.05,1.0+0.05,(numRows,numCols))
g[:] = g[:] - np.log(detectorGain[None,:,:])


# Add noise to the data (just for demonstration purposes)
I_0 = 50000.0
#g[:] = -np.log(np.random.poisson(I_0*np.exp(-g))/I_0)


# Reconstruct the data
f = leapct.allocateVolume()
leapct.FBP(g,f)


# Display the result with napari
leapct.display(f)
