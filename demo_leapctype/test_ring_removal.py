import sys
import os
import time
import numpy as np
from leapctype import *
leapct = tomographicModels()
sys.path.append(r'..\utils')
from preprocessing_algorithms import *

# Specify the number of detector columns which is used below
# Scale the number of angles and the detector pixel size with N
numCols = 512
numAngles = 2*2*int(360*numCols/1024)
pixelSize = 0.65*512/numCols*11/14

# Set the number of detector rows
numRows = 64

# Set the scanner geometry
# Let's start with a standard parallel-beam geometry, but this demo works for any CT geometry
leapct.set_parallelbeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1)+10, leapct.setAngleArray(numAngles, 360.0))

# Set the volume parameters.
# It is best to do this after the CT geometry is set
leapct.set_default_volume()

# If you want to specify the volume yourself, use this function:
#leapct.set_volume(numX, numY, numZ, voxelWidth=None, voxelHeight=None, offsetX=None, offsetY=None, offsetZ=None):


# Allocate space for the projections and the volume
g = leapct.allocateProjections()
f_true = leapct.allocateVolume()


# Specify simplified FORBILD head phantom
leapct.set_FORBILD(f_true,True)


# "Simulate" projection data
leapct.project(g,f_true)


# Add random detector gain to each pixel which will create ring artifacts in the reconstruction
detectorGain = np.random.uniform(1.0-0.04,1.0+0.04,(numRows,numCols))
g[:] = g[:] - np.log(detectorGain[None,:,:])

# Add noise to the data (just for demonstration purposes)
I_0 = 50000.0
g[:] = -np.log(np.random.poisson(I_0*np.exp(-g))/I_0)


# Perform ring removal
#g = ringRemoval_fast(leapct, g, 1.0-0.99, 30, 0.05)
g = ringRemoval(leapct, g, 1.0-0.99, 1.0e3, 30)


# Reconstruct the data
f = leapct.allocateVolume()
leapct.FBP(g,f)


# Display the result with napari
leapct.display(f)
