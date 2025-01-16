import sys
import os
import time
import numpy as np
from leapctype import *
leapct = tomographicModels()
from leap_preprocessing_algorithms import *

'''
The script demonstrates two different methods to handle outliers (zingers) and bad pixels in your projection data
'''

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


# Add noise to the data (just for demonstration purposes)
I_0 = 50000.0
g[:] = -np.log(np.random.poisson(I_0*np.exp(-g))/I_0)

# Choose which method you'd like to test
#whichMethod = 1
whichMethod = 2
#whichMethod = 3

if whichMethod == 1:

    # Make some of the detector pixels have zero value
    ind = np.abs(np.random.normal(0,1,g.shape)) > 3.0
    g[ind] = 0.0

    # Perform median filter based outlier correction
    # This is a very fast method which is good for isolated bad pixels
    outlierCorrection(leapct,g)

    # Reconstruct the data
    f = leapct.allocateVolume()
    leapct.FBP(g,f)
elif whichMethod == 2:

    # Make some detector pixels dead for all projections
    ind = np.abs(np.random.normal(0,1,(g.shape[1], g.shape[2]))) > 3.0
    g[:,ind] = 0.0
    
    # Correct for bad pixels
    badPixelMap = np.zeros((g.shape[1], g.shape[2]), dtype=np.float32)
    badPixelMap[ind] = 1.0
    badPixelCorrection(leapct, g, badPixelMap)
    
    # Reconstruct the data
    f = leapct.allocateVolume()
    leapct.FBP(g,f)
else:
    
    # Make some of the detector pixels have zero value
    ind = np.abs(np.random.normal(0,1,g.shape)) > 3.0
    g[ind] = 0.0

    # Perform RWLS reconstruction where we set the weights to be zero where there are bad pixels
    # This method is computationally expensive, but is good when you have large regions of bad pixels
    # or want to perform so-called Metal Artifact Reduction (MAR)
    # For the data we simulated here, this is definitely an overkill.
    
    # First identify the outliers/ bad pixels
    # You can use any method you want; we just use a simple method here
    g_filtered = g.copy()
    g_filtered = outlierCorrection(leapct,g_filtered)
    W = g.copy()
    W[:] = 1.0
    W[np.abs(g-g_filtered)>0.5] = 0.0

    
    # Reconstruct the data with RWLS
    f = leapct.allocateVolume()
    leapct.RWLS(g,f,50, 0.0, 0.0, W, True, True)
    
# Display the result with napari
leapct.display(f)
