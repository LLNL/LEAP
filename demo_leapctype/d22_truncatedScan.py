import sys
import os
import time
import numpy as np
from leapctype import *
leapct = tomographicModels()


'''
This script demonstrates how to perform an FBP reconstruction where the projections are truncated
on both the left and right side (i.e., the object extends past the detector on the left and right sides)
In this case, you should use the command: leapct.set_truncatedScan(True)
which uses extrapolation of the signal instead of zero-padding when applying the ramp filter
this reduces cupping artifacts and other truncation artifacts
'''


# Specify the number of detector columns which is used below
# Scale the number of angles and the detector pixel size with N
numCols = 512
numAngles = 4*int(360*numCols/1024)
pixelSize = 0.65*512/numCols

# Set the number of detector rows
numRows = numCols

# Reduce the number of columns so that the phantom extends past either the
# right or left side of the projections (but not both)
# Note that we translate the centerCol parameter so that only one side is truncated
numCols_notTruncated = numCols
numCols = numCols - int(80.0*numCols/512.0)
centerCol = 0.5*(numCols-1)

# Set the scanner geometry
#leapct.set_parallelbeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), centerCol, leapct.setAngleArray(numAngles, 360.0))
#leapct.set_fanbeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), centerCol, leapct.setAngleArray(numAngles, 360.0), 1100, 1400)
leapct.set_conebeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), centerCol, leapct.setAngleArray(numAngles, 360.0), 1100, 1400)

# Set the volume parameters
# You should expand the diameter of the field of view (see set_diameterFOV), so that it covers the whole object
# Try commenting out the set_truncatedScan command to see how bad the artifacts get
# if not accounted for in the reconstruction
leapct.set_volume(numCols_notTruncated,numCols_notTruncated,numRows)
leapct.set_diameterFOV(262.0)
leapct.set_truncatedScan(True)

# Trouble-Shooting Functions
leapct.print_parameters()
#leapct.sketch_system()

# Allocate space for the projections and the volume
g = leapct.allocateProjections()
f = leapct.allocateVolume()

# Specify simplified FORBILD head phantom
# One could easily do this in Python, but Python is soooooo slow for these types of operations,
# so we implemented this feature with multi-threaded C++
leapct.set_FORBILD(f,True)
#leapct.display(f)


# "Simulate" projection data
startTime = time.time()
leapct.project(g,f)
print('Forward Projection Elapsed Time: ' + str(time.time()-startTime))
#leapct.display(g)

# Add noise to the data (just for demonstration purposes)
I_0 = 50000.0
#g[:] = -np.log(np.random.poisson(I_0*np.exp(-g))/I_0)


# Reconstruct the data
startTime = time.time()
leapct.FBP(g,f)
print('Reconstruction Elapsed Time: ' + str(time.time()-startTime))

# For truncated scans, there are a lot of negative values, so we'll remove them
f[f<0.0] = 0.0

# Post Reconstruction Smoothing (optional)
#startTime = time.time()
#leapct.diffuse(f,0.02/20.0,4)
#leapct.MedianFilter(f)
#leapct.BlurFilter(f,2.0)
#print('Post-Processing Elapsed Time: ' + str(time.time()-startTime))

# Display the result with napari
leapct.display(f)
