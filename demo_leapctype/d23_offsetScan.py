import sys
import os
import time
import numpy as np
from leapctype import *
leapct = tomographicModels()
# Make sure you add: .../LEAP/src to your python path

'''
This script demonstrates how to perform an FBP reconstruction where the projections are truncated
on either the left or the right side (i.e., the object extends past the detector on the left or right side)
In this case, you should use the command: leapct.set_offsetScan(True)

This can happen if the detector is shifted horizontally (do this with the centerCol parameter) and/or
the source is shifted horizontally (do this with the tau parameter).

This is sometimes refered to as a half-fan or half-cone or half-scan.

Sometimes this is not on purpose, but in most cases this is done deliberately because it enables one
to nearly double the diameter of the field of view which is needed for large objects.

Details on this algorithm are covered in the LEAP technical manual here:
https://github.com/LLNL/LEAP/blob/main/documentation/LEAP.pdf
'''


# Specify the number of detector columns which is used below
# Scale the number of angles and the detector pixel size with N
numCols = 512
numAngles = 4*int(360*numCols/1024)
pixelSize = 0.65*512/numCols
centerCol = 0.5*(numCols-1)

# Set the number of detector rows
numRows = numCols

# Reduce the number of columns so that the phantom extends past either the
# right or left side of the projections (but not both)
# Note that we translate the centerCol parameter so that only one side is truncated
numCols = 300
centerCol = 0.5*(numCols-1)+100

# Set the scanner geometry
#leapct.set_parallelbeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), centerCol, leapct.setAngleArray(numAngles, 360.0))
#leapct.set_fanbeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), centerCol, leapct.setAngleArray(numAngles, 360.0), 1100, 1400)
leapct.set_conebeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), centerCol, leapct.setAngleArray(numAngles, 360.0), 1100, 1400)
#leapct.set_coneparallel(numAngles, numRows, numCols, pixelSize, pixelSize*11.0/14.0, 0.5*(numRows-1), centerCol, leapct.setAngleArray(numAngles, 360.0), 1100, 1400)
#leapct.set_curvedDetector()

# Set the offsetScan flag to True
# This needs to be done after the geometry is set
# If you plan on setting the volume with the set_default_volume command, please set this flag first
# Setting this flag will automatically enlarge the circular field of view window
leapct.set_offsetScan(True)

# Set the volume parameters.
# It is best to do this after the CT geometry is set
leapct.set_default_volume()
#leapct.set_diameterFOV(leapct.get_voxelWidth()*leapct.get_numX())
#leapct.convert_to_modularbeam()

# Trouble-Shooting Functions
leapct.print_parameters()
#leapct.sketch_system(0)
#quit()

# Allocate space for the projections and the volume
g = leapct.allocateProjections()
f = leapct.allocateVolume()

# Specify simplified FORBILD head phantom
# One could easily do this in Python, but Python is soooooo slow for these types of operations,
# so we implemented this feature with multi-threaded C++
leapct.set_FORBILD(f,True)
#leapct.display(f)

#f = leapct.copy_to_device(f)
#g = leapct.copy_to_device(g)

# "Simulate" projection data
startTime = time.time()
leapct.project(g,f)
print('Forward Projection Elapsed Time: ' + str(time.time()-startTime))
#leapct.display(g)
#quit()

# Add noise to the data (just for demonstration purposes)
I_0 = 50000.0
#g[:] = -np.log(np.random.poisson(I_0*np.exp(-g))/I_0)


# Reconstruct the data
startTime = time.time()
leapct.FBP(g,f)
#leapct.backproject(g,f)
print('Reconstruction Elapsed Time: ' + str(time.time()-startTime))


# Post Reconstruction Smoothing (optional)
#startTime = time.time()
#leapct.diffuse(f,0.02/20.0,4)
#leapct.MedianFilter(f)
#leapct.BlurFilter(f,2.0)
#print('Post-Processing Elapsed Time: ' + str(time.time()-startTime))

# Display the result with napari
leapct.display(f)
