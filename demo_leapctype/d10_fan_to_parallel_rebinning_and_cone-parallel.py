import sys
import os
import time
import numpy as np
from leapctype import *
leapct = tomographicModels()
leapct.about()

'''
This script demonstrates how to rebin fan-beam to parallel-beam or cone-beam to cone-parallel coordinates.
Cone-parallel coordinates is the standard coordinate system used in medical CT.

Note: if your data is already in cone-parallel coordinates, do NOT use the rebin command, just set the
geometry as cone-parallel with the: set_coneparallel command.

Currently, the rebinning routines are only implemented on the CPU.  If you require the rebinning to
take place on the GPU please submit a feature request.
'''


# Specify the number of detector columns which is used below
# Scale the number of angles and the detector pixel size with N
numCols = 512
numAngles = 2*2*int(360*numCols/1024)
pixelSize = 0.65*512/numCols

# Set the number of detector rows
numRows = numCols

# Set the scanner geometry
# This script can be run for fan- or cone-beam geometries
# Try commenting out different geometry types below to run this sample script for different geometries
# The line below that says: leapct.set_curvedDetector() only applies to cone-beam data and sets the detector to have a shape that is the surface of a cylinder
# whose diameter is given by the source-to-detector distance
#leapct.set_fanbeam(numAngles, numRows, numCols, pixelSize*11.0/14.0, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0), 1100, 1400)
leapct.set_conebeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0), 1100, 1400)
#leapct.set_curvedDetector()


# Set the volume parameters.
# It is best to do this after the CT geometry is set
leapct.set_default_volume()

# If you want to specify the volume yourself, use this function:
#leapct.set_volume(numX, numY, numZ, voxelWidth=None, voxelHeight=None, offsetX=None, offsetY=None, offsetZ=None):

# By default, LEAP applies a circular field of view mask to the CT volume.  The diameter of this mask is determined by the 
# CT geometry parameters, but this may be overriden (mask can be made smaller, bigger, or completely removed) by using the
# following function where d is the diameter of the field of view measured in mm
#leapct.set_diameterFOV(d)

# Trouble-Shooting Functions
leapct.print_parameters()
#leapct.sketch_system()

# Allocate space for the projections and the volume
# You don't have to use these functions; they are provided just for convenience
# All you need is for the data to be C contiguous float32 arrays with the right dimensions
g = leapct.allocateProjections() # shape is numAngles, numRows, numCols
f = leapct.allocateVolume() # shape is numZ, numY, numX

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

# Now we rebin the data
# Note that the rebin_parallel function may take an argument that specifies the order of the interpolating
# polynomial which can be between 2 and 6.  Higher orders perform more high-resolution interpolation, but any
# values larger than 2 may result is under/over shoots including possible negative values in the projection data
startTime = time.time()
#leapct.set_log_debug()
leapct.rebin_parallel(g,order=6)
print('Rebinning Elapsed Time: ' + str(time.time()-startTime))
#leapct.set_log_status()
leapct.print_parameters()
leapct.display(g)
#quit()

# Reset the volume array to zero, otherwise iterative reconstruction algorithm will start their iterations
# with the true result which is cheating
f[:] = 0.0

# Reconstruct the data
# Below you will see many different reconstruction algorithms that LEAP has available, but are commented out
# In these examples we provide some but not all of the possible reconstruction algorithm parameters, but please
# check the comments for each function in leapctype.py for a description of their arguments
# Also note that algorithm can be run in series.  For example, if you first perform an FBP reconstruction
# and then an iterative reconstruction, like RWLS, then the iterative reconstruction will start with the FBP reconstruction
# this trick can be used to accelerate an iterative reconstruction algorithm
# If you want an iterative reconstruction to start from scratch, just initialize it with zeros
startTime = time.time()
#leapct.backproject(g,f)
leapct.FBP(g,f)
#leapct.inconsistencyReconstruction(g,f)
#leapct.print_cost = True
filters = filterSequence(1.0e0)
filters.append(TV(leapct, delta=0.02/20.0))
#leapct.ASDPOCS(g,f,10,10,1,filters)
#leapct.SART(g,f,10,10)
#leapct.OSEM(g,f,10,10)
#leapct.LS(g,f,50,'SQS')
#leapct.RWLS(g,f,50,filters,None,'SQS')
#leapct.RDLS(g,f,50,filters,1.0,True,1)
#leapct.MLTR(g,f,10,10,filters)
print('Reconstruction Elapsed Time: ' + str(time.time()-startTime))


# Post Reconstruction Smoothing (optional)
# Here are some optional post reconstruction noise filters that can be applied
# Try uncommenting out these lines to test how they work
#startTime = time.time()
#leapct.diffuse(f,0.02/20.0,4)
#leapct.MedianFilter(f)
#leapct.BlurFilter(f,2.0)
#print('Post-Processing Elapsed Time: ' + str(time.time()-startTime))

# Display the result with napari
leapct.display(f)
#import matplotlib.pyplot as plt
#plt.imshow(np.squeeze(f[f.shape[0]//2,:,:]), cmap='gray')
#plt.show()
