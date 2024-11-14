import sys
import os
import time
import numpy as np
from leapctype import *
leapct = tomographicModels()
leapct.about()

'''
This demo script simulates data using a forward projection of the FORBILD head phantom and then performs
a reconstruction.  This same script can be used to test these features for parallel-, fan-, or cone-beam geometries

All memory for data structures, e.g., the projection data and the volume data are managed in python.
LEAP only tracks the specifications, i.e., geometry of the CT model, the volume parameters,
and a few other parameters that deal with how the code should be run, such as which GPUs to use.
These parameters exist in the C code and are set by python functions in the python class "tomographicModels".
Once these are set, one simply provides the numpy arrays of the projection data and volume data and
LEAP will perform the various operations.

Each of the four geometry types: parallel-, fan-, cone-, and modular-beam has its own function
for which to set its parameters, for example use set_conebeam to set a cone-beam geometry
with certain specifications.

Then one may specify the reconstruction volume specifications such as the number of voxels in each
dimension and the voxel size.  We suggest using the "set_default_volume" function which sets the volume
parameters such that the volume fills the field of view of the CT system and uses the nominal voxel sizes.
Using voxel sizes that are significantly smaller or significantly bigger than this default size may result
in poor computational performance.
'''


# Specify the number of detector columns which is used below
# Scale the number of angles and the detector pixel size with N
numCols = 512
numAngles = 2*2*int(360*numCols/1024)
pixelSize = 0.65*512/numCols

# Set the number of detector rows
numRows = numCols

# Set the scanner geometry
# This script can be run for parallel-, fan-, or cone-beam geometries
# Try commenting out different geometry types below to run this sample script for different geometries
# Note that the arguments for each of the three standard geometries are nearly equivalent, except set_fanbeam and set_conebeam have some extra parameters
# that specify the source-to-object distance (sod), the source-to-detector distance (sdd), and tau the horizontal translation of the source position
# set_conebeam also allows users to specify the helicalPitch if they wish but the default for this parameter is zero
# The line below that says: leapct.set_curvedDetector() only applies to cone-beam data and sets the detector to have a shape that is the surface of a cylinder
# whose diameter is given by the source-to-detector distance
#leapct.set_parallelbeam(numAngles=numAngles, numRows=numRows, numCols=numCols, pixelHeight=pixelSize, pixelWidth=pixelSize, centerRow=0.5*(numRows-1), centerCol=0.5*(numCols-1), phis=leapct.setAngleArray(numAngles, 360.0))
#leapct.set_fanbeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0), 1100, 1400)
leapct.set_conebeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0), 1100, 1400)
#leapct.set_coneparallel(numAngles, numRows, numCols, pixelSize, pixelSize*1100.0/1400.0, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0), 1100, 1400)
#leapct.set_curvedDetector()
#leapct.convert_to_modularbeam()
#leapct.rotate_detector(1.0)

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

# Set the backprojector model, 'SF' (the default setting), is more accurate, but 'VD' is faster
#leapct.set_projector('VD')

# Allocate space for the projections and the volume
# You don't have to use these functions; they are provided just for convenience
# All you need is for the data to be C contiguous float32 arrays with the right dimensions
g = leapct.allocate_projections() # shape is numAngles, numRows, numCols
f = leapct.allocate_volume() # shape is numZ, numY, numX

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
filters = filterSequence(1.0e0) # filter strength argument must be turned to your specific application
filters.append(TV(leapct, delta=0.02/20.0)) # the delta argument must be turned to your specific application
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
