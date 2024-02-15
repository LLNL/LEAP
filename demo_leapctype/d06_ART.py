import sys
import os
import time
import numpy as np
from leapctype import *
leapct = tomographicModels()
# Make sure you add: .../LEAP/src to your python path

'''
All memory for data structures, e.g., the projection data and the volume data is managed in python.
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
numAngles = 4*int(360*numCols/1024)
pixelSize = 0.65*512/numCols

# Set the number of detector rows
numRows = 1

# Set the scanner geometry
leapct.set_parallelbeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0))

# Set the volume parameters.
# It is best to do this after the CT geometry is set
leapct.set_default_volume()

# Trouble-Shooting Functions
leapct.print_parameters()
#leapct.sketch_system()

# Allocate space for the projections and the volume
g = leapct.allocateProjections()
f = leapct.allocateVolume()
mu = leapct.allocateVolume()

# Specify the attenuation map parameters
# In this demo, one can specify a voxelized attenuation or a cylindrical attenuation volume
muCoeff = 0.01
muRadius = 150.0

# Specify a phantom composed of a 300 mm diameter sphere
leapct.addObject(f, 4, np.array([0.0, 0.0, 0.0]), np.array([muRadius, muRadius, muRadius]), 1.0)
leapct.addObject(f, 0, np.array([0.0, 20.0, 0.0]), np.array([20.0, 20.0, 20.0]), 10.0)

# Specify a voxelized attenuation volume
leapct.addObject(mu, 4, np.array([0.0, 0.0, 0.0]), np.array([muRadius, muRadius, muRadius]), muCoeff)
leapct.addObject(mu, 0, np.array([0.0, -30.0, 0.0]), np.array([20.0, 20.0, 20.0]), 0.0)
leapct.BlurFilter(mu, 2.0)

# Here is whether you choose the voxelized or cylindrical attenuation map
leapct.set_attenuationMap(mu)
#leapct.set_cylindircalAttenuationMap(muCoeff, muRadius)

# "Simulate" projection data
startTime = time.time()
leapct.project(g,f)
print('Forward Projection Elapsed Time: ' + str(time.time()-startTime))
#leapct.display(g)

# Add noise to the data (just for demonstration purposes)
#g[:] = np.random.poisson(g)

# Reset the volume array to zero, otherwise iterative reconstruction algorithm will start their iterations
# with the true result which is cheating
f[:] = 0.0

# Reconstruct the data
startTime = time.time()
#leapct.backproject(g,f)
leapct.FBP(g,f)
#leapct.ASDPOCS(g,f,10,5,4,1.0/20.0)
#leapct.SART(g,f,10,10)
#leapct.MLEM(g,f,5,1)
#leapct.LS(g,f,100,True)
print('Reconstruction Elapsed Time: ' + str(time.time()-startTime))


# Post Reconstruction Smoothing (optional)
#startTime = time.time()
#leapct.diffuse(f,0.02/20.0,4)
#leapct.MedianFilter(f)
#leapct.BlurFilter(f,2.0)
#print('Post-Processing Elapsed Time: ' + str(time.time()-startTime))

# Display the result with napari
leapct.display(f)
