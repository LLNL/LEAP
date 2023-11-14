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
for which to set its parameters, for example use setConeBeamParams to set a cone-beam geometry
with certain specifications.

Then one may specify the reconstruction volume specifications such as the number of voxels in each
dimension and the voxel size.  We suggest using the "setDefaultVolume" function which sets the volume
parameters such that the volume fills the field of view of the CT system and uses the nominal voxel sizes.
Using voxel sizes that are significantly smaller and significantly bigger than this default size may result
in poor computational performance.
'''


# Specify the number of detector rows and columns which is used below
# Scale the number of angles and the detector pixel size with N
N = 1000
numAngles = int(720*N/1024)
pixelSize = 0.2*2048/N

# Set the scanner geometry
#leapct.setParallelBeamParams(1, N, N, pixelSize*11/14, pixelSize*11/14, 0.5*(N-1), 0.5*(N-1), 360.0)
leapct.setConeBeamParams(1, N, N, pixelSize, pixelSize, 0.5*(N-1), 0.5*(N-1), 360.0, 1100, 1400)

# Set the rotation angle (degrees) for the axis of symmetry
leapct.set_axisOfSymmetry(0.0)

# Set the volume parameters
leapct.setDefaultVolume()

leapct.printParameters()

# Allocate space for the projections and the volume
g = leapct.allocateProjections()
f = leapct.allocateVolume()

# Specify a phantom composed of a 300 mm diameter sphere
f_true = leapct.allocateVolume()
x,y,z=leapct.voxelSamples()
f_true[np.logical_and(y>=0.0, x**2 + y**2 + z**2 <= 150.0**2)] = 1.0
f_true[np.logical_and(y<0.0, x**2 + y**2 + z**2 <= 100.0**2)] = 1.0

# "Simulate" projection data
leapct.project(g,f_true)

# Reconstruct with FBP
leapct.FBP(g,f)

# Reconstruct with SART
#leapct.SART(g,f,50)
#leapct.RLS(g,f,50,0.0,0.0,True)

leapct.displayVolume(f)