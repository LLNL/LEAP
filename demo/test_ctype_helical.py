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
N = 300
numTurns = 3
numAngles = int(720*N/1024)*numTurns
pixelSize = 0.2*2048/N
M = N

# Set the scanner geometry
#leapct.set_parallelBeam(numAngles, N, N, pixelSize, pixelSize, 0.5*(N-1), 0.5*(N-1), leapct.setAngleArray(numAngles, 360.0))
#leapct.set_fanBeam(numAngles, N, N, pixelSize, pixelSize, 0.5*(N-1), 0.5*(N-1), leapct.setAngleArray(numAngles, 360.0), 1100, 1400)
leapct.set_conebeam(numAngles, M, N, pixelSize, pixelSize, 0.5*(M-1), 0.5*(N-1), leapct.setAngleArray(numAngles, 360.0*numTurns), 1100, 1400)
#leapct.set_normalizedHelicalPitch(0.2)
leapct.set_normalizedHelicalPitch(1.0)

# Set the volume parameters
leapct.set_default_volume()

# Trouble-Shooting Functions
leapct.print_parameters()
#leapct.sketch_system()
#quit()

# Allocate space for the projections and the volume
g = leapct.allocateProjections()
f = leapct.allocateVolume()

# Specify a phantom composed of a 300 mm diameter sphere
f = leapct.allocateVolume()
x,y,z=leapct.voxelSamples()
f[x**2 + y**2 + (z/1.05)**2 <= 150.0**2] = 1.0
f[x**2 + (y-20)**2 + z**2 <= 10.0**2] = 0.0
#leapct.displayVolume(f)

# "Simulate" projection data
leapct.project(g,f)
f[:] = 0.0
g[g<0.0] = 0.0
g[:] = np.random.poisson(g)

# Reconstruction
startTime = time.time()
#leapct.backproject(g,f)
leapct.ASDPOCS(g,f,20,5,3,1.0/20.0)
#leapct.diffuse(f,1.0/20.0,3)
#leapct.SART(g,f,10,5)
#leapct.MLEM(g,f,5,1)
#leapct.LS(g,f,10)
print('Elapsed time: ' + str(time.time()-startTime))

leapct.displayVolume(f)
