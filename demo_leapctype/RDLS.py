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
numAngles = 2*2*int(360*numCols/1024)
pixelSize = 0.65*512/numCols

# Set the number of detector rows
numRows = numCols*0+1

# Set the scanner geometry
#leapct.set_parallelbeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0))
#leapct.set_fanbeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0), 1100, 1400)
leapct.set_conebeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0), 1100, 1400)
#leapct.set_curvedDetector()

# Set the volume parameters.
# It is best to do this after the CT geometry is set
leapct.set_default_volume()

# If you want to specify the volume yourself, use this function:
#leapct.set_volume(numX, numY, numZ, voxelWidth=None, voxelHeight=None, offsetX=None, offsetY=None, offsetZ=None):

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

# Reset the volume array to zero, otherwise iterative reconstruction algorithm will start their iterations
# with the true result which is cheating
f[:] = 0.0

# Reconstruct the data
startTime = time.time()

numIter = 10
leapct.LS(g,f,1)
leapct.RDLS(g,f,numIter,2.0)
'''
Pf = g.copy()
Pf[:] = 0.0
LPf_minus_g = Pf.copy()
grad = f.copy()
d = f.copy()
Pd = Pf.copy()
LPd = Pf.copy()
for n in range(numIter):
    # Calculate gradient
    LPf_minus_g[:] = Pf[:]-g[:]
    leapct.Laplacian(LPf_minus_g)
    leapct.backproject(LPf_minus_g, grad)
    d[:] = grad[:]
    leapct.BlurFilter2D(d,2.0)

    # Calculate step size
    leapct.project(Pd,d)
    LPd[:] = Pd[:]
    leapct.Laplacian(LPd)
    alpha = np.sum(d*grad) / np.sum(Pd*LPd)
    print('alpha = ' + str(alpha))
    f[:] = f[:] - alpha*d[:]
    Pf[:] = Pf[:] - alpha*Pd[:]
#'''
print('Reconstruction Elapsed Time: ' + str(time.time()-startTime))



# Display the result with napari
leapct.display(f)
