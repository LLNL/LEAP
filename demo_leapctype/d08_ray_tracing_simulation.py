import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from leapctype import *
leapct = tomographicModels()

'''
Note that this script is nearly identical to d01_standard_geometries.py, except how the CT data is simulated.

This script demonstrates how to simulate CT data using the analytic ray tracing algorithms in LEAP.
Analytic ray tracing calculates the length of intersection of a collection of 3D geometric objects (ellipsoids, cylinders, parallelepipeds, etc.).
Simulating CT data by forward projecting a voxelized phantom is known as an "inverse crime" because one uses the same models to simulation and
reconstruct the data.  This mostly provides that an algorithm is self-consistent and can give unrealistic results due to overfitting.

Thus performing simulations using analytic ray tracing methods results in a better assessment of various CT reconstruction methods and is a better
method to debug and improve reconstruction methods.
In addition, one may employ ray oversampling in the analytic ray tracing methods to model the non-linear partial volume effect.  Thus is done as follows:
g = -log( (sum_{n=1}^N exp(-rayTrace sub ray n)) / N  )
i.e., the averaging is done in transmission space which is a more accurate way to model real CT data
'''


# Specify the number of detector columns which is used below
# Scale the number of angles and the detector pixel size with N
numCols = 512
numAngles = 2*2*int(360*numCols/1024)
pixelSize = 0.65*512/numCols

# Set the number of detector rows
numRows = numCols

# Set the scanner geometry
# This script can be run for any of the LEAP CT geometries
#leapct.set_parallelbeam(numAngles=numAngles, numRows=numRows, numCols=numCols, pixelHeight=pixelSize*1100.0/1400.0, pixelWidth=pixelSize*1100.0/1400.0, centerRow=0.5*(numRows-1), centerCol=0.5*(numCols-1), phis=leapct.setAngleArray(numAngles, 360.0))
#leapct.set_fanbeam(numAngles, numRows, numCols, pixelSize*1100.0/1400.0, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0), 1100, 1400)
leapct.set_conebeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0), 1100, 1400)
#leapct.set_coneparallel(numAngles, numRows, numCols, pixelSize, pixelSize*1100.0/1400.0, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0), 1100, 1400)
#leapct.set_curvedDetector()
#leapct.set_normalizedHelicalPitch(0.5)
#leapct.convert_to_modularbeam()

# Set the volume parameters.
# It is best to do this after the CT geometry is set
leapct.set_default_volume()

# If you want to specify the volume yourself, use this function:
#leapct.set_volume(numX, numY, numZ, voxelWidth=None, voxelHeight=None, offsetX=None, offsetY=None, offsetZ=None):

# Trouble-Shooting Functions
leapct.print_parameters()
#leapct.sketch_system()

# Allocate space for the projections and the volume
# You don't have to use these functions; they are provided just for convenience
# All you need is for the data to be C contiguous float32 arrays with the right dimensions
g = leapct.allocateProjections() # shape is numAngles, numRows, numCols
f = leapct.allocateVolume() # shape is numZ, numY, numX

# Specify the FORBILD head phantom
# For more information on how to specify phantoms, see https://leapct.readthedocs.io/en/latest/ctsimulation.html
leapct.set_FORBILD()
#leapct.addObject(None, 4, np.array([0.0, 0.0, 0.0]), 120.0*np.array([1.0, 1.0, 1.0]), 0.02, None, None, 1)


# Simulate projection data using analytic ray tracing methods
startTime = time.time()
leapct.rayTrace(g)
print('Ray Tracing Simulation Elapsed Time: ' + str(time.time()-startTime))
#leapct.display(g)
#quit()

# Display the central sinogram
#plt.imshow(np.squeeze(g[:,g.shape[1]//2,:]), cmap=plt.get_cmap('gray'))
#plt.show()

# Add noise to the data (just for demonstration purposes)
I_0 = 50000.0
#g[:] = -np.log(np.random.poisson(I_0*np.exp(-g))/I_0)


# Reconstruct the data
startTime = time.time()
leapct.FBP(g,f)
print('Reconstruction Elapsed Time: ' + str(time.time()-startTime))

# Display the central slice of the result
leapct.display(f)
#plt.imshow(np.squeeze(f[f.shape[0]//2,:,:]), cmap=plt.get_cmap('gray'))
#plt.show()
