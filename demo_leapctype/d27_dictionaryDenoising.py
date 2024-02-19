import sys
import os
import time
import numpy as np
from leapctype import *
leapct = tomographicModels()
sys.path.append(r'..\utils')
from generateRidgeletDictionary import *

'''
This sample script demonstrates how one can perform dictionary denoising.
'''


# Specify the number of detector columns which is used below
# Scale the number of angles and the detector pixel size with N
numCols = 512//4
numAngles = 2*2*int(360*numCols/1024)
pixelSize = 0.65*512/numCols

# Set the number of detector rows
numRows = 1

# Set the scanner geometry
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

# Add noise to the data (just for demonstration purposes)
I_0 = 50000.0
g[:] = -np.log(np.random.poisson(I_0*np.exp(-g))/I_0)

# Reset the volume array to zero, otherwise iterative reconstruction algorithm will start their iterations
# with the true result which is cheating
f[:] = 0.0

# Reconstruct the data
startTime = time.time()
leapct.FBP(g,f)
print('Reconstruction Elapsed Time: ' + str(time.time()-startTime))

D = generateRidgeletDictionary(4)
print(D.shape)

# Post Reconstruction Smoothing (optional)
startTime = time.time()
leapct.DictionaryDenoising(f, D, 20, 0.01)
print('Post-Processing Elapsed Time: ' + str(time.time()-startTime))

f[f<0.0] = 0.0
f[f>1.0] = 1.0

# Display the result with napari
leapct.display(f)
