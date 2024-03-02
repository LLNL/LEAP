import sys
import os
import time
import numpy as np
from leapctype import *
leapct = tomographicModels()
sys.path.append(r'..\utils')
from generateDictionary import *

'''
This sample script demonstrates how one can perform denoising with the bilateral filter (BLF).
'''


# Specify the number of detector columns which is used below
# Scale the number of angles and the detector pixel size with N
numCols = 512
numAngles = 2*2*int(360*numCols/1024)
pixelSize = 0.65*512/numCols

# Set the number of detector rows
#numRows = numCols
numRows = 8

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
I_0 = 10000.0
t = np.random.poisson(I_0*np.exp(-g))
t[t<1.0] = 1.0
g[:] = -np.log(t/I_0)

# Reset the volume array to zero, otherwise iterative reconstruction algorithm will start their iterations
# with the true result which is cheating
f[:] = 0.0

# Reconstruct the data
startTime = time.time()
leapct.FBP(g,f)
print('Reconstruction Elapsed Time: ' + str(time.time()-startTime))


# Run Bilateral Filter (BLF)
startTime = time.time()
f_0 = f.copy()

# This is the standard bilateral filter (BLF)
# Here we use a spatial window size of 4 voxels and
# an intensity window of 0.02.
# A larger spatial window size uses more voxels to perform the averaging
# so the denoising is stronger, but it takes longer to perform.
# A larger intensity window will blur the result more, but an intensity window that is
# too small won't do much denoising
leapct.BilateralFilter(f, 4, 0.02)

# One can also perform a Scaled BLF
# Here the volume used to calculate the intensity window weight is the original
# volume blurred by a low pass filter of FWHM 2 voxels
# This method usually works better than the standard BLF especially when
# the original image has very high noise, but uses 50% more memory
#leapct.BilateralFilter(f, 4, 0.02, 2.0)
print('Post-Processing Elapsed Time: ' + str(time.time()-startTime))


# Display the result
f_0_slice = np.squeeze(f_0[f.shape[0]//2,:,:])
f_slice = np.squeeze(f[f.shape[0]//2,:,:])
I = np.concatenate((f_0_slice, f_slice),axis=1)
I[I<0.0] = 0.0
I[I>0.04] = 0.04
import matplotlib.pyplot as plt
plt.imshow(I, cmap='gray')
plt.show()
