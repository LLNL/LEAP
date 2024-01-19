import sys
import os
import time
import numpy as np
from leapctype import *
leapct = tomographicModels()
sys.path.append(r'..\utils')
from preprocessing_algorithms import *

# Specify the number of detector columns which is used below
# Scale the number of angles and the detector pixel size with N
numCols = 512
numAngles = 2*2*int(360*numCols/1024)
pixelSize = 0.65*512/numCols

# Set the number of detector rows
numRows = numCols


# Set the scanner geometry
#leapct.set_parallelbeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0))
#leapct.set_fanbeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0), 1100, 1400)
leapct.set_conebeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0), 1100, 1400)
#leapct.set_curvedDetector()

# Set the volume parameters.
# It is best to do this after the CT geometry is set
leapct.set_default_volume()

# Set Blur Kernel
N_H1 = optimalFFTsize(2*numRows)
N_H2 = optimalFFTsize(2*numCols)
y = (np.array(range(N_H1), dtype=np.float32)-N_H1//2)
y = y / np.max(y)
x = (np.array(range(N_H2), dtype=np.float32)-N_H2//2)
x = x / np.max(x)
y,x = np.meshgrid(y,x, indexing='ij')

c = 9.0
b = (c+1.0-0.5*(1.0+c))/c

H = (1.0-b)+b/(1.0+c*np.sqrt(x**2+y**2))
H = np.fft.fftshift(H)

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

# Apply Detector Blur
startTime = time.time()
g = leapct.transmission_filter(g, H, True)
print('Filtering Elapsed Time: ' + str(time.time()-startTime))

# Deconvolve Detector Blur
startTime = time.time()
#g = detectorDeblur_FourierDeconv(leapct, g, H, isAttenuationData=True, WienerParam=0.0)
g = detectorDeblur_RichardsonLucy(leapct, g, H, isAttenuationData=True, numIter=10)
print('Deblur Elapsed Time: ' + str(time.time()-startTime))

# Reset the volume array to zero, otherwise iterative reconstruction algorithm will start their iterations
# with the true result which is cheating
f[:] = 0.0

# Reconstruct the data
startTime = time.time()
leapct.FBP(g,f)
print('Reconstruction Elapsed Time: ' + str(time.time()-startTime))


# Display the result with napari
leapct.display(f)
