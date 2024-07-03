import sys
import os
import time
import numpy as np
from leapctype import *
leapct = tomographicModels()
from leap_preprocessing_algorithms import *

'''
This demo script shows how LEAP can be used to model and deconvole detector blur.  Flat panel x-ray detectors
have a point spread function (psf) with very long tails.  The strength of this blur usually gets worse with higher energies.
If not correct for, this blur can cause reconstruction artifacts similar to those caused by beam hardening or scatter.
'''

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
# The example we use here is an analytic approximation to a typical detector psf
N_H1 = leapct.optimalFFTsize(2*numRows)
N_H2 = leapct.optimalFFTsize(2*numCols)
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

# Copy data to GPU
'''
# Comment this section out to revert back to multi-GPU solution
# with CPU-GPU data transfers to see when easy case is advantageous
device_name = "cuda:" + str(leapct.get_gpu())
device = torch.device(device_name)
g = torch.from_numpy(g).to(device)
f = torch.from_numpy(f).to(device)
#'''

# Apply Detector Blur
startTime = time.time()
g = leapct.transmission_filter(g, H, True)
print('Filtering Elapsed Time: ' + str(time.time()-startTime))

# Deconvolve Detector Blur
# We implement two different methods to choose
startTime = time.time()
detectorDeblur_FourierDeconv(leapct, g, H, WienerParam=0.0, isAttenuationData=True)
#detectorDeblur_RichardsonLucy(leapct, g, H, numIter=10, isAttenuationData=True)
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
