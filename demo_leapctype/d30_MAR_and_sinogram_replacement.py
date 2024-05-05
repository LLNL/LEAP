import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from leapctype import *
leapct = tomographicModels()
leapct.about()

'''
This script provides a demonstration of CT metal artifacts and so-called Metal Artifact Reduction (MAR) algorithms.

Metal artifacts are caused by metal or other highly attenuating materials being present in an object that is mostly composed
of lower attenuating materials.  In medical CT, there could be metal from implants or put there by some trauma (gun shot, car accident, etc.)
In luggage security, one wants to find explosives, but often luggage contains electronics which causes metal artifacts.

Metal object cause artifacts because of photon starvation (x-rays are completely blocked by the object and so no measurement can be made),
beam hardening, and scatter.  This script only simulates metal artifacts from photon starvation, and other effects could have been included.

Iterative reconstruction is a common method to deal with metal artifacts.  For example, one could use a Regularized Weighted Least Squares (RWLS)
cost function, but instead of specifying noise weights, one could specify weights based on whether measured rays traversed through metal.
You will see an example of this below.  Unfortunately these method sometimes washes out lower constast features or results in loss of object
edges.  It is actually very easy to remove metal artifacts.  The hard part is not creating secondary artifacts in the process.

The method we implemented in LEAP is referred to as "sinogram replacement" which is a method developed by Seemeen Karimi.  Here is the reference:
https://www.osti.gov/servlets/purl/1557944
Although this method is not as well-known as other MAR algorithm, it is much more robust and effective.  It is also a flexible tool that can be used for many applications.
We have used it to fuse x-ray and neutron CT data:
https://pubs.aip.org/aip/jap/article/132/15/154902/2837637
'''

# Specify the number of detector columns which is used below
# Scale the number of angles and the detector pixel size with N
numCols = 512
numAngles = 2*2*int(360*numCols/1024)
pixelSize = 0.65*512/numCols
numRows = 1

# Set the scanner geometry
# This script works for any geometry, but we'll just demonstrate with a cone-beam geometry
leapct.set_conebeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0), 1100, 1400)

# Set the volume parameters.
# It is best to do this after the CT geometry is set
leapct.set_default_volume()

# If you want to specify the volume yourself, use this function:
#leapct.set_volume(numX, numY, numZ, voxelWidth=None, voxelHeight=None, offsetX=None, offsetY=None, offsetZ=None):


# Allocate space for the projections and the volume
g = leapct.allocateProjections() # shape is numAngles, numRows, numCols
f = leapct.allocateVolume() # shape is numZ, numY, numX

# Specify simplified FORBILD head phantom with some additional pieces of metal
leapct.set_FORBILD()
leapct.addObject(None, 0, 10.0*np.array([-6.00000, 6.039200, 0.0]), 10.0*0.4*np.array([1.0, 1.0, 1.0]), 0.8106)
leapct.addObject(None, 0, 10.0*np.array([-6.00000, -6.039200, 0.0]), 10.0*0.4*np.array([1.0, 1.0, 1.0]), 1.2140)
leapct.addObject(None, 0, 10.0*np.array([6.40000, -6.039200, 0.0]), 10.0*0.05*np.array([1.0, 1.0, 1.0]), 0.8106)
leapct.addObject(None, 0, 10.0*np.array([5.80000, -6.4000, 0.0]), 10.0*0.1*np.array([1.0, 1.0, 1.0]), 0.8106)
leapct.addObject(None, 0, 10.0*np.array([6.30000, -5.6200, 0.0]), 10.0*0.05*np.array([1.0, 1.0, 1.0]), 0.8106)
leapct.addObject(None, 0, 10.0*np.array([5.90000, -6.039200, 0.0]), 10.0*0.05*np.array([1.0, 1.0, 1.0]), 0.8106)
leapct.addObject(None, 0, 10.0*np.array([6.00000, -5.739200, 0.0]), 10.0*0.1*np.array([1.0, 1.0, 1.0]), 0.8106)
leapct.addObject(None, 0, 10.0*np.array([6.50000, -6.5200, 0.0]), 10.0*0.1*np.array([1.0, 1.0, 1.0]), 0.8106)
leapct.addObject(None, 0, 10.0*np.array([5.70000, -5.29200, 0.0]), 10.0*0.05*np.array([1.0, 1.0, 1.0]), 0.8106)
leapct.addObject(None, 0, 10.0*np.array([5.60000, -5.69200, 0.0]), 10.0*0.05*np.array([1.0, 1.0, 1.0]), 0.8106)
leapct.addObject(None, 0, 10.0*np.array([5.40000, -6.2200, 0.0]), 10.0*0.05*np.array([1.0, 1.0, 1.0]), 0.8106)
leapct.addObject(None, 0, 10.0*np.array([5.20000, -6.4200, 0.0]), 10.0*0.05*np.array([1.0, 1.0, 1.0]), 0.8106)


# Simulate projection data
startTime = time.time()
leapct.rayTrace(g, oversampling=3)
print('Ray Tracing Elapsed Time: ' + str(time.time()-startTime))

# Add noise to the data and photon stravation to create metal artifacts
I_0 = 100000.0
t = I_0*np.exp(-g)
t = np.random.poisson(t)
t[t<=1.0] = 1.0
g[:] = -np.log(t/I_0)


# Reconstruct the data with FBP
# We blur the reconstruction a bit so that artifacts aren't too strong
leapct.FBP(g,f)
leapct.BlurFilter(f,2.0)
f_0 = f.copy()

# Segment out the metal pieces
# Here we segment using just some basic thresholding, but more sophisticated
# segmentation algorithms could be used in the step
metal = f.copy()
ind = metal>0.06
metal[~ind] = 0.0
metal[ind] = 1.0

# Forward project the metal mask to identify what parts of the data are
# corrupted by metal
W = g.copy()
leapct.project(W,metal)
metalTrace = W.copy()
ind_trace = metalTrace > 0.0
metalTrace[ind_trace] = 1.0
metalTrace[~ind_trace] = 0.0

# Now perform an RWLS reconstruction with metal weights
# This will create a reconstruction without metal artifacts, but does have
# some secondary artifacts
W = np.exp(-W)
f[:] = 0.0
filters = filterSequence(1.0e4)
filters.append(TV(leapct, delta=0.01/100.0, p=1.2))
leapct.RWLS(g,f,50,filters,W,'SQS')
leapct.RWLS(g,f,100,filters,W)
f_prior = f.copy()

# Now we create a "prior" sinogram by forward projecting the metal artifact-free reconstruction
# we will use this data to patch in the corrupted data
prior = g.copy()
leapct.project(prior, f)

# Now perform sinogram replacement on the original measured data
leapct.sinogram_replacement(g,prior,metalTrace)

# Now that the corrupted data has been corrected, we can just reconstruction it
# with any method we choose
leapct.FBP(g,f)

# The corrected data will not properly reconstruct the metal in the object, so we will
# simply just paste in the metal peices from the original reconstruction
f[ind] = f_0[ind]

plt.figure()
plt.subplot(1, 3, 1)
plt.title('FBP')
plt.imshow(np.squeeze(f_0), cmap='gray', vmin=0.0, vmax=0.04)
plt.subplot(1, 3, 2)
plt.title('RWLS (prior)')
plt.imshow(np.squeeze(f_prior), cmap='gray', vmin=0.0, vmax=0.04)
plt.subplot(1, 3, 3)
plt.title('MAR')
plt.imshow(np.squeeze(f), cmap='gray', vmin=0.0, vmax=0.04)
plt.show()
