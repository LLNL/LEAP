import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from leapctype import *

'''
!!! PLEASE READ THIS FIRST !!!

If all one is interested in is how to reconstruct curved detector cone-beam CT data, just see the first demo script: d01_standard_geometries.py
and uncomment the line that says: leapct.set_curvedDetector()

This script demonstrates a method to rebin an array of modules arranged on an arc to a continuous curved detector.

Most cone-beam systems with a curved detector are actually made up of a collection of modules that are arranged on an
arc whose radius is given by the source to detector distance (sdd).  In other words, each detector is normal to the
ray from the source to itself.  Making each detector module normal increases the x-ray flux and allows for anti-scatter
collimators to be put on the front face of the detector modules.  In airport security scanners this arc shape also
reduces the footprint size of the detector (instead of just one big flat panel detector) which is desirable.

Thus this script demonstrates how to use LEAP's rebin_curved function to rebin this type of data to a regular
curved detector which enables FBP reconstruction.  Of course, one could reconstruct the original data with iterative
reconstruction techniques, but FBP is not possible without this rebinning procedure.  In practice it might be best
to rebin the CT projections to perform an initial FBP reconstruction and then use the original data to refine this
FBP reconstruction with an iterative reconstruction method.
'''

# First we define the geometry
# The detector is composed of 32 modules or "stick" each with 32 rows and 16 columns each
# and each detector pixel is 0.875 mm X 0.875 mm
numCols_per_module = 16
numSticks = 32
numCols = numSticks*numCols_per_module
numRows = 32
numAngles = 720
pixelSize = 0.875 # mm

# Set source-to-object and source-to-detector distances
sod = 400.0
sdd = sod+300.0

leapct_regular = tomographicModels()

# Define the angular rotation samples
phis = leapct_regular.setAngleArray(numAngles, 360.0)

# Set the regular geometry as a curved cone-beam system
# Note that we are specifying detector positions that are perfectly symmetric, but one does not need to do this.
# This specifies the location of the interpolated samples and you may want to shift these to better align with the
# actual measured positions to reduce interpolation errors.  Also, if your original data was deliberately shift,
# e.g., like a offset-scan (also know as a half-cone), you will want to shift these as well.
leapct_regular.set_conebeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), phis, sod, sdd)
leapct_regular.set_curvedDetector()
leapct_regular.set_volume(numCols, numCols, numRows, pixelSize*sod/sdd, pixelSize*sod/sdd)
g = leapct_regular.allocate_projections()


######################################################################################################################################
######################################################################################################################################
### This section of the code performs the simulation of the data
######################################################################################################################################
######################################################################################################################################

# Make another tomographicModels object to track the true geometry
leapct_true = tomographicModels()

# Define the angular spacing between detector modules
# The angularSpacing array below defines the angular position
# of each detector module which is arranged on an arc
# The first definition of angularSpacing defines a "perfect" detector where
# each module is placed end-to-end.  In a real system this it is not possible
# to do this because the electronics and detector housing prevent it
# Thus we define a second definition of angularSpacing which has an additional
# 1 mm gap between modules.  We could also define a variable gap between detectors
# and as long as the rebinning algorithm is aware of these gaps, the code will work.
#angularSpacing = np.arctan(pixelSize*numCols_per_module / sdd)
angularSpacing = np.arctan((pixelSize*numCols_per_module + 1.0) / sdd)

# Define the angles of each module
alphas = (np.array(range(numSticks))-(numSticks-1)/2.0)*angularSpacing

# Now we define the CT geometry of the "true" system
# We use LEAP's modular-beam geometry to place each detector module
# We will do this for each projection angle individually.  Note that we
# specify numAngles=numSticks because each projection in the modular-beam geometry is one source and detector pair 
leapct_true.set_volume(numCols, numCols, numRows, pixelSize*sod/sdd, pixelSize*sod/sdd)
f = leapct_true.allocateVolume() # shape is numZ, numY, numX
leapct_true.set_FORBILD(f,True)

for n in range(numAngles):
    phi = phis[n]*np.pi/180.0 - np.pi/2.0
    sourcePositions = np.zeros((numSticks,3),dtype=np.float32)
    sourcePositions[:,0] = sod*np.cos(phi)
    sourcePositions[:,1] = sod*np.sin(phi)
    moduleCenters = np.zeros((numSticks,3),dtype=np.float32)
    colVectors = np.zeros((numSticks,3), dtype=np.float32)
    rowVectors = np.zeros((numSticks,3), dtype=np.float32)
    rowVectors[:,2] = 1.0
    for m in range(numSticks):
        colVectors[m,0] = -np.sin(phi-alphas[m])
        colVectors[m,1] = np.cos(phi-alphas[m])
        moduleCenters[m,0] = -sdd*np.cos(phi-alphas[m])+sourcePositions[m,0]
        moduleCenters[m,1] = -sdd*np.sin(phi-alphas[m])+sourcePositions[m,1]
    
    leapct_true.set_modularbeam(numSticks, numRows, numCols_per_module, pixelSize, pixelSize, sourcePositions, moduleCenters, rowVectors, colVectors)
    g_sim = leapct_true.allocateProjections() # shape is numAngles, numRows, numCols
    
    #leapct_true.sketch_system()
    #quit()
    
    # "Simulate" the data
    leapct_true.project(g_sim,f)

    # Concatenate the projections onto each modular into one projection
    for m in range(numSticks):
        g[n,:,m*numCols_per_module:(m+1)*numCols_per_module] = g_sim[m,:,:]


######################################################################################################################################
######################################################################################################################################
### This section of the code the rebinning and FBP reconstruction
######################################################################################################################################
######################################################################################################################################

# To enable the rebinning, we must specify the angle (in degrees) of each detector pixel along a single row
fanAngles = np.array(range(numCols),dtype=np.float32)
fanAngles_per_module = (np.array(range(numCols_per_module))-(numCols_per_module-1)/2.0)*np.arctan(pixelSize/sdd)
for m in range(numSticks):
    fanAngles[m*numCols_per_module:(m+1)*numCols_per_module] = alphas[m]+fanAngles_per_module
fanAngles *= 180.0/np.pi

# Now perform the rebinning.  The "order" argument specifies the order of the interpolating polynomial.
# Bigger numbers will produce a sharper interpolation and thus a sharper reconstruction.
# Try commenting out this line to see the reconstruction artifacts that appear.
startTime = time.time()
leapct_regular.rebin_curved(g, fanAngles, order=6)
print('Rebinning Elapsed Time: ' + str(time.time()-startTime) + ' seconds')

# Finally, perform FBP reconstruction and display the result
f = leapct_regular.FBP(g)
f[f<0.0] = 0.0
plt.imshow(np.squeeze(f[f.shape[0]//2,:,:]), cmap=plt.get_cmap('gray'))
plt.show()
