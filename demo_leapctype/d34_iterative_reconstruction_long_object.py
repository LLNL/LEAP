import sys
import os
import time
import numpy as np
from leapctype import *
leapct = tomographicModels()
leapct.about()

'''
This script demonstrates a method to perform iterative reconstruction of a long object, i.e., an object that extends past
the top and/or bottom of the detector which is very common is medical imaging.  If doing parallel- or fan-beam this does
not cause a problem.  For cone-beam, the diverging rays are only seen by certain projection angles.  FBP has no problem
reconstructing a reduced region of interest (ROI), but iterative reconstruction requires that one include every region/ every slice
that a ray passes through to be included in the reconstruction.  This forces one to include slices that extend past the
standard field of view to get artifact-free reconstructions.

There is a method called the "Ziegler method" (see reference below) that outlines a simple, yet effective method to
efficiently deal with the problem of reconstructing only a region of iterest with iterative reconstruction.  The basic
idea is that one first performs an FBP reconstruction of the entire region that the x-rays pass through, then one removes
(either sets to zero or removes slices) the region they wish to reconstruction, forward project these unwanted regions,
and subtract them from the measured data.  Then one may perform ROI reconstruction using this new data.

Slices near the edge of the field of view won't be reconstruction perfectly with FBP, but that does not matter for this method to work.
Even though tomographic reconstruction is a global problem, only the low frequency compoments are global in nature.
Thus even through the initial FBP reconstruct may not be perfect and may be very noisey, it is only required that it accurately
represents the low frequency components of the image.

In addition to demonstrating this Ziegler method, this script also demonstrates that the LEAP (non-helical) FBP reconstruction algorithms
employ zeroth order extrapolation off the top and bottom of the detector.  This is an imporant feature for properly reconstructing
slices near the edge of the field of view and gives reasonable results beyond the field of view- at least reasonable enough to
be effective for this application.

Ziegler, Andy, Tim Nielsen, and Michael Grass.
"Iterative reconstruction of a region of interest for transmission tomography."
Medical physics 35, no. 4 (2008): 1317-1327.
'''


# Similar to our other sample script, we first specify the scanner geometry
numCols = 512
numAngles = 2*2*int(360*numCols/1024)
pixelSize = 0.65*512/numCols
numRows = numCols//2
leapct.set_conebeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0), 1100, 1400)
#leapct.set_curvedDetector()

# Now we set the volume parameters, but we make sure to specfy extra volume slices
# because we want to simulate a long object that extends past the field of view
extraSlices = 128
leapct.set_default_volume()
leapct.set_numZ(leapct.get_numZ()+extraSlices)

# Allocate space for the projections and the volume
g = leapct.allocate_projections() # shape is numAngles, numRows, numCols
f = leapct.allocate_volume() # shape is numZ, numY, numX

# Specify FORBILD head phantom
leapct.set_FORBILD(f,True)

# "Simulate" projection data
leapct.project(g,f)

# Now we remove those extra slices and go back to just the slices that we are interested in reconstructing
leapct.set_numZ(leapct.get_numZ()-extraSlices)
f = leapct.allocate_volume()


#"""
# This code between the multi-line comments performs the Ziegler method to the long object problem.
# If you comment this section out (by removing the # sign before the triple quote, above)
# the Ziegler method will not be applied and one will see strong artifacts at the edge slices
# of coronal or sagittal views, but including this section of the code shoud mitigate these edge artifacts.

# First get the current z-slice locations
zs = leapct.z_samples()

# Now we ask LEAP to tell us all of the z-slices (including those "hypothetical z-slices" that extend past the current volume specification)
# that are effected by the measured projections.  The "False" argument tells LEAP to include slices outside the current volume specification.
zSliceRange = leapct.sliceRangeNeededForProjection(False)

# Now we record the locations of these z-slices at the bottom and top of the current volume specification
z_bottom = np.array(range(zSliceRange[0],0))*leapct.get_voxelHeight() + zs[0]
z_top = np.array(range(1,zSliceRange[1]+2-leapct.get_numZ()))*leapct.get_voxelHeight() + zs[-1]

# Allocate temporary projection data
Pf = leapct.allocate_projections()

# Make a copy of the current parameters to use for this method, so that we don't disturb the original parameters
leapct_caps = tomographicModels()
leapct_caps.copy_parameters(leapct)

# Set the volume for the bottom slices, forward project it and substract off the measured data
leapct_caps.set_numZ(int(z_bottom.size))
leapct_caps.set_offsetZ(np.mean(z_bottom))
f_bottom = leapct_caps.allocate_volume()
leapct_caps.FBP(g,f_bottom)
leapct_caps.project(Pf,f_bottom)
g[:] = g[:] - Pf[:]

# Set the volume for the top slices, forward project it and substract off the measured data
leapct_caps.set_numZ(int(z_top.size))
leapct_caps.set_offsetZ(np.mean(z_top))
f_top = leapct_caps.allocate_volume()
leapct_caps.FBP(g,f_top)
leapct_caps.project(Pf,f_top)
g[:] = g[:] - Pf[:]
#"""

# Iterative Reconstruction Step (any iterative algorithm will do; we just demonstrate with the simplest algorithn, LS)
leapct.LS(g,f,50,preconditioner='SQS')


# Display a sagittal slice
import matplotlib.pyplot as plt
plt.imshow(np.squeeze(f[:,:,f.shape[2]//2]), cmap='gray')
#plt.imshow(I, cmap='gray')
plt.show()
