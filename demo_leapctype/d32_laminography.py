import sys
import os
import time
import numpy as np
from leapctype import *
leapct = tomographicModels()
leapct.about()

'''
This demo script simulates and reconstructs cone-beam laminography data.

Note that in LEAP the sources and detectors move-not the object.  Thus one will need to specify the source and detector positions
from the point of view of a fixed object.

For this we will use the modular-beam geometry.  One could model a limited number of cases with the cone-beam
geometry by shifting the detector (centerRow) and the volume (offsetZ), but this is pretty limited.  Modular-beam
geometries allow more custom setups because you can specify the source location, detector location, and detector orientation
for all projections anywhere you want.  For convenience, we will start with a cone-beam geometry, convert it to modular-beam
and then make the adjustments for a laminography setup.
'''

# Specify the number of detector columns which is used below
# Scale the number of angles and the detector pixel size with numCols
numCols = 512//2
numAngles = 360
pixelSize = 0.65*512/numCols

# Set the number of detector rows
numRows = numCols#//3

# Set the scanner geometry
sod = 1100 # source-to-object distance (mm)
sdd = 1400 # source-to-detector distance (mm)
leapct.set_conebeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0), sod, sdd)

# Set the lamonography angle which is the rotation of the axis of rotation from the z-axis
laminographyAngle = 40.0 # degrees

# Switch to modular-beam coordinates
leapct.convert_to_modularbeam()

# Get the source positions, detector positions, and detector orientation for all projections
sourcePositions = leapct.get_sourcePositions()
moduleCenters = leapct.get_moduleCenters()
rowVecs = leapct.get_rowVectors()
colVecs = leapct.get_colVectors()

# Shift the source up and the detector down so that the source and detector are
# aiming down by "laminographyAngle" degrees
# one could also rotate the "rowVecs" parameter is necessary, but note that if this
# is rotated more than 5 degrees the FBP reconstruction algorithms will not work
# and one will be required to reconstruct with an iterative method
sourcePositions[:,2] = np.tan(laminographyAngle*np.pi/180.0)*sod
moduleCenters[:,2] = -np.tan(laminographyAngle*np.pi/180.0)*(sdd-sod)

# Use the following 7 lines to rotate the detector as well
from scipy.spatial.transform import Rotation as R
sin_theta = np.sin(-0.5*laminographyAngle*np.pi/180.0)
cos_theta = np.cos(-0.5*laminographyAngle*np.pi/180.0)
for n in range(numAngles):
    q = np.append(colVecs[n,:].copy()*sin_theta, cos_theta)
    A = R.from_quat(q).as_matrix()
    rowVecs[n,:] = np.matmul(A, rowVecs[n,:])

# Now re-set the modular-beam geometry with the modified source and detector locations
leapct.set_modularbeam(numAngles, numRows, numCols, pixelSize, pixelSize, sourcePositions, moduleCenters, rowVecs, colVecs)

# Set the volume parameters.
# It is best to do this after the CT geometry is set
leapct.set_default_volume()
leapct.set_numZ(2*int(np.ceil(12.0/leapct.get_voxelHeight()+1))) # reduce the number of z-slices to those that just occupy the object

# If you want to specify the volume yourself, use this function:
#leapct.set_volume(numX, numY, numZ, voxelWidth=None, voxelHeight=None, offsetX=None, offsetY=None, offsetZ=None):

# Trouble-Shooting Functions
# Print the parameters to the screen.  We also plot 5 of the projections to ensure
# that the geometry was set properly
leapct.print_parameters()
leapct.sketch_system([0, 45, 90, 135, 180])

# Allocate space for the projections and the volume
# You don't have to use these functions; they are provided just for convenience
# All you need is for the data to be C contiguous float32 arrays with the right dimensions
g = leapct.allocateProjections() # shape is numAngles, numRows, numCols
f = leapct.allocateVolume() # shape is numZ, numY, numX

# Specify a phantom to test the code
# Here we make a cylindrical phantom with some cross-hatched high-density features
leapct.addObject(f, 4, np.array([0.0, 0.0, 0.0]), 120.0*np.array([1.0, 1.0, 0.1]), 0.02, None, None, 3)

leapct.addObject(f, 1, np.array([-60.0, 0.0, -4.0]), np.array([5.0, 80.0, 1.0]), 0.04, None, None, 3)
leapct.addObject(f, 1, np.array([-40.0, 0.0, -4.0]), np.array([5.0, 80.0, 1.0]), 0.04, None, None, 3)
leapct.addObject(f, 1, np.array([-20.0, 0.0, -4.0]), np.array([5.0, 80.0, 1.0]), 0.04, None, None, 3)
leapct.addObject(f, 1, np.array([0.0, 0.0, -4.0]), np.array([5.0, 80.0, 1.0]), 0.04, None, None, 3)
leapct.addObject(f, 1, np.array([20.0, 0.0, -4.0]), np.array([5.0, 80.0, 1.0]), 0.04, None, None, 3)
leapct.addObject(f, 1, np.array([40.0, 0.0, -4.0]), np.array([5.0, 80.0, 1.0]), 0.04, None, None, 3)
leapct.addObject(f, 1, np.array([60.0, 0.0, -4.0]), np.array([5.0, 80.0, 1.0]), 0.04, None, None, 3)

leapct.addObject(f, 1, np.array([0.0, -60.0, 0.0]), np.array([80.0, 5.0, 1.0]), 0.04, None, None, 3)
leapct.addObject(f, 1, np.array([0.0, -40.0, 0.0]), np.array([80.0, 5.0, 1.0]), 0.04, None, None, 3)
leapct.addObject(f, 1, np.array([0.0, -20.0, 0.0]), np.array([80.0, 5.0, 1.0]), 0.04, None, None, 3)
leapct.addObject(f, 1, np.array([0.0, 0.0, 0.0]), np.array([80.0, 5.0, 1.0]), 0.04, None, None, 3)
leapct.addObject(f, 1, np.array([0.0, 20.0, 0.0]), np.array([80.0, 5.0, 1.0]), 0.04, None, None, 3)
leapct.addObject(f, 1, np.array([0.0, 40.0, 0.0]), np.array([80.0, 5.0, 1.0]), 0.04, None, None, 3)
leapct.addObject(f, 1, np.array([0.0, 60.0, 0.0]), np.array([80.0, 5.0, 1.0]), 0.04, None, None, 3)

leapct.addObject(f, 1, np.array([-60.0, 0.0, 4.0]), np.array([5.0, 80.0, 1.0]), 0.04, None, None, 3)
leapct.addObject(f, 1, np.array([-40.0, 0.0, 4.0]), np.array([5.0, 80.0, 1.0]), 0.04, None, None, 3)
leapct.addObject(f, 1, np.array([-20.0, 0.0, 4.0]), np.array([5.0, 80.0, 1.0]), 0.04, None, None, 3)
leapct.addObject(f, 1, np.array([0.0, 0.0, 4.0]), np.array([5.0, 80.0, 1.0]), 0.04, None, None, 3)
leapct.addObject(f, 1, np.array([20.0, 0.0, 4.0]), np.array([5.0, 80.0, 1.0]), 0.04, None, None, 3)
leapct.addObject(f, 1, np.array([40.0, 0.0, 4.0]), np.array([5.0, 80.0, 1.0]), 0.04, None, None, 3)
leapct.addObject(f, 1, np.array([60.0, 0.0, 4.0]), np.array([5.0, 80.0, 1.0]), 0.04, None, None, 3)

# Display the phantom with napari
leapct.display(f)


# "Simulate" projection data
startTime = time.time()
leapct.project(g,f)
print('Forward Projection Elapsed Time: ' + str(time.time()-startTime))

# Display the projection data to see that it appears as expected
leapct.display(g)


# Add noise to the data (just for demonstration purposes)
I_0 = 50000.0
#g[:] = -np.log(np.random.poisson(I_0*np.exp(-g))/I_0)

# Reset the volume array to zero, otherwise iterative reconstruction algorithm will start their iterations
# with the true result which is cheating
f[:] = 0.0

# Reconstruct the data
# We will start with an FBP reconstruction and refine it with an iterative method to remove some of the
# so-called "cone-beam" artifacts.  There are many iterative reconstruction algorithms to choose from.
startTime = time.time()
#leapct.backproject(g,f)
leapct.FBP(g,f)
filters = filterSequence(1.0e0)
filters.append(TV(leapct, delta=0.02/20.0))
#leapct.ASDPOCS(g,f,10,10,1,filters)
#leapct.SART(g,f,10,10)
#leapct.OSEM(g,f,10,10)
#leapct.LS(g,f,50,'SQS')
leapct.RWLS(g,f,100,filters,None,'SQS')
#leapct.RDLS(g,f,50,filters,1.0,True,1)
#leapct.MLTR(g,f,10,10,filters)
print('Reconstruction Elapsed Time: ' + str(time.time()-startTime))


# Post Reconstruction Smoothing (optional)
# Here are some optional post reconstruction noise filters that can be applied
# Try uncommenting out these lines to test how they work
#startTime = time.time()
#leapct.diffuse(f,0.02/20.0,4)
#leapct.MedianFilter(f)
#leapct.BlurFilter(f,2.0)
#print('Post-Processing Elapsed Time: ' + str(time.time()-startTime))

# Display the result with napari
leapct.display(f)
#import matplotlib.pyplot as plt
#plt.imshow(np.squeeze(f[f.shape[0]//2,:,:]), cmap='gray')
#plt.show()
