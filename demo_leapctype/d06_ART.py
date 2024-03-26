import sys
import os
import time
import numpy as np
from leapctype import *
leapct = tomographicModels()

'''
This demo script shows you how to use the Attenuated Radon Transform (ART) algorithms in LEAP
Note that only parallel-beam geometries are supported.
The attenuation map may either be a voxelized map or as a cylinder where the user
specifies the attenuation coefficient and radius of the cylinder
We provide analytic reconstruction (FBP) of the ART for either specification of the attenuation map
via Novikov's inversion formula.  Unfortunately we only have this implemented for 360 degree angular range.
Of course, one can always reconstruct with iterative reconstruction algorithms.

Two applications of the Attenuated Radon Transform are SPECT and VAM.  We provide VAM algorithms here:
https://github.com/LLNL/LEAP/tree/main/VAM
'''


# Specify the number of detector columns which is used below
# Scale the number of angles and the detector pixel size with N
numCols = 512
numAngles = 4*int(360*numCols/1024)
pixelSize = 0.65*512/numCols

# Set the number of detector rows
numRows = 1

# Set the scanner geometry
leapct.set_parallelbeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0))

# Set the volume parameters.
# It is best to do this after the CT geometry is set
leapct.set_default_volume()

# Trouble-Shooting Functions
leapct.print_parameters()
#leapct.sketch_system()

# Allocate space for the projections and the volume
g = leapct.allocateProjections()
f = leapct.allocateVolume()
mu = leapct.allocateVolume()

# Specify the attenuation map parameters
# In this demo, one can specify a voxelized attenuation or a cylindrical attenuation volume
muCoeff = 0.01
muRadius = 150.0

# Specify a phantom composed of a 300 mm diameter sphere
leapct.addObject(f, 4, np.array([0.0, 0.0, 0.0]), np.array([muRadius, muRadius, muRadius]), 1.0)
leapct.addObject(f, 0, np.array([0.0, 20.0, 0.0]), np.array([20.0, 20.0, 20.0]), 10.0)

# Specify a voxelized attenuation volume
leapct.addObject(mu, 4, np.array([0.0, 0.0, 0.0]), np.array([muRadius, muRadius, muRadius]), muCoeff)
leapct.addObject(mu, 0, np.array([0.0, -30.0, 0.0]), np.array([20.0, 20.0, 20.0]), 0.0)
leapct.BlurFilter(mu, 2.0)

# Here is whether you choose the voxelized or cylindrical attenuation map
leapct.set_attenuationMap(mu)
#leapct.set_cylindircalAttenuationMap(muCoeff, muRadius)

# "Simulate" projection data
startTime = time.time()
leapct.project(g,f)
print('Forward Projection Elapsed Time: ' + str(time.time()-startTime))
#leapct.display(g)

# Add noise to the data (just for demonstration purposes)
#g[:] = np.random.poisson(g)

# Reset the volume array to zero, otherwise iterative reconstruction algorithm will start their iterations
# with the true result which is cheating
f[:] = 0.0

# Reconstruct the data
startTime = time.time()
#leapct.backproject(g,f)
leapct.FBP(g,f)
#leapct.inconsistencyReconstruction(g,f)
#leapct.print_cost = True
filters = filterSequence(1.0e0)
filters.append(TV(leapct, delta=0.02/20.0))
#leapct.ASDPOCS(g,f,10,10,1,filters)
#leapct.SART(g,f,10,10)
#leapct.OSEM(g,f,10,10)
#leapct.LS(g,f,50,'SQS')
#leapct.RWLS(g,f,50,filters,None,'SQS')
#leapct.RDLS(g,f,50,filters,1.0,True,1)
#leapct.MLTR(g,f,10,10,filters)
print('Reconstruction Elapsed Time: ' + str(time.time()-startTime))


# Post Reconstruction Smoothing (optional)
#startTime = time.time()
#leapct.diffuse(f,0.02/20.0,4)
#leapct.MedianFilter(f)
#leapct.BlurFilter(f,2.0)
#print('Post-Processing Elapsed Time: ' + str(time.time()-startTime))

# Display the result with napari
leapct.display(f)
