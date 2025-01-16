import sys
import os
import time
import numpy as np
from leapctype import *
leapct = tomographicModels()
# Make sure you add: .../LEAP/src to your python path

'''
This script is nearly identical to d01_standard_geometries.py except it demonstrates LEAP's
helical cone-beam functionality.  LEAP has an implementation of helical FBP for the GPU only
Of course, just like all geometries in LEAP, one can use any iterative reconstruction.
For details of the helical FBP algorithm, please see the LEAP technical manual here:
https://github.com/LLNL/LEAP/blob/main/documentation/LEAP.pdf
'''


# Specify the number of detector columns which is used below
# Scale the number of angles and the detector pixel size with N
numCols = 512
numTurns = 10
numAngles = 2*2*int(360*numCols/1024)*numTurns
pixelSize = 0.65*512/numCols

# Set the number of detector rows
numRows = numCols//4

# Set the scanner geometry
leapct.set_conebeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0*numTurns), 1100, 1400)
#leapct.set_coneparallel(numAngles, numRows, numCols, pixelSize, pixelSize*1100.0/1400.0, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0*numTurns), 1100, 1400)
#leapct.set_curvedDetector()

# Set the helical pitch.
leapct.set_normalizedHelicalPitch(0.5)
#leapct.set_normalizedHelicalPitch(1.0)

# Set the volume parameters
leapct.set_default_volume()
#leapct.set_volume(numCols, numCols, numRows)

# Trouble-Shooting Functions
leapct.print_parameters()
#leapct.sketch_system()

# Set the backprojector model, 'SF' (the default setting), is more accurate, but 'VD' is faster
#leapct.set_projector('VD')

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

# Reset the volume array to zero, otherwise iterative reconstruction algorithm will start their iterations
# with the true result which is cheating
f[:] = 0.0

# Reconstruct the data
startTime = time.time()
#leapct.backproject(g,f)
leapct.FBP(g,f)
#leapct.print_cost = True
filters = filterSequence(1.0e0)
filters.append(TV(leapct, delta=0.02/20.0))
#leapct.ASDPOCS(g,f,10,10,1,filters)
#leapct.SART(g,f,10,10)
#leapct.OSEM(g,f,10,10)
#leapct.LS(g,f,50,'SQS')
#leapct.RWLS(g,f,100,filters,None,'SQS')
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
