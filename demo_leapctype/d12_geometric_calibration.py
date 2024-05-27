import sys
import os
import time
import numpy as np
from leapctype import *
leapct = tomographicModels()

'''
This demo script provides three examples of how to find some parameters that are commonly determined in a
geometric calibration.  The first two examples show methods of how to find the center detector column,
which in LEAP is called "centerCol".  The third example shows a robust method to find the detector tilt
or "clocking rotation" of the detector.
'''


# Specify the number of detector columns which is used below
# Scale the number of angles and the detector pixel size with N
numCols = 512
numAngles = 2*2*int(360*numCols/1024)
pixelSize = 0.65*512/numCols

# Set the number of detector rows
numRows = numCols

# Set the central column and row indices
# Here we will start with a detector that
# is not centered on the optical axis
centerRow = 0.5*(numRows-1) - 3.25
centerCol = 0.5*(numCols-1) + 5.75

# Set the scanner geometry
#leapct.set_parallelbeam(numAngles, numRows, numCols, pixelSize, pixelSize, centerRow, centerCol, leapct.setAngleArray(numAngles, 360.0))
#leapct.set_fanbeam(numAngles, numRows, numCols, pixelSize, pixelSize, centerRow, centerCol, leapct.setAngleArray(numAngles, 360.0), 1100, 1400)
leapct.set_conebeam(numAngles, numRows, numCols, pixelSize, pixelSize, centerRow, centerCol, leapct.setAngleArray(numAngles, 360.0), 1100, 1400)
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


# Add noise to the data (just for demonstration purposes)
print('Adding noise to data...')
I_0 = 50000.0
g[:] = -np.log(np.random.poisson(I_0*np.exp(-g))/I_0)


whichDemo = 2
if whichDemo == 1:
    # In this first demo, we show how LEAP can estimate the centerCol parameter
    # by minimizing the differences of conjugate rays.
    # Conjugate rays are measured ray paths that from different source positions, but
    # (approximately) pass through the same part of the object being imaged
    # This routine works best for a full scan (angular range of 360 degrees or more), but
    # will work for a short scan
    
    # First set centerCol to something else, so we know we aren't cheating
    leapct.set_centerCol(0.0)
    
    # Now use the data to estimate the centerCol
    leapct.find_centerCol(g)
    
    # Print the results
    print('True centerCol = ' + str(centerCol))
    print('Estimated centerCol = ' + str(leapct.get_centerCol()))
    
elif whichDemo == 2:
    # In this second demo, we show how one can use the parameter_sweep function to reconstruct a 
    # a sequence of single-slice reconstructions at a range of parameter values which one can inspect
    # to determine the correct value of a parameter.
    # This sequence of reconstructions can be performed with FBP or the so-called "inconsistency reconstruction"
    # An Inconsistency Reconstruction is an FBP reconstruction except it replaces the ramp filter with
    # a derivative.  For scans with angular ranges of 360 or more this will result in a pure noise
    # reconstruction if the geometry is calibrated and there are no biases in the data.  This can
    # be used as a robust way to find the centerCol parameter or estimate detector tilt.
    
    # You choose which slice to reconstruct by the iz parameter which is the z-slice index value of the
    # slice you want to reconstruct.  If you don't provide this value or provide it as None, then
    # the center-most slice will be used.
    
    # When using an Inconsistency Reconstruction, parameter_sweep will print out an inconsistency
    # metric for each reconstruction.  The LOWEST value of this metric should correspond to the
    # correct value for this parameter.
    
    # When using FBP reconstruction, parameter_sweep will print out an entropy metric
    # for each reconstruction..  The LARGEST value of this metric should correspond to the
    # correct value for this parameter.
    
    from leap_preprocessing_algorithms import *
    centerCols = np.array(range(-5,5+1))+leapct.get_centerCol()
    #f_stack = parameter_sweep(leapct, g, centerCols, 'centerCol', iz=None, algorithmName='FBP')
    f_stack = parameter_sweep(leapct, g, centerCols, 'centerCol', iz=None, algorithmName='inconsistencyReconstruction')
    leapct.display(f_stack)
    
    
elif whichDemo == 3:
    # This demo is similar to the second demo above, except we sweep over detector rotation angles
    # Internally, the paramer_sweep function is using the function convert_to_modularbeam() to convert
    # the given geometry to a modular-beam geometry because this is the only geometry that can handle rotations
    # and then uses the rotate_detector() function to rotate the coordinates of each detector position
    
    from leap_preprocessing_algorithms import *
    tiltAnglesInDegrees = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    #f_stack = parameter_sweep(leapct, g, tiltAnglesInDegrees, 'tilt', iz=None, algorithmName='FBP')
    f_stack = parameter_sweep(leapct, g, tiltAnglesInDegrees, 'tilt', iz=None, algorithmName='inconsistencyReconstruction')
    leapct.display(f_stack)
    
    