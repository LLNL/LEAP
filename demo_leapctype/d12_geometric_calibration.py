import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from leap_preprocessing_algorithms import *
from leapctype import *
leapct = tomographicModels()

'''
This demo script provides three examples of how to find some parameters that are commonly determined in a
geometric calibration.  The first two examples show methods of how to find the center detector column,
which in LEAP is called "centerCol".  The third example shows a robust method to find the detector tilt
or "clocking rotation" of the detector.
'''

whichDemo = 4

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
if whichDemo == 4:
    # demo 4 uses data with a rotated detector
    leapct_rot = tomographicModels()
    leapct_rot.copy_parameters(leapct)
    leapct_rot.convert_to_modularbeam()
    leapct_rot.rotate_detector(2.34)
    leapct_rot.project(g,f)
else:
    leapct.project(g,f)
print('Forward Projection Elapsed Time: ' + str(time.time()-startTime))


# Add noise to the data (just for demonstration purposes)
I_0 = 50000.0
print('Adding noise to data...')
g[:] = -np.log(np.random.poisson(I_0*np.exp(-g))/I_0)


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
    
    # Note that parameter_sweep does not automatically change any parameter.  It is up to the
    # user to look at the reconstructions and given metrics and set the parameters themselves.
    
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

elif whichDemo == 4:
    
    """
    The data here was simulated with a 2.34 degree detector tilt.
    
    This part of the script demonstrates the usage of conjugate rays to determine detector tilt
    (rotation of the detector around the optical axis) and horizontal detector shifts.
    
    Conjugate rays are pairs of rays whose azimuthal angle differ by 180 degrees.  In parallel-
    and fan-beam, these rays are identical other than they travel in opposite directions and thus
    these measurements are assumed to be essentially identical.  In cone-beam these rays are similar
    but no identical.  Regardless, the similarity of conjugate rays can be leveraged to estimate some
    CT geometry parameters.
    
    We shall demonstrate two functions: estimate_tilt and conjugate_difference.
    The estimate_tilt function estimates the detector tilt (rotation of the detector around the optical
    axis).  This function does not change any CT geometry parameters; it just returns the estimated
    angle of rotation.  Below we show how to update the geometry.
    The conjugate_difference function returns a 2D array of the difference of two conjugate projections.
    You may provide a detector rotation angle and/or a new centerCol parameter to test the accuracy of
    these particular parameters.  A good estimate should show mostly noise in the difference.  This function
    can be used to tune these parameters by hand (by observing the difference) or can be put into a cost
    function by summing the squares of the conjugate difference.  Just like the estimate_tilt function,
    this function does not update the geometry.  The user must do this themselves.
    """
    
    # First let's view the conjugate projection difference without modeling the detector tilt (rotation)
    diff_0 = leapct.conjugate_difference(g, 0.0)
    leapct.display(diff_0)
    
    # Now let's estimate the detector tilt
    # This function only estimates the tilt.  To update the LEAP CT geometry parameters do the following:
    #leapct.convert_to_modularbeam()
    #leapct.rotate_detector(alpha)
    alpha = leapct.estimate_tilt(g)
    print('estimated detector tilt: ' + str(alpha) + ' degrees')
    
    # Now we display the conjugate projection difference using the estimated rotation angle
    # Here you should see an image with fewer features (should look more just like noise)
    diff_alpha = leapct.conjugate_difference(g, alpha)
    leapct.display(diff_alpha)
    
    # Next we calculate the error metrics for these two cases
    print('conjugate projection error metric with no detector rotation: ' + str(np.sum(diff_0**2)))
    print('conjugate projection error metric with estimated detector rotation: ' + str(np.sum(diff_alpha**2)))    
    
elif whichDemo == 5:

    """
    This demo demonstrates the use of the automated geometric calibration procedure in LEAP.
    This function can estimate centerRow, centerCol, tau, and detector tilt (detector rotation around the optical axis).
    We recommend that the current geometry specification be close to the true values, otherwise this function may
    fail to find a global minimum or stop early before the absolute minimum is found.
    We also recommend that after this function is applied, that one still run the find_centerCol() function
    because this is the most accurate automatic estimation of the centerCol parameter
    """

    # First we perturb the true geometry specification
    Delta_centerRow = 10.0
    Delta_centerCol = 20.0
    #Delta_tau = 3.0

    leapct.set_centerRow(leapct.get_centerRow()+Delta_centerRow)
    leapct.set_centerCol(leapct.get_centerCol()+Delta_centerCol)
    #leapct.set_tau(Delta_tau)
    
    # Specify the number of times the minimize function is called
    # This may help estimate a more accurate estimate
    numIter = 2
    
    from scipy.optimize import minimize

    # Start by trying to find the center column
    leapct.find_centerCol(g)

    # We will now try to estimate both centerRow and centerCol
    # Define the cost function for the minimize function  
    # This function is in the tomographicModels Python class
    costFcn_rc = lambda x: leapct.consistency_cost(g, x[0], x[1], 0.0, 0.0)

    # Specify the initial guess and bounds for the parameters
    # Here we demonstrate using just 2 of the 4 possible parameters:  centerRow, centerCol
    # Feel free to make the search region (bounds) bigger or smaller to fit your needs
    x0 = (0.0, 0.0)
    bounds = ((-200, 200), (-200, 200))

    # Run the estimation algorithm
    print('\nestimating geometry parameters...')
    startTime = time.time()
    for n in range(numIter):
        res = minimize(costFcn_rc, x0, method='Powell', bounds=bounds)
        x0 = res.x

    # Print out estimated parameters and various cost values
    print(res.message)
    print('estimated perturbations: ' + str(res.x))
    print('optimized cost = ' + str(costFcn_rc(res.x)))
    
    # Set the estimated geometry parameters
    # The estimated values were perturbations of the current values
    # So make sure you add these estimates to the current values
    leapct.set_centerRow(leapct.get_centerRow() + res.x[0])
    leapct.set_centerCol(leapct.get_centerCol() + res.x[1])
    print('Current estimate of detector center: ' + str(leapct.get_centerRow()) + ', ' + str(leapct.get_centerCol()) + '\n')

    # Now let's repeat the above, adding in detector tilt (rotation around the optical axis) which is measured in degrees
    # Sometimes this estimate fails to provide a good estimate.  The search window (bounds) should be about +/- 2 degrees
    costFcn_rct = lambda x: leapct.consistency_cost(g, x[0], x[1], 0.0, x[2])
    x0 = (0.0, 0.0, 0.0)
    bounds = ((-50, 50), (-50, 50), (-2, 2))
    for n in range(numIter):
        res = minimize(costFcn_rct, x0, method='Powell', bounds=bounds)
        x0 = res.x
    
    # Print out estimated parameters and various cost values
    print(res.message)
    print('estimated perturbations: ' + str(res.x))
    print('optimized cost = ' + str(costFcn_rc(res.x)))
    
    if np.abs(res.x[2]) > 0.1:
        # A significant detector rotation was detected, so let's assume this is correct
        # In this case, we have to switch to modular-beam geometry
        leapct.set_centerRow(leapct.get_centerRow() + res.x[0])
        leapct.set_centerCol(leapct.get_centerCol() + res.x[1])
        leapct.convert_to_modularbeam()
        leapct.rotate_detector(res.x[2])
    else:
        # Assume the detector is not tilted
        # Refine the centerCol estimate now that we have a good estimate of the centerRow
        leapct.find_centerCol(g)
    print('Current estimate of detector center: ' + str(leapct.get_centerRow()) + ', ' + str(leapct.get_centerCol()))
    
    print('Elapsed time: ' + str(time.time()-startTime) + ' seconds')
    
    # Perform a single-slice reconstruction to inspect its quality
    leapct.set_numZ(1)
    f = leapct.FBP(g)
    plt.imshow(np.squeeze(f), cmap='gray')
    plt.show()
    
    