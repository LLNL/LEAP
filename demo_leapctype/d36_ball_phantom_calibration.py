from leap_preprocessing_algorithms import *
from leapctype import *
leapct = tomographicModels()

"""
This script demonstrates how to use the ball phantom calibration algorithm which is in leap_preprocessing_algorithms.py

A ball phantom calibration is a method to perform geometric calibration of an axial cone-beam CT system where one performs
a CT scan (of about 40 projections) of a stack of metal balls where the spacing between each ball is constant and known.

The ball phantom calibration algorithm assumes that one has processed the data into attenuation radiographs (flat field and -log).

It is required that all balls be present in all projections (i.e., they cannot fall outside the top, bottom or sides of the detector).
"""

### Set Nominal Geometry
L = 1
numCols = 4000//L
numAngles = 36
pixelSize = 11.62/1000.0*L
numRows = 2096//L
sod = 255.8
sdd = 345.1
centerRow = 0.5*(numRows-1)
centerCol = 0.5*(numCols-1) + 60.0
leapct.set_conebeam(numAngles, numRows, numCols, pixelSize, pixelSize, centerRow, centerCol, leapct.setAngleArray(numAngles, 360.0), sod, sdd)
leapct.set_default_volume()

### Set "True" Geometry
# Here we perturb the original CT geometry to mimic the case where the true CT geometry in unknown
leapct_true = tomographicModels()
leapct_true.copy_parameters(leapct)
leapct_true.set_sod(leapct_true.get_sod()+10.0)
leapct_true.set_sdd(leapct_true.get_sdd()+20.0)
leapct_true.set_centerRow(leapct_true.get_centerRow()-10.0)
leapct_true.set_centerCol(leapct_true.get_centerCol()-15.0)
leapct_true.convert_to_modularbeam()
leapct_true.rotate_detector(1.0)

### Simulate Data using the "true" or perturbed geometry
ballRadius = 0.5*1.0/2.0
r = 10.0
T_z = 4*ballRadius
numBalls = int(15.0/T_z)
z_0 = -T_z*0.5*(numBalls-1.0)
alpha = 5.0*np.pi/180.0

leapct_true.addObject(None, 4, 0.0, np.array([11.0, 11.0, 8.0]), 0.04, oversampling=1)
leapct_true.addObject(None, 4, 0.0, np.array([9.0, 9.0, 8.0]), 0.00, oversampling=1)
for k in range(numBalls):
    leapct_true.addObject(None, 0, np.array([r*np.cos(alpha), r*np.sin(alpha), T_z*k+z_0]), ballRadius, 2.25, oversampling=3)
g = leapct_true.allocate_projections()
leapct_true.rayTrace(g,oversampling=1)
#leapct_true.display(g)

# Initialize the ball phantom calibration class object
# one must provide the tomographicModels object, the spacing between balls (mm), and the
# projection data of the ball phantom
cal = ball_phantom_calibration(leapct, T_z, g, segmentation_threshold=np.max(g)/3.0)

# We now provide an initial guess of the CT geometry parameters, below is the format
# Note that LEAP usually specifes the distances as sod and sdd, but here we use sod and odd,
# where odd = object-to-detector distance (sdd = sod + odd)
# We do this because it is best to decouple these measurements in the calibration procedure
#x = [centerRow, centerCol, sod, odd, psi, theta, phi, r, z_0, phase]

# The ball phantom calibration class has a method to estimate an initial guess, we recommend using this
x = cal.initial_guess(g)
print('initial guess of CT geometry parameters: ', x)
print('cost function of intial guess: ', cal.cost(x))

# We can plot the locations of the center of each ball for each projection
# and compare to the predicted location based on the CT geometry parameters.
# An ideal estimate of the CT geometry parameters will have the red balls over the black balls
cal.do_plot(x)

# Now let's perform the optimization, this should be very fast
# If we provide a second argument of True, then after the CT geometry parameters are estimated,
# the routine will update these in LEAP so it is ready to reconstruct data
res = cal.optimize(x,True)
print('estimate of CT geometry parameters: ', res.x)
print('cost function of estimate: ', cal.cost(res.x))
cal.do_plot(res.x)

#leapct_true.print_parameters()

# Test the reconstruction with the calibration CT geometry parameters
#leapct.display(leapct.FBP(g))
