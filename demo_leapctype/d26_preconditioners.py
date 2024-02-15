import sys
import os
import time
import numpy as np
from leapctype import *
leapct = tomographicModels()

'''
This script demonstrates the usage of preconditioners in LS, WLS, RLS, and RWLS algorithms.
This family of algorithms uses conjugate gradient to minimize the cost function.
One may specify a preconditioner which may accelerate convergence even further in some cases.
The preconditioner are:
Separable Quadratic Surrogate (SQS): This is a method made popular by Jeff Fessler where the preconditioner
is given by 1/(P*WP1), where 1 is a volume of all ones and W is the weighting matrix which may be identity.
Note that the SART algorithm is a preconditioned gradient descent algorithm with constant step size, where W = 1/P1
and thus the surrogate becomes 1/(P*WP1) = 1/P*1.
RAMP: This method uses a 2D ramp filter applied to each z-slice of a volume.  This method was proposed by Clinthorne and Fessler and Booth.
Statistical-Analytic Regularized Reconstruction (SARR): This method is very similar to iterative FBP (IFBP) methods and was proposed by myself (Kyle).
This method convergences extremely fast, but only approximately minimizes the cost function.
The RAMP and SARR preconditioners should only be used when one has sufficient angular sampling and is not to be used for sparse-view CT reconstruction.
'''


# Specify the number of detector columns which is used below
# Scale the number of angles and the detector pixel size with N
numCols = 512
numAngles = 2*2*int(360*numCols/1024)
pixelSize = 0.65*512/numCols

# Set the number of detector rows
numRows = numCols*0+1

# Set the scanner geometry
#leapct.set_parallelbeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0))
#leapct.set_fanbeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0), 1100, 1400)
leapct.set_conebeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0), 1100, 1400)
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
#for n in range(11):
#    leapct.addObject(f, 4, 20*(n-5)*np.array([0.0, 0.0, 1.0]), np.array([120.0, 120.0, 5.0]), 1.0, None, None, 3)
#leapct.addObject(f, 4, np.array([0.0, 0.0, 0.0]), 10.0*np.array([1.0, 1.0, 1.0]), 1.0, None, None, 3)
#leapct.display(f)

# "Simulate" projection data
startTime = time.time()
leapct.project(g,f)
print('Forward Projection Elapsed Time: ' + str(time.time()-startTime))
#leapct.display(g)

# Add noise to the data (just for demonstration purposes)
I_0 = 50000.0
#g[:] = -np.log(np.random.poisson(I_0*np.exp(-g))/I_0)

'''
# Uncomment this section to enable processing completely on the GPU
# which may be faster because it avoids CPU-GPU data transfers
device = torch.device("cuda:" + str(leapct.get_gpu()))
g = torch.from_numpy(g).to(device)
f = torch.from_numpy(f).to(device)
#'''

# Reset the volume array to zero, otherwise iterative reconstruction algorithm will start their iterations
# with the true result which is cheating
f[:] = 0.0

# Reconstruct the data
# We'll start with an FBP reconstruction (which is optional) and then run
# a reconstruction which minimizes a Least Squares cost function
# using preconditioned conjugate gradient
# Let's turn on the cost function reporting so we can see how the preconditioners speed
# up the convergence rate
leapct.print_cost = True
startTime = time.time()
leapct.FBP(g,f)
#leapct.print_cost = True
#leapct.LS(g,f,10,'SQS')
#leapct.LS(g,f,10,'RAMP')
leapct.LS(g,f,10,'SARR')
print('Reconstruction Elapsed Time: ' + str(time.time()-startTime))


# Display the result with napari
leapct.display(f)
