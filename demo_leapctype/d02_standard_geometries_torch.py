import sys
import os
import time
import numpy as np
from leapctype import *
leapct = tomographicModels()

'''
This script is nearly identical to d01_standard_geometries.py
except it uses torch tensors on a GPU to run iterative reconstructions
This demonstrates LEAP's ability to process data that is already on a GPU.
Because no CPU-GPU data transfers are necessary these routines may run much faster.
However they are limited to the amount of GPU memory one has and only process on one GPU at a time.
Thus for small-ish data sizes this may be beneficial, but as the data sizes get larger it is worth
to pay the price of CPU-GPU data transfers because you can use multiple GPUs and aren't limited by GPU
memory, just CPU memory.

If CPU memory is also a concern, see "d13_cropping_subchunking.py"

Thus, one may want to install pytorch even if they are not interested in LEAP's ability to integrate with
PyTorch neural networks.

If any line below seems confusing, please read d01_standard_geometries.py which has more comments

The main difference between this file and d01_standard_geometries.py are the three lines below that
copy the numpy arrays to torch tensors on a GPU.  One may use these three line is any of the demo scripts
we provide
'''


# Specify the number of detector columns which is used below
# Scale the number of angles and the detector pixel size with N
numCols = 512
numAngles = 2*2*int(360*numCols/1024)
pixelSize = 0.65*512/numCols

# Set the number of detector rows
# You can increase this, but let's start with an easy case of just one detector row
numRows = 1

# Set the scanner geometry
leapct.set_parallelbeam(numAngles, numRows, numCols, pixelSize*1100.0/1400.0, pixelSize*1100.0/1400.0, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0))
#leapct.set_fanbeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0), 1100, 1400)
#leapct.set_conebeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0), 1100, 1400)
#leapct.set_coneparallel(numAngles, numRows, numCols, pixelSize, pixelSize*1100.0/1400.0, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0), 1100, 1400)
#leapct.set_curvedDetector()

# Set the volume parameters.
# It is best to do this after the CT geometry is set
leapct.set_default_volume()

# If you want to specify the volume yourself, use this function:
#leapct.set_volume(numX, numY, numZ, voxelWidth=None, voxelHeight=None, offsetX=None, offsetY=None, offsetZ=None):

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
leapct.set_FORBILD(f,True,3)
#leapct.display(f)



# "Simulate" projection data
startTime = time.time()
leapct.project(g,f)
print('Forward Projection Elapsed Time: ' + str(time.time()-startTime))
#leapct.display(g)

# Add noise to the data (just for demonstration purposes)
I_0 = 5000.0
#g[:] = -np.log(np.random.poisson(I_0*np.exp(-g))/I_0)

# Copy data to GPU
#'''
# Comment this section out to revert back to multi-GPU solution
# with CPU-GPU data transfers to see when easy case is advantageous
device = torch.device("cuda:" + str(leapct.get_gpu()))
g = torch.from_numpy(g).to(device)
f = torch.from_numpy(f).to(device)
#'''

# Reset the volume array to zero, otherwise iterative reconstruction algorithm will start their iterations
# with the true result which is cheating
f[:] = 0.0

# Reconstruct the data
startTime = time.time()
#leapct.backproject(g,f)
#leapct.FBP(g,f)
#leapct.inconsistencyReconstruction(g,f)
#leapct.print_cost = True
filters = filterSequence(1.0e0)
filters.append(TV(leapct, delta=0.02/20.0))
#leapct.ASDPOCS(g,f,10,10,1,filters)
#leapct.SART(g,f,10,10)
#leapct.OSEM(g,f,10,10)
#leapct.LS(g,f,50,'SQS')
#leapct.RWLS(g,f,100,filters,None,'SQS')
leapct.RWLS(g,f,1,filters,None,'SQS')
leapct.RDLS(g,f,20,filters,1.0,True,1)
#leapct.MLTR(g,f,10,10,filters)
print('Reconstruction Elapsed Time: ' + str(time.time()-startTime))


# Post Reconstruction Smoothing (optional)
#startTime = time.time()
#leapct.diffuse(f,0.02/20.0,4)
#leapct.MedianFilter(f)
#leapct.BlurFilter(f,2.0)
#print('Post-Processing Elapsed Time: ' + str(time.time()-startTime))


# To convert a torch tensor back to a numpy array, use the following
#f = f.cpu().detach().numpy()


# Display the result with napari
#leapct.display(f)
import matplotlib.pyplot as plt
plt.imshow(np.squeeze(f.cpu().detach().numpy()), cmap='gray')
plt.show()
