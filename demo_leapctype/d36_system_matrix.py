import sys
import os
import time
import numpy as np
from leapctype import *
leapct = tomographicModels()

'''
This script demonstrates how to use the system_matrix calculation routine in LEAP.  The system matrix is calculated
for a fixed projection angle and fixed detector row pixel and variable detector column for all CT volume voxels.

For parallel- and fan-beam
geometries the syntax is as follows:
A, indices = leapct.system_matrix(iAngle)
where iAngle is the projection angle index
A is of size (numCols, index of elements) and stores the system matrix values
indices is of size (numCols, index of elements, 2) and stores the CT volume voxel indices,
where indices[iCol, j, 0] is the y-coordinate voxel index
and indices[iCol, j, 1] is the x-coordinate voxel index

Cone-beam and modular-beam coordinate are not ready yet, but when they are,
indices will be of size (numCols, index of elements, 3)
'''


# Specify the number of detector columns which is used below
# Scale the number of angles and the detector pixel size with N
numCols = 64
#numAngles = 2*int(360*numCols/1024)*0+1 # just using one projection angle because this takes forever
numAngles = 90
pixelSize = 0.65*512/numCols

# Set the number of detector rows
# You can increase this, but let's start with an easy case of just one detector row
numRows = numCols*0+1

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

# Copy data to GPU
#'''
# Comment this section out to revert back to multi-GPU solution
# with CPU-GPU data transfers to see when easy case is advantageous
device = torch.device("cuda:" + str(leapct.get_gpu()))
g = torch.from_numpy(g).to(device)
f = torch.from_numpy(f).to(device)
#'''

# Now perform a forward projection with the system matrix
# This takes a very long time!!!
Pf = leapct.copyData(g)
Pf[:] = 0.0

#'''
for iAngle in range(leapct.get_numAngles()):
    print(str(iAngle) + ' of ' + str(leapct.get_numAngles()))
    A, indices = leapct.system_matrix(iAngle)
    #print(A.shape)
    #print(indices.shape)
    A = A.cpu().numpy()
    indices = indices.cpu().numpy()
    #for i in range(A.shape[1]):
    #    print("[%d] %f, (%d, %d)" % (i, A[32, i], indices[32, i, 0], indices[32, i, 1]))
    for iCol in range(A.shape[0]):
        for ind in range(A.shape[1]):
            a = A[iCol,ind]
            if a > 0.0: # some elements are zero, so check first
                for iRow in range(numRows):
                    Pf[iAngle,iRow,iCol] += a*f[iRow,indices[iCol,ind,0],indices[iCol,ind,1]]
#'''

#A, indices = leapct.system_matrix(40, 0, 32, max_size=None, onGPU=True)
#print(A)
#print(indices)
    
leapct.display(g)
leapct.display(Pf)
leapct.display(Pf-g)
