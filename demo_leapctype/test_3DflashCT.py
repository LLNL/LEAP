import sys
import os
import time
import numpy as np
from leapctype import *
leapct = tomographicModels()

# Specify the number of detector columns which is used below
# Scale the number of angles and the detector pixel size with N
numCols = 512//2
numAngles = 15
pixelSize = 0.65*512/float(numCols)*2.0*11.0/14.0

# Set the number of detector rows
numRows = numCols
#numRows = 1

# Make this modular-beam geometry just like a cone-beam dataset
# so let's define sod and sdd when defining our geometry
# In general, if your data fits into one of the standard geometry types,
# such as, parallel-, fan-, or cone-beam, then it is STRONGLY recommended
# that you use the standard type.
# The modular-beam projectors are not as fast and not as accurate
sod = 1000.0
sdd = 2000.0

# Set the scanner geometry
sourcePositions = np.ascontiguousarray(np.zeros((numAngles,3)).astype(np.float32), dtype=np.float32)
moduleCenters = np.ascontiguousarray(np.zeros((numAngles,3)).astype(np.float32), dtype=np.float32)
colVectors = np.ascontiguousarray(np.zeros((numAngles,3)).astype(np.float32), dtype=np.float32)
rowVectors = np.ascontiguousarray(np.zeros((numAngles,3)).astype(np.float32), dtype=np.float32)

T_theta = 2.0*np.pi/5.0
ind = 0
for m in range(3):
    phi = 2.0*m*np.pi/3.0
    for n in range(numAngles//3): # 5
        #theta = n*T_theta + 2.0*np.pi*m/15.0
        theta = n*T_theta + np.pi/180.0*m*(12.0)
        
        sourcePositions[ind,0] = np.cos(phi)*np.sin(theta)
        sourcePositions[ind,1] = np.sin(phi)*np.sin(theta)
        sourcePositions[ind,2] = np.cos(theta)
        
        moduleCenters[ind,0] = (sod-sdd)*np.cos(phi)*np.sin(theta)
        moduleCenters[ind,1] = (sod-sdd)*np.sin(phi)*np.sin(theta)
        moduleCenters[ind,2] = (sod-sdd)*np.cos(theta)
        
        rowVectors[ind,0] = np.cos(phi)*np.cos(theta)
        rowVectors[ind,1] = np.sin(phi)*np.cos(theta)
        rowVectors[ind,2] = -np.sin(theta)
        
        colVectors[ind,:] = np.cross(sourcePositions[ind,:], rowVectors[ind,:])
        #colVectors[ind,0] = -np.sin(phi)*np.sin(theta)
        #colVectors[ind,1] = np.cos(phi)*np.sin(theta)
        #colVectors[ind,2] = np.cos(theta)
        
        sourcePositions[ind,:] *= sod
        
        ind += 1

print(sourcePositions-moduleCenters)
leapct.set_modularbeam(numAngles, numRows, numCols, pixelSize, pixelSize, sourcePositions, moduleCenters, rowVectors, colVectors)

# Set the volume parameters
leapct.set_default_volume()


# Trouble-Shooting Functions
leapct.print_parameters()
#leapct.sketch_system()
#quit()

# Allocate space for the projections and the volume
g = leapct.allocateProjections()
f = leapct.allocateVolume()

# Specify simplified FORBILD head phantom
leapct.set_FORBILD(f,True)
#leapct.display(f)


# "Simulate" projection data
startTime = time.time()
leapct.project(g,f)
print('Forward Projection Elapsed Time: ' + str(time.time()-startTime))
#leapct.display(g)
#quit()


# Reset the volume array to zero, otherwise iterative reconstruction algorithm will start their iterations
# with the true result which is cheating
f[:] = 0.0

# Reconstruct the data
# This is a lot of iterations but should take us close to full convergence
startTime = time.time()
leapct.ASDPOCS(g,f,400,15,5,1.0/20.0)
print('Reconstruction Elapsed Time: ' + str(time.time()-startTime))

leapct.display(f)
