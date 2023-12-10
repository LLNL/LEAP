import sys
import os
import time
import numpy as np
from leapctype import *
leapct = tomographicModels()

# Specify the number of detector columns which is used below
# Scale the number of angles and the detector pixel size with N
numCols = 512//2
numAngles = 2*int(360*numCols/1024)
pixelSize = 0.65*512/numCols

# Set the number of detector rows
numRows = numCols

# Make this modular-beam geometry just like a cone-beam dataset
# so let's define sod and sdd when defining our geometry
# In general, if your data fits into one of the standard geometry types,
# such as, parallel-, fan-, or cone-beam, then it is STRONGLY recommended
# that you use the standard type.
# The modular-beam projectors are not as fast and not as accurate
sod = 1100.0
sdd = 1400.0

# Set the scanner geometry
sourcePositions = np.ascontiguousarray(np.zeros((numAngles,3)).astype(np.float32), dtype=np.float32)
moduleCenters = np.ascontiguousarray(np.zeros((numAngles,3)).astype(np.float32), dtype=np.float32)
colVectors = np.ascontiguousarray(np.zeros((numAngles,3)).astype(np.float32), dtype=np.float32)
rowVectors = np.ascontiguousarray(np.zeros((numAngles,3)).astype(np.float32), dtype=np.float32)

T_phi = 2.0*np.pi/float(numAngles)
for n in range(numAngles):
    phi = n*T_phi-0.5*np.pi
    
    sourcePositions[n,0] = sod*np.cos(phi)
    sourcePositions[n,1] = sod*np.sin(phi)
    
    moduleCenters[n,0] = (sod-sdd)*np.cos(phi)
    moduleCenters[n,1] = (sod-sdd)*np.sin(phi)
    
    rowVectors[n,2] = 1.0
    
    colVectors[n,0] = -np.sin(phi)
    colVectors[n,1] = np.cos(phi)

leapct.set_modularbeam(numAngles, numRows, numCols, pixelSize, pixelSize, sourcePositions, moduleCenters, rowVectors, colVectors)

# Set the volume parameters
scale = 1.0
leapct.set_volume(int(numCols/scale),int(numCols/scale),int(numCols/scale),sod/sdd*pixelSize*scale,sod/sdd*pixelSize*scale)

# Trouble-Shooting Functions
leapct.print_parameters()
#leapct.sketch_system(0)
#quit()

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
g[:] = -np.log(np.random.poisson(I_0*np.exp(-g))/I_0)

# Reset the volume array to zero, otherwise iterative reconstruction algorithm will start their iterations
# with the true result which is cheating
f[:] = 0.0

# Reconstruct the data
startTime = time.time()
#leapct.backproject(g,f)
#leapct.ASDPOCS(g,f,10,1,4,1.0/20.0)
#leapct.SART(g,f,10)
#leapct.MLEM(g,f,10)
leapct.LS(g,f,20,True)
print('Reconstruction Elapsed Time: ' + str(time.time()-startTime))


# Post Reconstruction Smoothing (optional)
#startTime = time.time()
#leapct.diffuse(f,0.02/20.0,4)
#leapct.MedianFilter(f)
#leapct.BlurFilter(f,2.0)
#print('Post-Processing Elapsed Time: ' + str(time.time()-startTime))

# Display the result with napari
leapct.display(f)


    