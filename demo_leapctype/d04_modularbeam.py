import sys
import os
import time
import numpy as np
from leapctype import *
leapct = tomographicModels()

'''
This script demonstrates how to use LEAP's "modular-beam" geometry
This geometry type is similar to ASTRA's cone_vec data type or the flexible geometries allowed by TIGRE
One difference in LEAP is that our cone-beam geometry is much, much more flexible that ASTRA's cone geometry
Thus, most cone-beam geometries should be covered by LEAP's standard cone-beam geometry which is much more
simple to specify and the algorithms (e.g., forward and back projectors) are faster

LEAP's modular-beam geometry works by the user specifying the location and orientation of every source and detector pair
These can be anywhere in space, but if the vector along the detector columns for all projections is within 5 degrees of
the positive z axis, then one can perform analytic, i.e., FBP algorithms to reconstruct the data
One will know if their geometry falls into this category by running the leapct.print_parameters() command
If the output includes the statement "axially aligned", then it can be reconstructed with FBP, otherwise
one will have to use iterative reconstruction algorithms.

Thus the major difference between cone-beam FBP and modular-beam (axially aligned) FBP is that modular-beam
enables detector rotations around the optical axis

The example below is of "axially aligned" type as it is just a generic cone-beam, but we specify it this way to
demonstrate how it works.

Another way to specify a modular-beam geometry that is just a perturbation of a cone-beam geometry is to specify the
cone-beam geometry and then use the command: leapct.convert_to_modularbeam() and then one can get the modular-beam parameters like this:
sourcePositions = self.get_sourcePositions()
moduleCenters = self.get_moduleCenters()
rowVecs = self.get_rowVectors()
colVecs = self.get_colVectors()
and then one can manipulate these numpy arrays and then run the command: leapct.set_modularbeam(...)

If one wishes to specify a flexible geometry with parallel rays, just put the source very far away from the object
'''

# Specify the number of detector columns which is used below
# Scale the number of angles and the detector pixel size with N
numCols = 512
numAngles = 2*2*int(360*numCols/1024)
pixelSize = 0.65*512/numCols

# Set the number of detector rows
numRows = numCols
#numRows = 1

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
leapct.set_default_volume()

# The next line is optional.  It sets the diameter of the circular field of view mask on the reconstruction volume
leapct.set_diameterFOV(leapct.get_numX()*leapct.get_voxelWidth())

# Trouble-Shooting Functions
leapct.print_parameters()
#leapct.sketch_system(0)

# Allocate space for the projections and the volume
g = leapct.allocateProjections()
f = leapct.allocateVolume()

# Specify simplified FORBILD head phantom
# One could easily do this in Python, but Python is soooooo slow for these types of operations,
# so we implemented this feature with multi-threaded C++
leapct.set_FORBILD(f,True)
#leapct.display(f)

#leapct.set_gpu(0)

# "Simulate" projection data
startTime = time.time()
leapct.project(g,f)
print('Forward Projection Elapsed Time: ' + str(time.time()-startTime))
#leapct.display(g)
#quit()

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
#leapct.inconsistencyReconstruction(g,f)
#leapct.print_cost = True
filters = filterSequence(1.0e0)
filters.append(TV(leapct, delta=0.02/20.0))
#leapct.ASDPOCS(g,f,10,10,1,filters)
#leapct.SART(g,f,10,10)
#leapct.OSEM(g,f,10,10)
#leapct.LS(g,f,50,'SQS')
#leapct.RWLS(g,f,100,filters,None,'SQS')
#leapct.RDLS(g,f,100,filters,1.0,True,1)
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


''' Compare to cone-beam
ct2 = tomographicModels()
ct2.set_conebeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0), 1100, 1400)
ct2.set_default_volume()
f_cone = ct2.allocateVolume()
ct2.FBP(g,f_cone)
leapct.display(f_cone)
leapct.display(f_cone-f)
quit()
#'''
