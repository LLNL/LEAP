import sys
import os
import time
import numpy as np
from leapctype import *
leapct = tomographicModels()

'''
This demo script provides examples of cropping your data.  Cropping your projections may be necessary to remove
detector edge pixels that are bad or crop to the detector down to only the region of interest.  One may also
need to crop your data to processes it in smaller chunks so that you don't run out of CPU memory.

Of course you can crop the data yourself, but the advantage of using LEAP's cropping utility functions is that
they automatically update the CT geometry specification for you so you don't have to worry about how these parameters
are effected by this cropping.  For example if you crop the detector columns more from one side than another, LEAP will
update the "centerCol" parameter (or the moduleCenters parameter for modular-beam data) accordingly
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
f_true = leapct.allocateVolume()

# Specify simplified FORBILD head phantom
# One could easily do this in Python, but Python is soooooo slow for these types of operations,
# so we implemented this feature with multi-threaded C++
leapct.set_FORBILD(f_true, True)
#leapct.display(f_true)


# "Simulate" projection data
startTime = time.time()
leapct.project(g, f_true)
print('Forward Projection Elapsed Time: ' + str(time.time()-startTime))

whichDemo = 2
if whichDemo == 1:
    # In this first demo, we crop out the top half of the detector and then perform an FBP reconstruction.
    # This involves changing the numRows parameter and either the centerRow parameter (if parallel-, fan-, or cone-beam)
    # or the moduleCenters parameter (modular-beam) and then cropping the data array (numpy array or torch tensor)
    # We recommend that you use the cropProjections function for all of this because it properly updates all necessary parameters.
    # The cropProjection function takes three arguments, the first two being the range of detector rows indices and detector column
    # indices that you want to keep, respectively.  This third optional argument is the input projection data.  If this is given,
    # then a new array is produced by cropping the original.
    g=leapct.cropProjections([250, 512-1-10], None, g)
    leapct.set_default_volume()
    leapct.print_parameters()
    #quit()

    f = leapct.FBP(g)
    leapct.display(f)
elif whichDemo == 2:
    # In this second demo, we show how to crop the projection data down to only the rows necessary to reconstruction a subvolume.
    # This procedure is essential to processing large data sets where you don't have sufficient CPU memory to perform a certain operation.
    # Here we start with the full data projection size and crop it down, but for some applications you will run out of memory if you do
    # this, so you'll need to have the data saved to disk and then just read off the section that you need
    
    z = leapct.z_samples()
    chunkSize = leapct.get_numZ()//4
    numChunks = leapct.get_numZ()//chunkSize

    leapct_chunk = tomographicModels()

    for n in range(numChunks):
        print('*************************************************************')
        leapct_chunk.copy_parameters(leapct)
    
        sliceStart = n*chunkSize
        sliceEnd = min(z.size-1, sliceStart + chunkSize - 1)
        numZ = sliceEnd - sliceStart + 1
        if numZ <= 0:
            break

        # Set the number of z-slices for this chunk and the shift the volume to the next slab location        
        leapct_chunk.set_numZ(numZ)
        leapct_chunk.set_offsetZ(leapct_chunk.get_offsetZ() + z[sliceStart]-leapct_chunk.get_z0())
        
        # This function determines the detector row indices that you need to perform reconstruction with the
        # CT volume parameters that are currently defined.  We will use this to determine how to crop the projections
        rowRange = leapct_chunk.rowRangeNeededForBackprojection()
        
        g_chunk=leapct_chunk.cropProjections(rowRange, None, g)
        leapct_chunk.print_parameters()
        #quit()

        f_chunk = leapct_chunk.FBP(g_chunk)
        leapct_chunk.display(f_chunk)
        #'''
        