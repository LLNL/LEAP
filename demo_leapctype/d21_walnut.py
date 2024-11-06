import sys
import os
import time
import numpy as np
from leapctype import *
leapct = tomographicModels()

'''
This demo script shows you how to reconstruct the publically available CT data sets of walnuts (see link below).
This is the script that was used to generate the example result on the main page of the LEAP repo.
We chose the reconstruction volume size to match what they did in the ASTRA script for comparison purposes, but
we find their choices of parameters strange.  There are unnecessary voxels added to the field of view and a lot of
"air" is reconstructed and the voxel sizes are actually quite large and a better resolution reconstruction could have
been performed.

Anyway, we chose this to match what they did with ASTRA and hopefully this helps others that are familiar with ASTRA
to use LEAP instead
'''

# Data downloaded from: https://zenodo.org/records/6986012
# ASTRA scripts found here: https://github.com/cicwi/WalnutReconstructionCodes

# Set the folder name where the data was downloaded
dataPath = r'C:\Users\champley\Downloads\Walnut17\Walnut17\Projections\tubeV2'


# Read geometry file
geom = np.genfromtxt(os.path.join(dataPath,'scan_geom_corrected.geom'), dtype=np.float32)


# Set the LEAP modular-beam source and detector locations
numAngles = geom.shape[0]
sourcePositions = np.zeros((numAngles,3),dtype=np.float32)
moduleCenters = np.zeros((numAngles,3),dtype=np.float32)
colVectors = np.zeros((numAngles,3),dtype=np.float32)
rowVectors = np.zeros((numAngles,3),dtype=np.float32)

sourcePositions[:,:] = geom[:,0:3]
moduleCenters[:,:] = geom[:,3:6]
colVectors[:,:] = geom[:,6:9]
rowVectors[:,:] = geom[:,9:12]

sourcePositions[:,2] *= -1.0
moduleCenters[:,2] *= -1.0
#colVectors[:,2] *= -1.0
#rowVectors[:,2] *= -1.0


# Set LEAP CT geometry parameters
pixelSize = 0.149600
leapct.set_modularbeam(numAngles, 972, 768, pixelSize, pixelSize, sourcePositions, moduleCenters, rowVectors, colVectors)
#leapct.set_conebeam(numAngles, 972, 768, pixelSize, pixelSize, 972/2, 768/2, leapct.setAngleArray(numAngles, 360.0), 66.010880, 199.011551)

# Set LEAP CT volume parameters
#leapct.set_default_volume()
leapct.set_volume(501, 501, 501, 0.1, 0.1)
leapct.print_parameters()


# Read data
import imageio
drk = np.array(imageio.imread(os.path.join(dataPath,'di000000.tif')))
bak = np.array(imageio.imread(os.path.join(dataPath,'io000000.tif')))
bak2 = np.array(imageio.imread(os.path.join(dataPath,'io000001.tif')))
bak = 0.5*(bak+bak2)
g = leapct.load_projections(os.path.join(dataPath,'scan.tif'))


# Perform Flat Fielding
print('Flat Fielding...')
g = (g - drk) / (bak - drk)
#g[g<2.0**-16] = 2.0**-16
g = -np.log(g)


# Transpose and flip data
g = np.transpose(g, (0,2,1))
g = np.flip(g,axis=1)
g = np.ascontiguousarray(g, dtype=np.float32)


# Set some reconstruction parameters
# These are optional
leapct.set_diameterFOV(0.049620*768)
leapct.set_rampFilter(10)


# Do FDK reconstruction
# If we set inplaceProcessing to true, the filtering operations in FBP are
# done in-place.  This runs faster, but you are not able to recover the 
# original projection data.
#inplaceProcessing = True
#leapct.set_projector('VD')
f = leapct.allocate_volume()
print('FBP reconstruction...')
startTime = time.time()
leapct.FBP(g,f)
#leapct.backproject(g,f)
print('FBP time: ' + str(time.time()-startTime))

# Flip the reconstruction so that it is aligned with the ASTRA reconstruction
f = np.flip(f,axis=0)
f = np.ascontiguousarray(np.flip(f,axis=2), dtype=np.float32)


# Display result with napari
#leapct.display(f)
#quit()


# Save results as a tif sequence
outputDir = os.path.join(dataPath,'FBP_eSF')
if not os.path.exists(outputDir):
    os.makedirs(outputDir)
leapct.save_volume(os.path.join(outputDir,'rec.tif'), f)
