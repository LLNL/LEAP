import sys
import os
import time
import numpy as np
from leapctype import *
leapct = tomographicModels()
# Make sure you add: .../LEAP/src to your python path


# Specify the number of detector rows and columns which is used below
# Scale the number of angles and the detector pixel size with N
N = 300
numAngles = int(720*N/1024)
pixelSize = 0.2*2048/N

# Set the scanner geometry
#leapct.setParallelBeamParams(numAngles, N, N, pixelSize, pixelSize, 0.5*(N-1), 0.5*(N-1), leapct.setAngleArray(numAngles, 360.0))
#leapct.setFanBeamParams(numAngles, N, N, pixelSize, pixelSize, 0.5*(N-1), 0.5*(N-1), leapct.setAngleArray(numAngles, 360.0), 1100, 1400)
leapct.setConeBeamParams(numAngles, N, N, pixelSize, pixelSize, 0.5*(N-1), 0.5*(N-1), leapct.setAngleArray(numAngles, 360.0), 1100, 1400)

# Set the volume parameters
leapct.setDefaultVolume()

# Allocate space for the projections and the volume
g = leapct.allocateProjections()
f = leapct.allocateVolume()

# Specify a phantom composed of a sphere
f_true = leapct.allocateVolume()
x,y,z=leapct.voxelSamples()
f_true[x**2 + y**2 + z**2 <= 150.0**2] = 1.0

# "Simulate" projection data
leapct.project(g,f_true)

# Reconstruct with FBP
leapct.FBP(g,f)

# Reconstruct with SART
#leapct.SART(g,f,50)

leapct.displayVolume(f)
