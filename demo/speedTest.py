import sys
import os
import time
import numpy as np
from leapctype import *
leapct = tomographicModels()

# Specify the number of detector rows and columns which is used below
N = 512
#N = 800

# Scale the number of angles and the detector pixel size with N
numAngles = int(720*N/1024)
pixelSize = 0.2*2048/N

# Set the scanner geometry
#leapct.setParallelBeamParams(numAngles, N, N, pixelSize, pixelSize, 0.5*(N-1), 0.5*(N-1), leapct.setAngleArray(numAngles, 360.0))
leapct.setFanBeamParams(numAngles, N, N, pixelSize, pixelSize, 0.5*(N-1), 0.5*(N-1), leapct.setAngleArray(numAngles, 360.0), 1100, 1400)
#leapct.setConeBeamParams(numAngles, N, N, pixelSize, pixelSize, 0.5*(N-1), 0.5*(N-1), leapct.setAngleArray(numAngles, 360.0), 1100, 1400)

# Set the volume parameters
leapct.setDefaultVolume()

# Allocate space for the projections and the volume
g = leapct.allocateProjections()
f = leapct.allocateVolume()

# Project
startTime = time.time()
leapct.project(g,f)
print('projection time: ' + str(time.time()-startTime))

# Backproject
startTime = time.time()
leapct.backproject(g,f)
print('backprojection time: ' + str(time.time()-startTime))

# FBP
startTime = time.time()
leapct.FBP(g,f)
print('FBP time: ' + str(time.time()-startTime))
