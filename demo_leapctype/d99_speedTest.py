import sys
import os
import time
import numpy as np
from leapctype import *
leapct = tomographicModels()

'''
The only purpose of this demo script is to evaluate the speed of LEAP's forward projections, backprojection, and FBP algorithms.
'''

# Specify the number of detector rows and columns which is used below
N = 1024
#N = 1300
M = N
#M = 1024
#M = 512+128
#N = 800
#leapct.set_GPU(0)

# Scale the number of angles and the detector pixel size with N
numAngles = int(720*N/1024)
pixelSize = 0.2*2048/N

# Set the scanner geometry
#leapct.set_parallelbeam(numAngles, M, N, pixelSize, pixelSize, 0.5*(M-1), 0.5*(N-1), leapct.setAngleArray(numAngles, 360.0))
#leapct.set_fanbeam(numAngles, M, N, pixelSize, pixelSize, 0.5*(M-1), 0.5*(N-1), leapct.setAngleArray(numAngles, 360.0), 1100, 1400)
leapct.set_conebeam(numAngles, M, N, pixelSize, pixelSize, 0.5*(M-1), 0.5*(N-1), leapct.setAngleArray(numAngles, 360.0), 1100, 1400, tiltAngle=0.2)

# Set the volume parameters
leapct.set_default_volume()
leapct.print_parameters()

# Set the backprojector model, 'SF' (the default setting), is more accurate, but 'VD' is faster
leapct.set_projector('VD')
#leapct.convert_to_modularbeam()

# Allocate space for the projections and the volume
g = leapct.allocateProjections()
f = leapct.allocateVolume()

print('using ' + str(4.0*(g.size+f.size)/2.0**30.0) + ' GB of memory')

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

# Filter Projections
startTime = time.time()
leapct.filterProjections(g)
print('filter projections time: ' + str(time.time()-startTime))
