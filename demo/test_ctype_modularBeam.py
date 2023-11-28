import sys
import os
import time
import numpy as np
from leapctype import *
leapct = tomographicModels()

#'''
N = 300
numAngles = int(720*N/1024)
pixelSize = 0.2*2048/N

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

leapct.set_modularBeam(numAngles, N, N, pixelSize, pixelSize, sourcePositions, moduleCenters, rowVectors, colVectors)

# Set the volume parameters
leapct.set_volume(N,N,N,sod/sdd*pixelSize,sod/sdd*pixelSize)

# Trouble-Shooting Functions
#leapct.printParameters()
#leapct.sketchSystem(0)
#quit()

# Allocate space for the projections and the volume
g = leapct.allocateProjections()
f = leapct.allocateVolume()

# Specify a phantom composed of a 300 mm diameter sphere
f_true = leapct.allocateVolume()
x,y,z=leapct.voxelSamples()
f_true[x**2 + y**2 + (z/1.05)**2 <= 150.0**2] = 1.0
f_true[x**2 + (y-20)**2 + z**2 <= 10.0**2] = 0.0
#f_true[x**2 + (y-20)**2 + z**2 <= 10.0**2] = 1.0

# "Simulate" projection data
startTime = time.time()
leapct.project(g,f_true)
print('Elapsed time: ' + str(time.time()-startTime))
#leapct.displayVolume(g)

startTime = time.time()
#leapct.backproject(g,f)
#leapct.set_diameterFOV(sod/sdd*pixelSize*N)
leapct.SART(g,f,100)
print('Elapsed time: ' + str(time.time()-startTime))

leapct.displayVolume(f)
quit()
#'''


'''
numAngles = 10
N = 300
pixelWidth = 0.9
pixelHeight = 1.0
N = 100
M = 400

sod = 250.0
sdd = 500

sourcePositions = np.ascontiguousarray(np.zeros((numAngles,3)).astype(np.float32), dtype=np.float32)
moduleCenters = np.ascontiguousarray(np.zeros((numAngles,3)).astype(np.float32), dtype=np.float32)
colVectors = np.ascontiguousarray(np.zeros((numAngles,3)).astype(np.float32), dtype=np.float32)
rowVectors = np.ascontiguousarray(np.zeros((numAngles,3)).astype(np.float32), dtype=np.float32)

for n in range(numAngles//2):
    phi = 0.0*np.pi/2.0
    m = n-(numAngles//2-1)/2.0
        
    sourcePositions[n,0] = -sdd*np.cos(phi)
    sourcePositions[n,1] = -sdd*np.sin(phi)
    
    moduleCenters[n,0] = sod*np.cos(phi)
    moduleCenters[n,1] = sod*np.sin(phi) + N*m
    
    rowVectors[n,2] = 1.0
    
    colVectors[n,0] = -np.sin(phi)
    colVectors[n,1] = np.cos(phi)
    
for n in range(numAngles//2,numAngles):
    phi = 1.0*np.pi/2.0
    m = n-numAngles//2-(numAngles//2-1)/2.0
        
    sourcePositions[n,0] = -sdd*np.cos(phi)
    sourcePositions[n,1] = -sdd*np.sin(phi)
    
    moduleCenters[n,0] = sod*np.cos(phi)+ N*m
    moduleCenters[n,1] = sod*np.sin(phi)
    
    rowVectors[n,2] = 1.0
    
    colVectors[n,0] = -np.sin(phi)
    colVectors[n,1] = np.cos(phi)
    
leapct.set_modularBeam(numAngles, M, N, pixelHeight, pixelWidth, sourcePositions, moduleCenters, rowVectors, colVectors)
leapct.set_volume(5*N,5*N,M,0.5*pixelHeight,0.5*pixelHeight)
#'''


leapct.sketchSystem()


    