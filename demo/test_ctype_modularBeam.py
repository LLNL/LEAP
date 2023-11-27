import sys
import os
import time
import numpy as np
from leapctype import *
leapct = tomographicModels()

'''
numAngles = 10
N = 300
pixelSize = 0.2*2048/N

sod = 1000
sdd = 2000

sourcePositions = np.ascontiguousarray(np.zeros((numAngles,3)).astype(np.float32), dtype=np.float32)
moduleCenters = np.ascontiguousarray(np.zeros((numAngles,3)).astype(np.float32), dtype=np.float32)
colVectors = np.ascontiguousarray(np.zeros((numAngles,3)).astype(np.float32), dtype=np.float32)
rowVectors = np.ascontiguousarray(np.zeros((numAngles,3)).astype(np.float32), dtype=np.float32)

for n in range(numAngles):
    phi = n*np.pi/2.0
    
    sourcePositions[n,0] = -sod*np.cos(phi)
    sourcePositions[n,1] = -sod*np.sin(phi)
    
    moduleCenters[n,0] = (sdd-sod)*np.cos(phi)
    moduleCenters[n,1] = (sdd-sod)*np.sin(phi)
    
    rowVectors[n,2] = 1.0
    
    colVectors[n,0] = -np.sin(phi)
    colVectors[n,1] = np.cos(phi)
    
leapct.set_modularBeam(numAngles, N, N, pixelSize, pixelSize, sourcePositions, moduleCenters, rowVectors, colVectors)
leapct.set_volume(N,N,N,0.5*pixelSize,0.5*pixelSize)
#'''

#'''

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


    