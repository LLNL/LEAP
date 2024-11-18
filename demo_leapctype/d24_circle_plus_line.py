################################################################################
# This demo script performs a circle+line simulation and iterative reconstruction
# Note that it is an iterative reconstruction and not the circle+line
# analytic reconstruction; this analytic reconstruction method is not implemented in LEAP.
# The purpose of collecting data of this type is to mitigate cone-beam artifacts.
################################################################################
import sys
import os
import time
import numpy as np
from leapctype import *

'''
This script demonstrates the so-called circle+line trajectory where two cone-beam scans are performed
The first is a standard axial scan.
The second is a scan where the patient bed or turn table are continuously translated, but no rotations are performed.
The purpose of this line scan is to mitigate so-called cone-beam artifacts (which can also be mitigated by a helical scan).

Note that LEAP does not have an implementation of the analytic circle+line algorithm.  We solve this problem here by performing
an axial FBP (FDK) reconstruction followed by an iterative reconstruction refinement using the line scan.

This script also demonstrates how to use multiple LEAP instances to track multiple CT acquisitions simultaneously
'''

ct_circular = tomographicModels()
ct_line = tomographicModels()


# Specify the number of detector columns which is used below
# Scale the number of angles and the detector pixel size with N
numCols = 512
numAngles = 2*2*int(360*numCols/1024)
pixelSize = 0.65*512/numCols
sod = 1100.0
sdd = 1400.0

# Set the number of detector rows
numRows = numCols

# Set the scanner geometry
# First we define the axial (circular) cone-beam data
ct_circular.set_conebeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1)+10, ct_circular.setAngleArray(numAngles, 360.0), sod, sdd)

# Now we define the line cone-beam data which requires that we use the modular-beam geometry
L = 16
#L = 32
numShifts = numRows//L
shiftSpacing = pixelSize*L
sourcePositions = np.ascontiguousarray(np.zeros((numShifts,3)).astype(np.float32), dtype=np.float32)
moduleCenters = np.ascontiguousarray(np.zeros((numShifts,3)).astype(np.float32), dtype=np.float32)
colVectors = np.ascontiguousarray(np.zeros((numShifts,3)).astype(np.float32), dtype=np.float32)
rowVectors = np.ascontiguousarray(np.zeros((numShifts,3)).astype(np.float32), dtype=np.float32)

for n in range(numShifts):
    phi = -0.5*np.pi
    
    sourcePositions[n,0] = sod*np.cos(phi)
    sourcePositions[n,1] = sod*np.sin(phi)
    sourcePositions[n,2] = (n-0.5*(numShifts-1))*shiftSpacing
    
    moduleCenters[n,0] = (sod-sdd)*np.cos(phi)
    moduleCenters[n,1] = (sod-sdd)*np.sin(phi)
    moduleCenters[n,2] = (n-0.5*(numShifts-1))*shiftSpacing
    
    rowVectors[n,2] = 1.0
    
    colVectors[n,0] = -np.sin(phi)
    colVectors[n,1] = np.cos(phi)
ct_line.set_modularbeam(numShifts, numRows, numCols, pixelSize, pixelSize, sourcePositions, moduleCenters, rowVectors, colVectors)


# Set the volume parameters.
# It is best to do this after the CT geometry is set
ct_circular.set_default_volume()
ct_line.set_volume(ct_circular.get_numX(), ct_circular.get_numY(), ct_circular.get_numZ(), ct_circular.get_voxelWidth(), ct_circular.get_voxelHeight())

# If you want to specify the volume yourself, use this function:
#ct_circular.set_volume(numX, numY, numZ, voxelWidth=None, voxelHeight=None, offsetX=None, offsetY=None, offsetZ=None):


# Allocate space for the projections and the volume
g_circular = ct_circular.allocateProjections()
g_line = ct_line.allocateProjections()
f_true = ct_circular.allocateVolume()


# Specify simplified FORBILD head phantom
#ct_circular.set_FORBILD(f_true,True)
for i in range(11):
    ct_circular.addObject(f_true, 4, [0.0, 0.0, 20.0*(i-5)], [100.0, 100.0, 5.0], 0.02)
#ct_circular.display(f_true)
#quit()

# "Simulate" projection data
ct_circular.project(g_circular,f_true)
ct_line.project(g_line,f_true)


f = ct_circular.allocateVolume()

# Reconstruct the axial cone-beam data with FBP (FDK)
ct_circular.FBP(g_circular, f)
#ct_circular.display(f)

# Now refine the reconstruction with the "line" CT data
# This reconstruction should have no cone-beam artifacts
# Try commenting out this line to see cone-beam artfiact in the reconstruction
ct_line.LS(g_line, f, 10)#, 'SQS', True)

# Display the result
ct_circular.display(f)
