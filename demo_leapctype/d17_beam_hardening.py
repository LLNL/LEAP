import sys
import os
import time
import numpy as np
from leapctype import *
leapct = tomographicModels()
leapct.about()

'''
This script demonstrates how to perform single-material beam hardening correction (BHC)
The methodology we use to perform multi-material BHC in this paper:
https://www.osti.gov/servlets/purl/1158895
'''

# Specify the number of detector columns which is used below
# Scale the number of angles and the detector pixel size with N
numCols = 512
numAngles = 2*2*int(360*numCols/1024)
pixelSize = 0.65*512/numCols

# Set the number of detector rows
numRows = 64

# Set the scanner geometry
leapct.set_conebeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1)+10, leapct.setAngleArray(numAngles, 360.0), 1100, 1400)

# Set the volume parameters.
# It is best to do this after the CT geometry is set
leapct.set_default_volume()

# If you want to specify the volume yourself, use this function:
#leapct.set_volume(numX, numY, numZ, voxelWidth=None, voxelHeight=None, offsetX=None, offsetY=None, offsetZ=None):


# Allocate space for the projections and the volume
g = leapct.allocate_projections()

# Specify simplified FORBILD head phantom
material = 'water'
leapct.addObject(None, 4, np.array([0.0, 0.0, 0.0]), 100.0*np.array([1.0, 1.0, 1.0]), material)

# LEAP uses mm for all length-based units and thus for densities it uses g/mm^3
# *** Note that g/cm^3 = 1.0e-3 g/mm^3 ***
# So just add "e-3" to the end of the densities so that they are expressed in g/mm^3

### Source Spectra Modeling
# This is done with the XrayPhysics package.  Note that the units in this
# package are in cm, so be careful!
# Define the kV of the source voltage and the take-off angle (degrees)
kV = 80.0
takeOffAngle = 11.0

# First simulate the source spectrum (units are photons/(bin * mAs * sr))
Es, s = leapct.simulateSpectra(kV,takeOffAngle)

# Then model the detector response as the product of the
# x-ray energy and the stopping power of the scintillator
# Here we assume the scintillator is GOS, 0.1 mm thick, and density of 7.32 g/cm^3
detResp = leapct.detectorResponse('O2SGd2', 7.32e-3, 0.1, Es)

# Finally model the attenuation due to the filters
filtResp = leapct.filterResponse('Al', 2.7e-3, 0.1, Es)

# Take the product of all three factors
s_total = s * detResp * filtResp

# Beam Hardening Correction essentially synthesizes monochromatic projections
# from the measured polychromatic projections.
# LEAP allows you to choose the output energy of this transformation
# as long as it is in the range of the original spectrum
# Here we shall just use the mean energy of the spectra
referenceEnergy = leapct.meanEnergy(s_total, Es)

# Perform polychromatic simulation
leapct.rayTrace(g, s_total, Es)

# The code below will remove the beam hardening, i.e., Beam Hardening Correction (BHC)
BHC_LUT, T_lut = leapct.setBHClookupTable(s_total, Es, material, referenceEnergy)
leapct.applyTransferFunction(g, BHC_LUT, T_lut)

# Reconstruct the data
f = leapct.FBP(g)

# Display the result with napari
leapct.display(f)
