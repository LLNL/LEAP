import sys
import os
import time
import numpy as np
from leapctype import *
leapct = tomographicModels()
try:
    from xrayphysics import *
    physics = xrayPhysics()
except:
    print('This demo script requires the XrayPhysics package found here:')
    print('https://github.com/kylechampley/XrayPhysics')
    quit()

# Specify the number of detector columns which is used below
# Scale the number of angles and the detector pixel size with N
numCols = 512
numAngles = 2*2*int(360*numCols/1024)
pixelSize = 0.65*512/numCols

# Set the number of detector rows
numRows = 64

# Set the scanner geometry
# Let's start with a standard cone-beam geometry so make it easier to specify,
# then convert it to modular beam for we can perform detector dithering
leapct.set_conebeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1)+10, leapct.setAngleArray(numAngles, 360.0), 1100, 1400)

# Set the volume parameters.
# It is best to do this after the CT geometry is set
leapct.set_default_volume()

# If you want to specify the volume yourself, use this function:
#leapct.set_volume(numX, numY, numZ, voxelWidth=None, voxelHeight=None, offsetX=None, offsetY=None, offsetZ=None):


# Allocate space for the projections and the volume
g = leapct.allocateProjections()
f_true = leapct.allocateVolume()


# Specify simplified FORBILD head phantom
leapct.addObject(f_true, 4, np.array([0.0, 0.0, 0.0]), 100.0*np.array([1.0, 1.0, 1.0]), 0.02)
#leapct.set_FORBILD(f_true,True)


# "Simulate" projection data
leapct.project(g,f_true)


### Source Spectra Modeling
# This is done with the XrayPhysics package.  Note that the units in this
# package are in cm, so be careful!
# Define the kV of the source voltage and the take-off angle (degrees)
kV = 80.0
takeOffAngle = 11.0

# First simulate the source spectrum (units are photons/(bin * mAs * sr))
Es, s = physics.simulateSpectra(kV,takeOffAngle)

# Then model the detector response as the product of the
# x-ray energy and the stopping power of the scintillator
# Here we assume the scintillator is GOS, 0.01 cm thick, and density of 7.32 g/cm^3
detResp = physics.detectorResponse('O2SGd2', 7.32, 0.01, Es)

# Finally model the attenuation due to the filters
# We are going to comment this out so we get more beam hardening
#filtResp = physics.filterResponse('Al', 2.7, 0.1, Es)

# Take the product of all three factors
s_total = s*detResp #*filtResp

# Generate the Beam Hardening (BH) lookup table
# which assumes the input is monochromatic attenuation at 63.9544 keV
# We chose this energy because the LAC of water is 0.02 mm^-1 which
# agrees with the value in the phantom
BH_LUT, T_lut = physics.setBHlookupTable('H2O', s_total, Es, 0.0, 0, 63.9544)

# Apply Beam Hardening
g_save = g.copy()
leapct.applyTransferFunction(g, BH_LUT, T_lut)

# The code below will remove the beam hardening, i.e., Beam Hardening Correction (BHC)
# Applying these steps should just undo the BH you applied above, so isn't too exciting
# in this demo, but for real data this can remove beam hardening
#BHC_LUT, T_lut = physics.setBHClookupTable('H2O', s_total, Es, 0.0, 0, 63.9544)
#leapct.applyTransferFunction(g, BHC_LUT, T_lut)

# Another thing we could do is change in output energy.  This will effectively
# change the scaling of the reconstruction.  Let's try with 40 keV which should
# get us a reconstructed value of about 0.2684
#BHC_LUT, T_lut = physics.setBHClookupTable('H2O', s_total, Es, 0.0, 0, 40.0)
#leapct.applyTransferFunction(g, BHC_LUT, T_lut)

# Reconstruct the data
f = leapct.FBP(g)


# Display the result with napari
leapct.display(f)
