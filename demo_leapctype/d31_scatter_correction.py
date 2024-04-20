import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from leapctype import *
leapct = tomographicModels()
from xrayphysics import *
physics = xrayPhysics()
leapct.about()


# Specify the number of detector columns which is used below
# Scale the number of angles and the detector pixel size with N
numCols = 512
numAngles = 2*2*int(360*numCols/1024)
pixelSize = 0.65*512/numCols

# Set the number of detector rows
numRows = numCols

# Set the scanner geometry
leapct.set_conebeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0), 1100, 1400)
#leapct.set_curvedDetector()

# Set the volume parameters.
# It is best to do this after the CT geometry is set
leapct.set_default_volume()

# If you want to specify the volume yourself, use this function:
#leapct.set_volume(numX, numY, numZ, voxelWidth=None, voxelHeight=None, offsetX=None, offsetY=None, offsetZ=None):

# Trouble-Shooting Functions
leapct.print_parameters()
#leapct.sketch_system()

# The next line tells the XrayPhysics library to use mm-based units so that everything agrees with LEAP which is mm-based
# Note that it is natural to express different quantities with different units, e.g., mm or cm
# But to avoid confusion of which parameter uses which units, everything should use the same units
# This should be fine for most things, but one thing to look out for is densities
# *** Note that g/cm^3 = 1.0e-3 g/mm^3 ***
# So just add "e-3" to the end of the densities so that they are expressed in g/mm^3
physics.use_mm()


#######################################################################################################################
# Now we define the total system spectral response model
# The XrayPhysics package provides methods to estimate this, but you can certainly use your own models
# The models in XrayPhysics are quite accurate, but for best results, one should perform a spectral calibration
#######################################################################################################################

# Define the kV of the source voltage and the take-off angle (degrees)
kV = 100.0
takeOffAngle = 11.0

# First simulate the source spectrum (units are photons/(bin * mAs * sr))
# "Es" are the energy samples (in keV) and "s" is the source spectrum
Es, s = physics.simulateSpectra(kV,takeOffAngle)

# Then model the detector response as the product of the
# x-ray energy and the stopping power of the scintillator
# Here the scintillator is 0.1 mm thick GOS with density 7.32 g/cm^3
detResp = physics.detectorResponse('O2SGd2', 7.32e-3, 0.1, Es)

# Finally model the attenuation due to the filters
# Here the filter is 1.0 mm thick aluminum
filtResp = physics.filterResponse('Al', 2.7e-3, 1.0, Es)

# Take the product of all three factors to get the total system spectral response
s = s*filtResp*detResp

Es_2 = Es[0:-1:10] + 0.5*Es[0]
print(Es_2)
s_2 = physics.resample(Es,s,Es_2)
print(physics.meanEnergy(s,Es))
print(physics.meanEnergy(s_2,Es_2))
plt.plot(Es,s,'k-',Es_2,(1.0/10.0)*s_2,'r-*')
plt.show()
quit()

# Scatter correction requires the detector response at energies from 1 keV, in 1 keV bins
Es_full = np.array(range(int(Es[-1])),dtype=np.float32)
detResp = physics.detectorResponse('O2SGd2', 7.32e-3, 0.1, Es_full)

sigma_water = physics.sigma('H2O',Es_full)
thetas = np.array(range(180+1),dtype=np.float32)
dsigma_incoh = physics.incoherentScatterDistribution('H2O', 60.0, thetas, doNormalize=True)
dsigma_coh = physics.coherentScatterDistribution('H2O', 60.0, thetas, doNormalize=True)
quit()

#######################################################################################################################
# Simulate data
#######################################################################################################################
# Specify simplified FORBILD head phantom
# One could easily do this in Python, but Python is soooooo slow for these types of operations,
# so we implemented this feature with multi-threaded C++
leapct.addObject(None, 4, np.array([0.0, 0.0, 0.0]), 120.0*np.array([1.0, 1.0, 0.5]), 0.02)
#leapct.set_FORBILD(f,True)
#leapct.display(f)


# "Simulate" projection data
g = leapct.allocateProjections() # shape is numAngles, numRows, numCols
leapct.rayTrace(g)

# Add noise to the data (just for demonstration purposes)
I_0 = 50000.0
#g[:] = -np.log(np.random.poisson(I_0*np.exp(-g))/I_0)

f = leapct.FBP(g)
device = torch.device("cuda:" + str(leapct.get_gpu()))
#f = torch.from_numpy(f).to(device)

#######################################################################################################################
# Down Sample
#######################################################################################################################
leapct_LR = tomographicModels()
leapct_LR.copy_parameters(leapct)

g_dn = leapct_LR.down_sample_projections([1,4,4],g)
f_dn = leapct_LR.down_sample_volume([4,4,4], f)

leapct.display(f_dn)
