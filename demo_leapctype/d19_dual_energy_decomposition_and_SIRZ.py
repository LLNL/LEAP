import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from leapctype import *
leapct = tomographicModels()
from xrayphysics import *
physics = xrayPhysics()

'''
This script demonstrates how to perform dual energy decomposition and requires the XrayPhysics package
which can be found here: https://github.com/kylechampley/XrayPhysics

For a full explanation of how this works, it is helpful to read the example script in the XrayPhysics package here:
https://github.com/kylechampley/XrayPhysics/blob/main/demo/dual_energy_decomp.py

And the methodology we use to perform dual energy decomposition is described in this paper:
https://ieeexplore.ieee.org/abstract/document/8638824?casa_token=K_9cFGKJGvMAAAAA:EzTpZfY0qJHMvdxGniguZBS_dATpx-4vqhsDPZwB1VFh02loJFD0hvizr5RNKj5z5xgvU8Iq8g
'''

# Set the CT geometry and CT volume
numCols = 512
numAngles = 2*2*int(360*numCols/1024)
pixelSize = 0.65*512/numCols
numRows = 1
leapct.set_fanbeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0), 1100, 1400)
leapct.set_default_volume()

# Tell the XrayPhysics library to use mm-based units so that everything agrees with LEAP which is mm-based
# Note that it is natural to express different quantities with different units, e.g., mm or cm
# But to avoid confusion of which parameter uses which units, everything should use the same units
# This should be fine for most things, but one thing to look out for is densities
# *** Note that g/cm^3 = 1.0e-3 g/mm^3 ***
# So just add "e-3" to the end of the densities so that they are expressed in g/mm^3
physics.use_mm()


#######################################################################################################################
# Now we define the total system spectral response for each of the two energies
#######################################################################################################################

# Define the kV of the source voltage and the take-off angle (degrees)
kV_L = 100.0
kV_H = 160.0
takeOffAngle = 11.0

# First simulate the source spectrum (units are photons/(bin * mAs * sr))
Es, s_H = physics.simulateSpectra(kV_H,takeOffAngle)
Es, s_L = physics.simulateSpectra(kV_L,takeOffAngle, None, Es)

# Then model the detector response as the product of the
# x-ray energy and the stopping power of the scintillator
# Here we assume the scintillator is GOS, 0.1 mm thick, and density of 7.32 g/cm^3
detResp = physics.detectorResponse('O2SGd2', 7.32e-3, 0.1, Es)

# Finally model the attenuation due to the filters
filtResp_L = physics.filterResponse('Al', 2.7e-3, 1.0, Es) # 1.0 mm of Al
filtResp_H = physics.filterResponse('Cu',  8.96e-3, 1.0, Es) # 1.0 mm of Cu
filtResp_H[:] = filtResp_H[:]*filtResp_L[:]

# Take the product of all three factors
s_L = s_L*filtResp_L*detResp
s_H = s_H*filtResp_H*detResp


#######################################################################################################################
# Now we calculate the dual energy decomposition lookup table transfer function
# This only depends on the spectra models and the basis functions one chooses and is independent of CT geometry
# Thus these tables can be saved to disk for repeated use
# Below we provide examples using a basis set composed of two materials or the so-called "Compton-Photoelectric basis"
#######################################################################################################################
sigma_water = physics.sigma('H2O', Es)
sigma_Al = physics.sigma('Al', Es)

# Calculate the PCA bases from C, N, O, and Al (as is done in the SIRZ paper)
PCA_1, PCA_2 = physics.PCAbases(['C','N','O','Al'], Es)

# Now let's choose the reference energies of the decomposition.  The user is free to choose what they want,
# but we recommend that these energies be within the energy range of the spectra and the low energy
# reference energy be lower than the high energy reference energy.
# Here we will just use the mean energies of each spectra as the reference energy.
# This is what would be chosen if these parameters were not specified (i.e., the default value)
# Although not necessary, we round these values to the nearest whole number
referenceEnergy_L = np.round(physics.meanEnergy(s_L, Es))
referenceEnergy_H = np.round(physics.meanEnergy(s_H, Es))

startTime = time.time()
#LUT,T_atten = physics.setDEDlookupTable(s_L, s_H, Es, sigma_water, sigma_Al, [referenceEnergy_L, referenceEnergy_H])
#LUT,T_atten = physics.setDEDlookupTable(s_L, s_H, Es, physics.PhotoelectricBasis(Es), physics.ComptonBasis(Es), [referenceEnergy_L, referenceEnergy_H])
LUT,T_atten = physics.setDEDlookupTable(s_L, s_H, Es, PCA_1, PCA_2, [referenceEnergy_L, referenceEnergy_H])
print('DED LUT generation time: ' + str(time.time()-startTime) + ' seconds')


#######################################################################################################################
# Simulate dual energy data
#######################################################################################################################
g_L = leapct.allocate_projections() # polychromatic low energy attenuation data
g_H = leapct.allocate_projections() # polychromatic high energy attenuation data

g_water = leapct.allocate_projections() # forward projection of water density map
g_Al = leapct.allocate_projections() # forward projection of aluminum density map
f_water = leapct.allocate_volume() # water density map
f_Al = leapct.allocate_volume() # aluminum density map

# Set the aluminum map as two cylinders
leapct.addObject(f_Al, 4, 50.0*np.array([1.0, 0.0, 0.0]), 30.0*np.array([1.0, 1.0, 1.0]), 1.0, None, None, 3)
leapct.addObject(f_Al, 4, 50.0*np.array([-1.0, 0.0, 0.0]), 30.0*np.array([1.0, 1.0, 1.0]), 1.0, None, None, 3)

# Set the water map as one big cylinder.  Need to subtract out region that aluminum occupies
leapct.addObject(f_water, 4, np.array([0.0, 0.0, 0.0]), 120.0*np.array([1.0, 1.0, 1.0]), 1.0, None, None, 3)
f_water = f_water - f_Al

# Now we scale the water and aluminum maps by the water and aluminum densities, respectively
# Note that the density of water is 1.0 g/cm^3 and the density of aluminum is 2.7 g/cm^3
# But LEAP is in units of mm, so this means the densities must be divided by 1000
# This is why you see "e-3" after each density value
f_water *= 1.0e-3
f_Al *= 2.7e-3

# Now forward project the density maps
leapct.project(g_water, f_water)
leapct.project(g_Al, f_Al)

# Now we calculate the polychromatic attenuation of the low and high energy spectra
physics.normalizeSpectrum(s_L, Es)
physics.normalizeSpectrum(s_H, Es)
for n in range(Es.size):
    if s_L[n] > 0.0:
        g_L[:] += s_L[n]*np.exp(-sigma_water[n]*g_water[:] - sigma_Al[n]*g_Al[:])
    if s_H[n] > 0.0:
        g_H[:] += s_H[n]*np.exp(-sigma_water[n]*g_water[:] - sigma_Al[n]*g_Al[:])
g_L = -np.log(g_L)
g_H = -np.log(g_H)


#######################################################################################################################
# Apply dual energy decomposition transfer function and reconstruct
# The reconstructions will be in LAC units at the reference energies specified above
# If you comment out the "applyDualTransferFunction" function, you should see beam hardening artifacts 
#######################################################################################################################
startTime = time.time()
leapct.applyDualTransferFunction(g_L, g_H,  LUT, T_atten)
print('DED time: ' + str(time.time()-startTime) + ' seconds')

f_L = leapct.FBP(g_L)
f_H = leapct.FBP(g_H)
#leapct.display(f_L)
#leapct.display(f_H)

# Now let's convert to rhoe and Ze (electron density and effective atomic number)
sigmae_L = np.zeros((100),dtype=np.float32)
sigmae_H = sigmae_L.copy()
for n in range(100):
    Z = n+1
    sigmae_L[n] = physics.sigma_e(Z, referenceEnergy_L)
    sigmae_H[n] = physics.sigma_e(Z, referenceEnergy_H)
    
f_Ze, f_rhoe = leapct.convertToRhoeZe(f_L, f_H, sigmae_L, sigmae_H)
leapct.display(f_Ze) # Ze volume
leapct.display(f_rhoe) # rhoe volume
