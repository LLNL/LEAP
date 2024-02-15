import sys
import numpy as np
import time
#import matplotlib.pyplot as plt
from leapctype import *
leapct = tomographicModels()
from xrayphysics import *
physics = xrayPhysics()

'''
This script demonstrates how to perform multi-material beam hardening correction (BHC) and requires the XrayPhysics package
which can be found here: https://github.com/kylechampley/XrayPhysics
And the methodology we use to perform multi-material BHC in this paper:
https://www.osti.gov/servlets/purl/1158895
'''

# Set the CT geometry and CT volume
numCols = 512
numAngles = 2*2*int(360*numCols/1024)
pixelSize = 0.65*512/numCols
numRows = 1
leapct.set_fanbeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0), 1100, 1400)
leapct.set_default_volume()

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


#######################################################################################################################
# Now we calculate the multi-material BHC lookup table transfer function
# This only depends on the spectra models and the materials one chooses and is independent of CT geometry
# Thus these tables can be saved to disk for repeated use
# Below we provide an example for water and titanium, but this can be changed
# If you do change this make sure the simulated polychromatic attenuation makes sense
# The largest attenuation that a 16-bit detector can measure is -log(1/2^16) = 11.0904
# By default the x-ray physics lookup tables generate go up to np.ceil(-np.log(2.0**(-physics.detectorBits)))
# One can change the "detectorBits" parameter, but only do this if you know what you are doing
#######################################################################################################################
loZ_material = {'chemical_formula': 'H2O', 'mass_density': 1.0e-3}
#hiZ_material = {'chemical_formula': 'Al', 'mass_density': 2.7e-3}
hiZ_material = {'chemical_formula': 'Ti', 'mass_density': 4.5e-3}
sigma_loZ = physics.sigma(loZ_material['chemical_formula'], Es)
sigma_hiZ = physics.sigma(hiZ_material['chemical_formula'], Es)

# Now let's choose the reference energy BHC.  The user is free to choose what they want,
# but we recommend that this energy be within the energy range of the spectra.
# Here we will just use the mean energy of the spectra as the reference energy.
# This is what would be chosen if this parameter were not specified (i.e., the default value)
# Although not necessary, we round this value to the nearest whole number
referenceEnergy = np.round(physics.meanEnergy(s, Es))

# Now we generate the tables which should only take less than a second
startTime = time.time()
LUT_single,T_single_atten = physics.setBHClookupTable(s, Es, loZ_material['chemical_formula'], referenceEnergy)
LUT_dual,T_dual_atten = physics.setTwoMaterialBHClookupTable(s, Es, sigma_loZ, sigma_hiZ, referenceEnergy)
T_frac = 1.0/(LUT_dual.shape[0]-1)
print('BHC LUT generation time: ' + str(time.time()-startTime) + ' seconds')


#######################################################################################################################
# Simulate dual energy data
#######################################################################################################################
g = leapct.allocate_projections() # polychromatic attenuation data

g_loZ = leapct.allocate_projections() # forward projection of low Z density map
g_hiZ = leapct.allocate_projections() # forward projection of high Z density map
f_loZ = leapct.allocate_volume() # low Z density map
f_hiZ = leapct.allocate_volume() # high Z density map

# Set the high Z map as three cylinders
hiZ_diameter = 30.0/4.0
hiZ_location = 50.0/2.0
leapct.addObject(f_hiZ, 4, hiZ_location*np.array([np.cos(0.0*np.pi/180.0), np.sin(0.0*np.pi/180.0), 0.0]), hiZ_diameter*np.array([1.0, 1.0, 1.0]), 1.0, None, None, 3)
leapct.addObject(f_hiZ, 4, hiZ_location*np.array([np.cos(120.0*np.pi/180.0), np.sin(120.0*np.pi/180.0), 0.0]), hiZ_diameter*np.array([1.0, 1.0, 1.0]), 1.0, None, None, 3)
leapct.addObject(f_hiZ, 4, hiZ_location*np.array([np.cos(240.0*np.pi/180.0), np.sin(240.0*np.pi/180.0), 0.0]), hiZ_diameter*np.array([1.0, 1.0, 1.0]), 1.0, None, None, 3)

# Set the low Z map as one big cylinder.  Need to subtract out region that high Z material occupies
leapct.addObject(f_loZ, 4, np.array([0.0, 0.0, 0.0]), 120.0*np.array([1.0, 1.0, 1.0]), 1.0, None, None, 3)
f_loZ = f_loZ - f_hiZ

# Now we scale the low and high Z maps by their respective densities
f_loZ *= loZ_material['mass_density']
f_hiZ *= hiZ_material['mass_density']

# Now forward project the density maps
leapct.project(g_loZ, f_loZ)
leapct.project(g_hiZ, f_hiZ)

# Now we calculate the polychromatic attenuation
physics.normalizeSpectrum(s, Es)
for n in range(Es.size):
    if s[n] > 0.0:
        g[:] += s[n]*np.exp(-sigma_loZ[n]*g_loZ[:] - sigma_hiZ[n]*g_hiZ[:])
g = -np.log(g)

# Check that you didn't simulate data with an unrealistic attenuation
print(np.max(g))


#######################################################################################################################
# Run the multi-material BHC algorithm
# The reconstructions will be in LAC units at the reference energy specified above
#######################################################################################################################

# We need a method to classify each voxel as low Z, high Z, or a mixture of the two
# One can choose any method they want, including sophisticated segmentation algorithms
# Here we shall just use thresholding.  For this we first need to know the LAC values
# at the reference energy.
mu_loZ_ref = physics.mu(loZ_material['chemical_formula'], referenceEnergy, loZ_material['mass_density'])
mu_hiZ_ref = physics.mu(hiZ_material['chemical_formula'], referenceEnergy, hiZ_material['mass_density'])

# Print out the target values of the reconstruction (mm^-1) to verify that everything reconstructed properly
print('mu_loZ_ref = ' + str(mu_loZ_ref))
print('mu_hiZ_ref = ' + str(mu_hiZ_ref))

# We will run five iterations of this algorithm which should be nearly converged
# It only takes 2-3 iterations to remove most of the beam hardening artifacts.
# The remainder of the iterations are needed for a quantitatively-accurate reconstruction
numIter = 5

# To avoid confusion, we shalle remove g to "g_poly" so remind ourselves that the measured data
# (or in this case the simulated data) is polychromatic.  The output of the mult-material BHC
# algorithm will be g_mono, the synthesized monochromatic attenuation data
g_poly = g

# For the segmentation, we need to define the threshold such that
# [mu_loZ_ref, threshold] is classified as a mixture of the low and high Z materials
# and everything above the threshold is assumed to be purely the high Z material (but at a lower density)
# Note that this algorithm can handle variable density of the two materials, but in this case the segmentation
# algorithm must still successfully classify the voxels
threshold = (mu_hiZ_ref - mu_loZ_ref)*0.1 + mu_loZ_ref


g_mono = g_poly.copy()
for n in range(numIter):
    if n == 0:
        # For the first iteration, just pretend the entire object was composed of the low Z material
        # and perform a single-material BHC.  This should happen in near real-time
        leapct.applyTransferFunction(g_mono, LUT_single, T_single_atten)
    else:
        # There are 4 steps to perform the correction
        # With each outer iteration (the loop over n), we get better and better
        # estimates of the fraction of the attenuation is composed of the high
        # material which leads to convergence

        # Step 1: perform a reconstruction
        # You don't have to use FBP; you can use whatever algorithm you like
        f = leapct.FBP(g_mono)
        
        # Step 2: make a map of the high Z material
        # Here, we shall just use simple thresholding
        f_hiZ = f.copy()
        f_hiZ[f_hiZ<=mu_loZ_ref] = 0.0
        ind = np.logical_and(f>mu_loZ_ref, f<threshold)
        f_hiZ[ind] = f[ind]*(f[ind] - mu_loZ_ref) / (threshold - mu_loZ_ref)

        # Step 3: calculate the fraction of the attenuation that is due to the high Z material
        g_hiZ = g.copy()
        leapct.project(g_hiZ, f_hiZ)
        g_rel = g_hiZ
        ind = g_mono > 0.0
        g_rel[ind] = g_hiZ[ind] / g_mono[ind]
        g_rel[~ind] = 0.0
        
        # Step 4: apply the multi-material beam hardening correction via bi-linear interpolation of the lookup table
        g_mono[:] = g_poly[:]
        leapct.applyDualTransferFunction(T_dual_atten*(LUT_dual.shape[0]-1)*g_rel, g_mono, LUT_dual, T_dual_atten)
        
        # Alternatively, one could do nearest neighbor interpolation using python
        #g_mono = LUT_dual[np.floor(0.5+g_rel/T_frac).astype(int), np.floor(0.5+g_poly/T_dual_atten).astype(int)]
        
# Now reconstruct the corrected data
f = leapct.FBP(g_mono)
leapct.display(f)
