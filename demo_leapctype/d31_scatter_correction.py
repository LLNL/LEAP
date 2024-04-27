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

'''
This script provide a demonstration of LEAP's scatter simulation/ correction functionality.  Scatter correction
(or mitigation using collimators) is essential to quantitatively accurate reconstruction and mitigation of artifacts.
It is especially important when scanning highly attenutating objects where the scatter to primary ratio is high.

There are many scatter correction algorithms.  The one in LEAP is physics-based first order scatter.  This means that
the scatter signal is calculating using physics models, rather than heuristic-based kernel methods.  This type of
scatter correction is highly accurate, but computationally expensive.

The scatter signal is a very low frequency signal in transmission space (but can create high-frequency artifacts in
attenuation space and in the reconstructed image if not accounted for).  We exploit this fact to speed up the computation
by down-sample the data, performing the scatter estimate, and then up-sampling back to the measure data resolution to
apply the correction.  Note that transmission data and attenuation data are related by:
attenuation = -log(transmission)
The attenuation data is what one uses for reconstruction.

The best physics-based scatter correction algorithm we know is outlined in the following papers:
Maslowski, Alexander, Adam Wang, Mingshan Sun, Todd Wareing, Ian Davis, and Josh Star-Lack.
"Acuros CTS: A fast, linear Boltzmann transport equation solver for computed tomography scatter–Part I: Core algorithms and validation."
Medical physics 45, no. 5 (2018): 1899-1913.

Wang, Adam, Alexander Maslowski, Philippe Messmer, Mathias Lehmann, Adam Strzelecki, Elaine Yu, Pascal Paysan et al.
"Acuros CTS: A fast, linear Boltzmann transport equation solver for computed tomography scatter–Part II: System modeling, scatter correction, and optimization."
Medical physics 45, no. 5 (2018): 1914-1925.

This algorithm accounts for all orders of scatter, while our only models first order scatter.
Unfortunately it is extremely difficult to implement and the Varian implementation is proprietary.  Although our scatter estimation
algorithm is different, we use their method to perform the actual scatter correction.  The method works as follows, where
P is the simulated primary transmission and S is the simulated scatter transmission, "measured" is the measured data (in transmission space)
and "corrected" is the scatter-corrected measured data (also in transmission space).

corrected = measured - measured / (P + S) * S
          = measured * (1 - S/(P+S))
          = measured * (P / (P+S))

Performing scatter correction of a multiplicative correction in this way is brilliant.  It avoids the possibility of the corrected
transmission data of going negative and does not rely on a perfectly accurate scatter estimation.

The LEAP scatter correction algorithm work on any scanner geometry except fan-beam, but the calculation is done using modular-beam
specification.  Note that any geometry (except fan-beam) can be modeled as modular-beam and the translation can be done by calling
the function "convert_to_modularbeam()".  Note that after the scatter correction is completed, you can complete your other calculations
in the native geomery type.

The demonstration provided here used the geometry from the following paper as a means to verify the algorithm:
Kyriakou, Yiannis, Michael Meyer, and Willi A. Kalender.
"comparing coherent and incoherent scatter effects for cone-beam CT."
Physics in Medicine & Biology 53, no. 10 (2008): N175.
'''


# Set the scanner geometry
numCols = 1280
numRows = 1024
numAngles = 180
#numAngles = 1
pixelSize = 0.65*512/numCols
leapct.set_conebeam(numAngles, numRows, numCols, 300.0/1024.0, 400.0/1280.0, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0), 785.0, 785.0+415.0)
#leapct.set_curvedDetector()


# Set the volume parameters.
leapct.set_default_volume()
#leapct.set_volume(numX, numY, numZ, voxelWidth=None, voxelHeight=None, offsetX=None, offsetY=None, offsetZ=None):


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
kV = 120.0
takeOffAngle = 11.0

# First simulate the source spectrum (units are photons/(bin * mAs * sr))
# "Es" are the energy samples (in keV) and "s" is the source spectrum
Es = np.array(range(30,int(kV)+2,2),dtype=np.float32)
#Es = np.linspace(16, int(kV), num=(int(kV)-16)//4+1)
Es,s = physics.simulateSpectra(kV,takeOffAngle,gammas=Es)

# Then model the detector response as the product of the
# x-ray energy and the stopping power of the scintillator
# Here the scintillator is 0.6 mm thick CsI
detResp = physics.detectorResponse('CsI', physics.massDensity('CsI'), 0.6, Es)

# Finally model the attenuation due to the filters
# Here the filter is 1.0 mm thick aluminum
filtResp = physics.filterResponse('Al', physics.massDensity('Al'), 2.0, Es) * physics.filterResponse('Cu', physics.massDensity('Cu'), 2.0, Es)

# Take the product of all three factors to get the total system spectral response
s_total = s*filtResp*detResp

# The data is polychromatic which means the reconstruction will have some beam hardening artifacts
# and it means we need to estimate the approximate effective energy of the spectra passing through
# the object.  In this demo, we shall model the object as water, but one can choose any material.
effectiveEnergy = physics.effectiveEnergy('H2O', 1.0e-3, 125.0, s_total, Es)

# Now we define some tables the scatter correction needs
# We will down-sample the spectra so that it has around 5-10 samples
# We do this to speed up the computation because a finely-sampled
# spectra is not needed for scatter estimation

gammas_dn = Es[0:Es.shape[0]:5]
gammas_dn = np.ascontiguousarray(gammas_dn,dtype=np.float32)
s_dn = physics.resample(Es,s,gammas_dn)

# Now we define the detector response, cross sections, and differential
# scattering cross sections from 1 keV to the maximum energy of the spectra
# for the object material, which in this case is water.
Es_full = np.array(range(1,int(gammas_dn[-1])+1),dtype=np.float32)
detResp = physics.detectorResponse('CsI', physics.massDensity('CsI'), 0.6, Es_full)

sigma_water = np.zeros((3,Es_full.size), dtype=np.float32)

sigma_water[0,:] = physics.sigmaPE('H2O',Es_full)
sigma_water[1,:] = physics.sigmaCS('H2O',Es_full)
sigma_water[2,:] = physics.sigmaRS('H2O',Es_full)
thetas = np.array(range(180+1),dtype=np.float32)
dsigma = np.zeros((2,Es_full.size,181),dtype=np.float32)
for n in range(Es_full.size):
    #dsigma[0,n,:] = physics.KleinNishinaScatterDistribution(Es_full[n], thetas, True)
    dsigma[0,n,:] = physics.incoherentScatterDistribution('H2O', Es_full[n], thetas, doNormalize=True)
    dsigma[1,n,:] = physics.coherentScatterDistribution('H2O', Es_full[n], thetas, doNormalize=True)

# If one wishes to only model Compton or Rayleigh scatter, just set the corresponding
# scatter distribution to zero
#dsigma[0,:,:] = 0.0 # this line will turn off Compton (incoherent) scatter
#dsigma[1,:,:] = 0.0 # this line will turn off Rayleigh (coherent) scatter


#######################################################################################################################
# Simulate Scatter
#######################################################################################################################
# Specify simplified FORBILD head phantom
# One could easily do this in Python, but Python is soooooo slow for these types of operations,
# so we implemented this feature with multi-threaded C++
f_rho = leapct.allocate_volume()
g = leapct.allocate_projections()
#leapct.addObject(f, 4, np.array([0.0, 0.0, 0.0]), np.array([160.0*0.5, 160.0*0.5, 150.0]), 1.0e-3, None, None, 1)
leapct.addObject(f_rho, 4, np.array([0.0, 0.0, 0.0]), np.array([250.0*0.5, 250.0*0.5, 150.0]), 1.0e-3, None, None, 1)
#leapct.set_FORBILD(f,True)

# Perform polychromatic simulation through the material (water)
f_mu = f_rho.copy()/np.max(f_rho)*physics.mu('water',effectiveEnergy)
BH_LUT, T_lut = physics.setBHlookupTable(s_total, Es, 'water', effectiveEnergy)
leapct.project(g,f_mu)
leapct.applyTransferFunction(g, BH_LUT, T_lut)
#'''

# Now define the down-sampled (Low Resolution, LR) data which will be used when modeling scatter
leapct_LR = tomographicModels()
leapct_LR.copy_parameters(leapct)

# Down-Sample projection data, volume data, and their respective parameters in LEAP
# We aim for a reconstruction volume that is between 100**3 and 200**3 voxels
downSampleFactor = 10.0
g_dn = leapct_LR.down_sample_projections([1,downSampleFactor,downSampleFactor],g)
f_rho_dn = leapct_LR.down_sample_volume([downSampleFactor,downSampleFactor,downSampleFactor], f_rho)

# Convert to modular-beam geometry
leapct_LR.convert_to_modularbeam()

physics.normalizeSpectrum(s_dn, gammas_dn)
print('Starting scatter simulation...')
startTime = time.time()
scatterGain = leapct_LR.scatter_model(f_rho_dn, s_dn, gammas_dn, detResp, sigma_water, dsigma, 1)
print('Scatter Simulation Elapsed Time: ' + str(time.time()-startTime))


# Upsample the scatter signal back to full resolution
scatterGain = leapct_LR.up_sample_projections([1,downSampleFactor,downSampleFactor],scatterGain,g.shape)

# Add scatter to data
g = leapct.negLog(leapct.expNeg(g) * scatterGain)


#######################################################################################################################
# Now we perform scatter correction
#######################################################################################################################

# First perform a reconstruction which will be used to model scatter
f = leapct.FBP(g)

# Let's display the data to observe the "cupping" effect on the reconstruction
# that is caused by scatter
leapct.display(f)


# The reconstruction is in LAC units (1/mm).  This needs to be converted to mass density units.
# LAC = (mass density) * (mass cross section), so
# mass density = LAC / (mass cross section)
# The cross section will be evaluated at the effective energy which we already calculated above for water and the given spectra
f_rho = f/physics.sigma('water',effectiveEnergy)

# Now we down-sample the data.  So we copy the LEAP parameters to a new LEAP object class and down-sample
leapct_LR.copy_parameters(leapct)
g_dn = leapct_LR.down_sample_projections([1,downSampleFactor,downSampleFactor],g)
f_rho_dn = leapct_LR.down_sample_volume([downSampleFactor,downSampleFactor,downSampleFactor], f_rho)

# Convert to modular-beam geometry
leapct_LR.convert_to_modularbeam()

# Estimate the scatter correction gain factors
# Notice that this is the same as what is done above for scatter simulation, but here the last
# argument is -1 which means one wants to remove scatter, i.e., perform scatter correction
print('Starting scatter simulation...')
startTime = time.time()
scatterGain = leapct_LR.scatter_model(f_rho_dn, s_dn, gammas_dn, detResp, sigma_water, dsigma, -1)
print('Scatter Simulation Elapsed Time: ' + str(time.time()-startTime))


# Upsample the scatter correction signal back to full resolution
scatterGain = leapct_LR.up_sample_projections([1,downSampleFactor,downSampleFactor],scatterGain,g.shape)

# Remove scatter from data
g = leapct.negLog(leapct.expNeg(g) * scatterGain)

# Let's also remove the beam hardening artifacts
BHC_LUT, T_lut = physics.setBHClookupTable(s_total, Es, 'water', effectiveEnergy)
leapct.applyTransferFunction(g, BHC_LUT, T_lut)

# Perform reconstruction and display the result.
f = leapct.FBP(g)
leapct.display(f)
