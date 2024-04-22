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
numCols = 1280
numAngles = 2*2*int(360*numCols/1024)
numAngles = 1
pixelSize = 0.65*512/numCols

# Set the number of detector rows
numRows = 1024
#numRows = 

# Set the scanner geometry
leapct.set_conebeam(numAngles, numRows, numCols, 300.0/1024.0, 400.0/1280.0, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0), 785.0, 785.0+415.0)
#leapct.set_curvedDetector()
print(leapct.setAngleArray(numAngles, 360.0))

# Set the volume parameters.
# It is best to do this after the CT geometry is set
leapct.set_default_volume()

# If you want to specify the volume yourself, use this function:
#leapct.set_volume(numX, numY, numZ, voxelWidth=None, voxelHeight=None, offsetX=None, offsetY=None, offsetZ=None):

# Trouble-Shooting Functions
#leapct.print_parameters()
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
#detResp = physics.detectorResponse('O2SGd2', 7.32e-3, 0.1, Es)
detResp = physics.detectorResponse('CsI', 4.51e-3, 0.6, Es)

# Finally model the attenuation due to the filters
# Here the filter is 1.0 mm thick aluminum
filtResp = physics.filterResponse('Al', 2.7e-3, 1.0, Es)

# Take the product of all three factors to get the total system spectral response
s_total = s*filtResp*detResp

# Scatter correction requires:
# the detector response at energies from 1 keV, in 1 keV bins
#gammas_dn = Es[0:-1:10] + 0.5*Es[0]
#s_dn = physics.resample(Es,s,gammas_dn)

monoEnergy = 40.0
gammas_dn = np.array([monoEnergy], dtype=np.float32)
s_dn = np.array([1.0], dtype=np.float32)

Es_full = np.array(range(1,int(gammas_dn[-1])+1),dtype=np.float32)
#detResp = physics.detectorResponse('O2SGd2', 7.32e-3, 0.1, Es_full)
detResp = physics.detectorResponse('CsI', 4.51e-3, 0.6, Es_full)

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
#plt.plot(thetas,dsigma[1,-1,:])
#plt.show()
#print(Es_full.size)
#quit()

#dsigma[:] = 1.0
#dsigma[0,:,:] = 0.0
#dsigma[1,:,:] = 0.0

#######################################################################################################################
# Simulate Scatter
#######################################################################################################################
# Specify simplified FORBILD head phantom
# One could easily do this in Python, but Python is soooooo slow for these types of operations,
# so we implemented this feature with multi-threaded C++
f = leapct.allocate_volume()
g = leapct.allocate_projections()
leapct.addObject(f, 4, np.array([0.0, 0.0, 0.0]), np.array([160.0*0.5, 160.0*0.5, 150.0]), 1.0e-3, None, None, 1)
#leapct.addObject(f, 4, np.array([0.0, 0.0, 0.0]), np.array([320.0*0.5, 320.0*0.5, 150.0]), 1.0e-3, None, None, 1)
#leapct.set_FORBILD(f,True)
#leapct.display(f)
#quit()

leapct_LR = tomographicModels()
leapct_LR.copy_parameters(leapct)

downSampleFactor = 20.0

#leapct_LR.print_parameters()
g_dn = leapct_LR.down_sample_projections([1,downSampleFactor,downSampleFactor],g)
f_dn = leapct_LR.down_sample_volume([downSampleFactor,downSampleFactor,downSampleFactor], f)

#leapct_LR.print_parameters()
leapct_LR.convert_to_modularbeam()
leapct_LR.print_parameters()
#quit()

#leapct.display(f_dn)

g_save = leapct_LR.allocate_projections()
leapct_LR.project(g_save,f_dn)
g_save = physics.sigma('H2O',monoEnergy)*g_save
scalar = np.exp(-g_save[0,g_save.shape[1]//2,g_save.shape[2]//2])
#leapct_LR.display(g_save)
#quit()

physics.normalizeSpectrum(s_dn, gammas_dn)
#detResp = detResp / np.sum(detResp)
#detResp[:] = 1.0
startTime = time.time()
g_dn = leapct_LR.simulate_scatter(f_dn, s_dn, gammas_dn, detResp, sigma_water, dsigma)
print('Scatter Simulation Elapsed Time: ' + str(time.time()-startTime))

g_dn = g_dn / scalar

g = leapct_LR.up_sample_projections([1,downSampleFactor,downSampleFactor],g_dn)
print(g.shape)

#leapct.display(g_dn)
if np.max(g_dn) > 0.0 or np.min(g_dn) < 0.0:
    #print(np.max(g_dn))
    #plt.imshow(np.squeeze(g_dn[0,:,:]))
    plt.imshow(np.squeeze(g[0,:,:]))
    #plt.plot(np.squeeze(g[0,g.shape[1]//2,:]))
    #I = 100.0*(g_dn-np.exp(-g_save))/np.exp(-g_save)
    #plt.imshow(np.squeeze(I))
    plt.show()
else:
    print('NOTHING!')
quit()
    
print(s_dn)
print(gammas_dn)
print(detResp)
print(sigma_water)
