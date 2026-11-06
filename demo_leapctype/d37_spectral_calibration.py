import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from leapctype import *
leapct = tomographicModels()

'''
This script demonstrates how to perform a spectral calibration by scanning well-known high-purity reference materials.
The accuracy of the initial guess is very important because the calibration itself cannot fully determine the spectra.
The main purpose of the calibration is to account for filter materials that weren't included in the model and/or
nonlinearities of the photodiode respose (a part of the detector response).

We will make changes to this demo script in the future to make it more robust and run faster.  For now, we just
want to provide a basic working example.

One thing this script leaves out is that the model locations of the reference cylinders must be specified with
very, very high accuracy, so in practice one should perform a registration algorithm on these locations.
'''

# Model total system spectral response as 100 kV source, 2 mm Al filter, and GOS scintillator 0.1 mm thick
#Es = np.array(range(10,102,2), dtype=np.float32)
Es, s_source = leapct.simulateSpectra(100.0,11.0)
detResp = leapct.detectorResponse('GOS', None, 0.1, Es)
filtResp = leapct.filterResponse('Al', None, 2.0, Es)
s_total = s_source*detResp*filtResp
leapct.normalizeSpectrum(s_total, Es)


# Set the scanner geometry
numAngles = 720
numCols = 1200
numRows = 1
pixelSize = 0.2
leapct.set_fanbeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1)+10, leapct.setAngleArray(numAngles, 360.0), 1100, 1400)


# Specify spectral calibration phantom as three half inch diameter cylinders of delrin, teflon, and magnesium
referenceEnergy = 50.0
sigma_hat_delrin = leapct.mu('delrin',Es) / leapct.mu('delrin', referenceEnergy)
leapct.addObject(None, 4, 80.0*np.array([np.cos(0.0*np.pi/180.0), np.sin(0.0*np.pi/180.0), 0.0]), 6.35*np.array([1.0, 1.0, 1.0]), leapct.mu('delrin', referenceEnergy))
g_delrin = leapct.allocate_projections()
leapct.rayTrace(g_delrin, oversampling=3)

leapct.clearPhantom()
sigma_hat_teflon = leapct.mu('teflon',Es) / leapct.mu('teflon', referenceEnergy)
leapct.addObject(None, 4, 80.0*np.array([np.cos(120.0*np.pi/180.0), np.sin(120.0*np.pi/180.0), 0.0]), 6.35*np.array([1.0, 1.0, 1.0]), leapct.mu('teflon', referenceEnergy))
g_teflon = leapct.allocate_projections()
leapct.rayTrace(g_teflon, oversampling=3)

leapct.clearPhantom()
sigma_hat_Mg = leapct.mu('Mg',Es) / leapct.mu('Mg', referenceEnergy)
leapct.addObject(None, 4, 80.0*np.array([np.cos(240.0*np.pi/180.0), np.sin(240.0*np.pi/180.0), 0.0]), 6.35*np.array([1.0, 1.0, 1.0]), leapct.mu('Mg', referenceEnergy))
g_Mg = leapct.allocate_projections()
leapct.rayTrace(g_Mg, oversampling=3)


# Calculate measured polychromatic transmission data
print('Simulating polychromatic data...')
t = leapct.allocate_projections()
for n in range(Es.size):
    t[:] += s_total[n]*np.exp(-g_delrin[:]*sigma_hat_delrin[n] - g_teflon[:]*sigma_hat_teflon[n] - g_Mg[:]*sigma_hat_Mg[n])


# Now calculate model for optimization
g_delrin = np.ravel(g_delrin).astype(np.float64)
g_teflon = np.ravel(g_teflon).astype(np.float64)
g_Mg = np.ravel(g_Mg).astype(np.float64)
t = np.ravel(t).astype(np.float64)


# The spectral calibration attempts to solve the following: As = t, where s is the unknown spectra model, t is the measured
# polychromatic transmission data, and A is the model matrix.  The matrix A will be rather large, so to reduce the size of model
# we shall instead solve A'As = A'b (where the apostrophy indicates the matrix transpose)
# The next few lines of code set A'A (which we just call A) and A'b (which we just call b)
print('Setting model matrices...')
b = np.zeros(Es.size)
A = np.zeros((Es.size,Es.size))
for i in range(Es.size):
    b[i] = np.sum(np.exp(-g_delrin[:]*sigma_hat_delrin[i] - g_teflon[:]*sigma_hat_teflon[i] - g_Mg[:]*sigma_hat_Mg[i])*t)
    for j in range(i,Es.size):
        accum = np.sum(np.exp(-g_delrin[:]*(sigma_hat_delrin[i]+sigma_hat_delrin[j]) - g_teflon[:]*(sigma_hat_teflon[i]+sigma_hat_teflon[j]) - g_Mg[:]*(sigma_hat_Mg[i]+sigma_hat_Mg[j])))
        A[i,j] = accum
        A[j,i] = accum


# Set the initial guess of the spectra, we know what it really is, but let's change it to prove our method works
s_est = s_source.copy()
s_est[:] = 1.0
#s_est = s_source*leapct.filterResponse('Al', None, 0.25, Es)
s_est = s_source*detResp*leapct.filterResponse('Al', None, 0.25, Es)
leapct.normalizeSpectrum(s_est, Es)
s_init = s_est.copy()
print('mean energy of initial guess: ', leapct.meanEnergy(s_est, Es))
print('mean energy of true spectra: ' , leapct.meanEnergy(s_total, Es))


# Use Richard-Lucy algorithm to optimize spectra model, we use this model because it is easy to perserve nonnegativity and provides smooth updates
print('Solving for spectra model...')
At1 = np.matmul(A, np.ones(Es.size))
for n in range(10000000):
    s_est = (s_est / At1) * np.matmul(A, b / np.matmul(A,s_est))
print('mean energy of optimized guess: ', leapct.meanEnergy(s_est, Es))
plt.plot(Es,s_init,'b-', Es,s_est,'r-o', Es,s_total,'k-')
plt.show()