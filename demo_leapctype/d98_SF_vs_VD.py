import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from leapctype import *
leapct = tomographicModels()
from leap_preprocessing_algorithms import *
#leapct.about()

"""
This script demonstrates some of the tradeoffs between Separable Footprint* (SF) and Voxel-Driven (VD) projectors models.
LEAP is capable of performing SF-based forward and backprojection and VD-based backprojection for all CT geometries.

Briefly, SF projectors are more accurate, but VD backprojection is faster.  Use d99_speedTest.py to measure the
speed difference between these two methods.  For cone-beam, VD is approximately two times faster.
In this script we estimate the noise and resolution properties of these two methods.

Most other reconstruction packages (e.g., ASTRA, TIGRE, etc.) use VD-based backprojection.

*The SF projectors in LEAP are actually slightly different from the original paper.
"""

dataPath = os.path.dirname(os.path.realpath(__file__))
saveFigures = False

# Set scanner geometry
numCols = 512
numAngles = 2*2*int(360*numCols/1024)
pixelSize = 0.65*512/numCols
numRows = 1
sod = 1100.0
sdd = 1400.0
#leapct.set_fanbeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0), sod, sdd)
leapct.set_conebeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0), sod, sdd)

# Simulate projection data with ray tracing
leapct.addObject(None, 4, 0.0, 120.0, val=0.02)
#leapct.set_FORBILD()
g = leapct.allocate_projections()
leapct.rayTrace(g,3)

# Add noise to data
I_0 = 50000.0
g_noisey = g.copy()
g_noisey[:] = -np.log(np.random.poisson(I_0*np.exp(-g))/I_0)

# Set the ramp filter, larger numbers use a shaper filter
# Smooth Filter: 0
# Shepp-Logan: 2 (default value)
# Sharp Filter: 10 (nearly as sharp as Ram-Lak, but creates result with less ringing artifacts)
# Ram-Lak: 12
leapct.set_rampFilter(10)

# Set CT volume
for n in range(0,3):
    # We test our projectors with "small", "medium", and "large" voxels.
    # The "nominal" or "median" voxel size is sod / sdd * pixelWidth which is the detector pixel width
    # projected at the center of the field of view
    if n == 0:
        leapct.set_default_volume(0.5) # make the voxels half the size (mm) of the nominal size
    elif n == 1:
        leapct.set_default_volume() # make the voxels have the nominal size
    else:
        leapct.set_default_volume(2.0) # make the voxels double the size (mm) of the nominal size

    # Specify the region where SNR is calculated
    x,y,z = leapct.voxelSamples()
    ind = x**2 + y**2 < 100.0**2

    # Reconstruct using Voxel-Driven Backprojection
    leapct.set_projector('VD')
    f_VD = leapct.FBP(g)
    f_VD_noisey = leapct.FBP(g_noisey)

    # Reconstruct using (modified) Separable-Footprint Backprojection
    leapct.set_projector('SF')
    f_SF = leapct.FBP(g)
    f_SF_noisey = leapct.FBP(g_noisey)
    
    # Convert to MHU (Modified Housfield Units)
    f_VD *= 50000.0
    f_SF *= 50000.0
    f_VD_noisey *= 50000.0
    f_SF_noisey *= 50000.0

    # Calculate SNR
    SNR_VD = np.mean(f_VD_noisey[ind])/np.std(f_VD_noisey[ind])    
    SNR_SF = np.mean(f_SF_noisey[ind])/np.std(f_SF_noisey[ind])
    print('SNR using VD: ' + str(SNR_VD))
    print('SNR using SF: ' + str(SNR_SF))
    
    # Specify Display Window
    vmin = 1000-150
    vmax = 1000+150
    
    """ Display Noiseless Images
    #plt.figure(figsize=(10,5))
    fig, axs = plt.subplots(2, 1)
    fig.suptitle('Noiselss Reconstructions')
    plt.subplot(1, 2, 1)
    plt.title('VD')
    plt.imshow(np.squeeze(f_VD), cmap='gray', vmin=vmin, vmax=vmax)
    plt.subplot(1, 2, 2)
    plt.title('SF')
    plt.imshow(np.squeeze(f_SF), cmap='gray', vmin=vmin, vmax=vmax)
    fig.set_figheight(3.5)
    if saveFigures:
        fig.savefig(os.path.join(dataPath,'noiseless_'+str(n)+'.png'))
    else:
        plt.show()
    quit()
    #"""
    
    """ Display Noisey Images
    fig, axs = plt.subplots(2, 1)
    fig.suptitle('Noisey Reconstructions')
    plt.subplot(1, 2, 1)
    plt.title('VD')
    plt.imshow(np.squeeze(f_VD_noisey), cmap='gray', vmin=vmin, vmax=vmax)
    plt.subplot(1, 2, 2)
    plt.title('SF')
    plt.imshow(np.squeeze(f_SF_noisey), cmap='gray', vmin=vmin, vmax=vmax)
    fig.set_figheight(3.5)
    if saveFigures:
        fig.savefig(os.path.join(dataPath,'noisey_'+str(n)+'.png'))
    else:
        plt.show()
    #quit()
    #"""

    #""" Display MTF plot
    # Here we use the Robust ISO MTF Procedure, which done as follows:
    #1) Reconstruct Cylinder
    #2) Take central slice
    #3) Sort all pixels by their radial distance from center
    #4) Interpolate samples onto equi-spaced samples at four times of original pixel size
    #5) Calculate derivate by central finite differences
    #6) Calculate FFT and multiply by 1/sinc to account for frequency response of finite difference 

    #useage: MTF(leapct, f, r, center=None, getEdgeResponse=False, oversamplingRate=4)
    MTF_VD = MTF(leapct, f_VD, 120.0)
    MTF_SF = MTF(leapct, f_SF, 120.0)
    x = np.array(range(MTF_VD.size))/float(MTF_VD.size)
    plt.plot(x,MTF_VD,'k-',x,MTF_SF,'r-')
    plt.legend(['VD','SF'])
    plt.title('Resolution (MTF)')
    plt.ylabel('MTF (unitless)')
    plt.xlim((0.0, 1.0))
    plt.ylim((0.0, 1.1))
    if saveFigures:
        plt.savefig(os.path.join(dataPath,'MTF_'+str(n)+'.png'))
    else:
        plt.show()
    quit()
    #"""
    
    