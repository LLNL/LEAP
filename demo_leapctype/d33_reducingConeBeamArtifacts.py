import sys
import os
import time
import numpy as np
from leapctype import *
leapct = tomographicModels()
leapct.about()

'''
This script provides a demonstration of a method to reduce cone-beam artifacts.
This script is motivated by the following papers.  Ignore the fact that these papers include ultrasound;
they are definitely applicable when only has CT data.

Leach, William, Jordan Lum, Kyle Champley, Stephen Azevedo, Casey Gardner, Hyojin Kim, David Stobbe, Andrew Townsend, and Joseph W. Tringe.
"Fourier method for 3-dimensional data fusion of X-ray Computed Tomography and ultrasound."
NDT & E International 127 (2022): 102600.
https://www.sciencedirect.com/science/article/am/pii/S0963869521001997

Stobbe, David, James Kelly, Brian Rogers, Kyle Champley, Andrew Townsend, and Joseph Tringe.
"Ultrasound and X-ray Cross-Characterization of a Graded Impedance Impactor used for Shock-Ramp Compression Experiments."
Sensing and Imaging 24, no. 1 (2023): 39.
https://link.springer.com/article/10.1007/s11220-023-00444-3

'''


# Specify the number of detector columns which is used below
L = 2 # down-sampling factor, used to made the code run faster
numCols = 804//L
numAngles = 720//L
pixelSize = 1.15*float(L)

# Set the number of detector rows
numRows = 256//L

# Set the scanner geometry
sod = 570.0
sdd = 1040.0
leapct.set_conebeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0), sod, sdd)
leapct.set_curvedDetector()

# Set the volume parameters.
# It is best to do this after the CT geometry is set
leapct.set_default_volume()

# If you want to specify the volume yourself, use this function:
#leapct.set_volume(numX, numY, numZ, voxelWidth=None, voxelHeight=None, offsetX=None, offsetY=None, offsetZ=None):

# Trouble-Shooting Functions
leapct.print_parameters()
#leapct.sketch_system()

# Allocate space for the projections and the volume
g = leapct.allocateProjections() # shape is numAngles, numRows, numCols
f = leapct.allocateVolume() # shape is numZ, numY, numX

# Specify a phantom with the strong gradient in the cone angle direction
# which should produce strong cone-beam artifacts
# We also put in lots of small, low contract spheres to show this method's robustness
x,y,z = leapct.voxelSamples(True)
f[(x/200.0)**2+(y/200.0)**2+(z/70.0)**8<=1] = 0.04
f[(x/190.0)**2+(y/190.0)**2+(z/60.0)**8<=1] = 0.02
leapct.LowPassFilter(f,2.0)
for n in range(1800):
    x_c = np.random.uniform(low=-1.0, high=1.0)*190.0
    y_c = np.random.uniform(low=-1.0, high=1.0)*190.0
    z_c = np.random.uniform(low=-1.0, high=1.0)*60.0
    if x_c**2 + y_c**2 < 190.0**2:
        leapct.addObject(f, 0, np.array([x_c, y_c, z_c]), 5.0*np.array([1.0, 1.0, 1.0]), 0.025)
f_true = f.copy()

# Define the cone-shaped frequency filter
# Refer to the references above to the motivation for this filter
H = f.copy()
beta = np.arctan(0.5*pixelSize*numRows/sdd)
H[:] = 0.0
x = x / np.max(x)
y = y / np.max(y)
z = z / np.max(z)
ind = np.logical_and(x**2+y**2 < (z*np.tan(beta))**2, np.abs(z)>0.0)
H[ind] = 1.0
leapct.LowPassFilter(H,3.0) # soften the edges to avoid aliasing artifacts
H = np.fft.fftshift(H)


# "Simulate" projection data
startTime = time.time()
leapct.project(g,f)
print('Forward Projection Elapsed Time: ' + str(time.time()-startTime))


# Reset the volume array to zero, otherwise iterative reconstruction algorithm will start their iterations
# with the true result which is cheating
f[:] = 0.0

# First perform an FBP reconstruction
# This result should show the most cone-beam artifacts
startTime = time.time()
leapct.FBP(g,f)
FBP_slice = np.squeeze(f[:,f.shape[1]//2,:])

# Next perform an RWLS reconstruction, starting from the FBP result
# This result should show fewer cone-beam artifacts
filters = filterSequence(1.0e0)
filters.append(TV(leapct, delta=0.005/20.0))
leapct.RWLS(g,f,50,filters,None,'SQS')
RWLS_slice = np.squeeze(f[:,f.shape[1]//2,:])

# Now perform an over-regularized reconstruction
# This reconstruction should have almost no cone-beam artifacts but it may
# be over-regularized resulting in a loss of some low contrast objects which is OK
# The added regularizer encourages the reconstruction to have only a limited number
# of values in it.  Here we choose some values, but deliberately leave out
# some values that are in the true object to illustrate the robustness of this approach
f_cartoon = f.copy()
filters.append(histogramSparsity(leapct, [0.0, 0.02, 0.04]))
leapct.RWLS(g,f_cartoon,25,filters,None,'SQS')
cartoonized_slice = np.squeeze(f_cartoon[:,f.shape[1]//2,:])

# Combine over-regularized result and RWLS result in frequency space
f_filt = np.real(np.fft.ifftn(np.fft.fftn(f)*(1.0-H))) + np.real(np.fft.ifftn(np.fft.fftn(f_cartoon)*H))
filt_slice = np.squeeze(f_filt[:,f.shape[1]//2,:])
print('Reconstruction Elapsed Time: ' + str(time.time()-startTime))

# Display the results
import matplotlib.pyplot as plt
plt.subplot(2, 2, 1)
plt.imshow(FBP_slice, cmap='gray', vmin=0.0, vmax=0.04)
plt.axis('off')
plt.title('FBP')
plt.subplot(2, 2, 2)
plt.imshow(RWLS_slice, cmap='gray', vmin=0.0, vmax=0.04)
plt.axis('off')
plt.title('RWLS')
plt.subplot(2, 2, 3)
plt.imshow(cartoonized_slice, cmap='gray', vmin=0.0, vmax=0.04)
plt.axis('off')
plt.title('over regularized')
plt.subplot(2, 2, 4)
plt.imshow(filt_slice, cmap='gray', vmin=0.0, vmax=0.04)
plt.axis('off')
plt.title('corrected')
plt.subplots_adjust(wspace=0, hspace=0)
#I = np.concatenate((FBP_slice, cartoonized_slice, filt_slice))
#plt.imshow(I, cmap='gray')
plt.show()
