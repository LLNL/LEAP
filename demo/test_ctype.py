import sys
import os
import time
import numpy as np
sys.path.append(r'C:\Users\champley\Documents\git_leap\LEAP\src')
from leapctype import Projector
leapct = Projector()

L = 16.0
arange = 360.0
N_phis = int(arange)
M = int(2048/L)
N = int(2048/L)
pixelSize = 0.2*L
phis = leapct.setAngleArray(N_phis, arange)

# Set scanner geometry and volume
#leapct.setParallelBeamParams(N_phis, M, N, pixelSize, pixelSize, 0.5*float(M-1), 0.5*float(N-1), phis)
leapct.setConeBeamParams(N_phis, M, N, pixelSize, pixelSize, 0.5*float(M-1), 0.5*float(N-1), phis, 1100, 1400)
leapct.setDefaultVolume()

# Optional settings
#leapct.setProjector(0)
leapct.setVolumeDimensionOrder(1)
#leapct.printParameters()
#leapct.setGPU(-1)


# Simulate data by forward projecting a cube
f = leapct.allocateVolume()
f[f.shape[0]//2-25:f.shape[0]//2+25,f.shape[1]//2-25:f.shape[1]//2+25,f.shape[2]//2-25:f.shape[2]//2+25] = 1.0
g = leapct.allocateProjections()
leapct.project(g,f)
f[:] = 0.0

# Add Noise
g[g<0.0] = 0.0
g[:] = np.random.poisson(g)

# Reconstruct and Smooth
startTime = time.time()
#leapct.FBP(g,f)
#leapct.SART(g,f,1)
leapct.MLEM(g,f,1)
print('reconstruction elapsed time: ' + str(time.time()-startTime))
f[f<0.0] = 0.0
print('max value = ' + str(np.max(f)))
#leapct.MedianFilter(f, 0.0)
#leapct.diffuse(f, 0.1, 10)

# Visual data (requires napari)
leapct.displayVolume(f)

#import matplotlib.pyplot as plt
#plt.imshow(np.squeeze(f[f.shape[0]//2,:,:]))
#plt.show()
