import sys
import os
import numpy as np
sys.path.append(r'C:\Users\champley\Documents\git_leap\LEAP\src')
from leapctype import Projector
leapct = Projector()

N_phis = 180
arange = 180
M = 100
N = 256
pixelSize = 0.2*8.0
phis = leapct.setAngleArray(N_phis, arange)

# Set scanner geometry and volume
#leapct.setParallelBeamParams(N_phis, M, N, pixelSize, pixelSize, 0.5*float(M-1), 0.5*float(N-1), phis)
leapct.setConeBeamParams(N_phis, M, N, pixelSize, pixelSize, 0.5*float(M-1), 0.5*float(N-1), phis, 1100, 1400)
#leapct.setVolumeParams(N, N, M, pixelSize)
leapct.setDefaultVolume()

# Optional settings
#leapct.setProjector(0)
#leapct.setVolumeDimensionOrder(1)

# Simulate data by forward projecting a cube
f = leapct.allocateVolume()
f[f.shape[0]//2-25:f.shape[0]//2+25,f.shape[1]//2-25:f.shape[1]//2+25,f.shape[2]//2-25:f.shape[2]//2+25] = 1.0
g = leapct.allocateProjections()
leapct.project(g,f)

# Add Noise
g[g<0.0] = 0.0
g[:] = np.random.poisson(g)

# Reconstruct and Smooth
leapct.FBP(g,f)
#leapct.BPF(g,f)
f[f<0.0] = 0.0
#leapct.MedianFilter(f, 0.0)
leapct.diffuse(f, 0.1, 10)

# Visual data (requires napari)
leapct.displayVolume(f)
