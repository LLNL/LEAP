
import time
import numpy as np
from VAM import VAM

# Set the target geometry which needs to be a numpy array of size
# numZ X numY x numX, where numZ = number of z-slices, etc.
# Here we will set up a hollow cylinder
N = 512
N_z = N//N
targetGeometry = np.zeros((N_z,N,N), dtype=np.float32)
x = np.array(range(N))-0.5*N
x,y = np.meshgrid(x,x)
targetGeometry[:,(x-20)**2+y**2<=(0.5*0.8*N)**2] = 1.0
targetGeometry[:,(x-20)**2+y**2<=(0.5*0.7*N)**2] = 0.0


# alpha is the attenuation coefficient of the resin, mm^-1
# D_c is the criticalDose, J / mm^3 (typical value 50-100 miliJoules / cm^3)
# Omega is the rotation rate, 1/sec
# resinDiameter is the diameter of the cyliner of resin, if not given as a positive value
#               then it is assumed that the diameter is pixelSize * targetGeometry.shape[1]
pixelSize = 0.1 # mm
alpha = 1.0/(pixelSize*0.5*float(targetGeometry.shape[1])) # 1/radius
v = VAM(alpha=alpha, D_c=1.0, Omega=0.0, resinDiameter=0.0, useDegPerSec=False)

#mu = targetGeometry.copy()
#mu[:] = alpha

g_init = None # initial guess

startTime = time.time()
g, Pstar_g = v.execute(targetGeometry, numIter=100, pixelSize=pixelSize, numAngles=360, g=g_init, mu=None)
print('total elapsed time: ' + str(time.time()-startTime))

''' Display results (requires napari)
from leapctype import *
leapct = tomographicModels()
leapct.display(Pstar_g)
leapct.display(g)
#'''
