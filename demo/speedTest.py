import sys
import os
import time
import numpy as np
sys.path.append(r'C:\Users\champley\Documents\git_leap\LEAP\src')
from leapctype import *
leapct = tomographicModels()

#objfile = r'C:\Users\champley\Documents\tools\LTT\sampleScripts\FORBILD_head_noEar.pd'
#from LTTserver import LTTserver
#LTT = LTTserver()

#N = 512
N = 800

numAngles = int(720*N/1024)
pixelSize = 0.2*2048/N
#leapct.setFanBeamParams(numAngles, N, N, pixelSize*11.0/14.0, pixelSize, 0.5*(N-1), 0.5*(N-1), leapct.setAngleArray(numAngles, 360.0), 1100, 1400)
leapct.setConeBeamParams(numAngles, N, N, pixelSize, pixelSize, 0.5*(N-1), 0.5*(N-1), leapct.setAngleArray(numAngles, 360.0), 1100, 1400)
leapct.setDefaultVolume()
leapct.printParameters()

#leapct.setGPU(0)

g = leapct.allocateProjections()
f = leapct.allocateVolume()

f_true = leapct.allocateVolume()
x = (np.array(range(N)) - 0.5*(N-1))*pixelSize*11.0/14.0
z,y,x=np.meshgrid(x,x,x, indexing='ij')
f_true[x**2 + y**2 + z**2 <= 150.0**2] = 1.0
#f_true[:] = 1.0

startTime = time.time()
leapct.project(g,f_true)
print('projection time: ' + str(time.time()-startTime))

leapct.displayVolume(g)
#quit()

startTime = time.time()
leapct.FBP(g,f)
print('FBP time: ' + str(time.time()-startTime))

leapct.displayVolume(f)
