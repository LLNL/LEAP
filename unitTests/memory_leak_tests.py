import sys
import os
import time
import numpy as np
from leapctype import *
leapct = tomographicModels()
leapct.about()

'''
This purpose of this script is just to test for memory leaks.
'''


numCols = 512
numAngles = numCols
pixelSize = 0.65*512/numCols
numRows = numCols
centerCol = 0.5*(numCols-1)

numCols = 300
centerCol = 0.5*(numCols-1) + 100
leapct.set_conebeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), centerCol, leapct.setAngleArray(numAngles, 360.0), 1100, 1400)
leapct.set_offsetScan(True)
leapct.set_default_volume()

#leapct.print_parameters()
#quit()

g = leapct.allocate_projections() # shape is numAngles, numRows, numCols
f = leapct.allocate_volume() # shape is numZ, numY, numX
leapct.set_FORBILD(None,False)
leapct.rayTrace(g)

#leapct.set_projector('VD')
#g = leapct.copy_to_device(g)
#f = leapct.copy_to_device(f)

#leapct.set_gpu(0)

leapct.FBP(g,f)
#leapct.SART(g,f,5,8)
#leapct.LS(g,f,1000,preconditioner='SARR')
leapct.LS(g,f,100,preconditioner='SQS')
#leapct.OSEM(g,f,10,10)
#leapct.LS(g,f,50,'SQS')
#leapct.RWLS(g,f,50,filters,None,'SQS')
#leapct.RDLS(g,f,50,filters,1.0,True,1)
#leapct.MLTR(g,f,10,10,filters)

leapct.display(f)
