import sys
import time
import matplotlib.pyplot as plt
from leapctype import *
leapct = tomographicModels()

numAngles = 720
numCols = 512
numRows = int(numCols*25.0/24.0)
pixelSize = 1.03*2.0*250.0/512 # 0.5*512*x=240
centerRow = 0.5*(numRows-1)
centerCol = 0.5*(numCols-1)
#sdd = 765
sdd = 1500
sod = 0.5*sdd

#centerCol += centerCol*0.1

#print(0.5*3808*0.125/700.0)
#print(np.arctan(0.5*3808*0.125/700.0)*180.0/np.pi)
#print(0.5*numRows*pixelSize/sdd)
#print(np.arctan(0.5*numRows*pixelSize/sdd)*180.0/np.pi)
#quit()

includeEar = False

voxelScales = [0.5, 1.0, 2.0]
angularRanges = [200.0, 360.0]
projection_methods = ['VD', 'SF']
angularRange = 360.0
tau = 0.0
pitch = 0.0

#voxelScales = [1.0]
#voxelScales = [2.0]
#projection_methods = ['VD']


for igeom in range(6):
    leapct.reset()
    #leapct.set_rampFilter(12)
    geo = None
    if igeom == 0:
        leapct.set_parallelbeam(numAngles, numRows, numCols, sod/sdd*pixelSize, sod/sdd*pixelSize, centerRow, centerCol, leapct.setAngleArray(numAngles, angularRange))
    elif igeom == 1:
        leapct.set_fanbeam(numAngles, numRows, numCols, sod/sdd*pixelSize, pixelSize, centerRow, centerCol, leapct.setAngleArray(numAngles, angularRange), sod, sdd, tau)
    elif igeom == 2:
        leapct.set_coneparallel(numAngles, numRows, numCols, pixelSize, sod/sdd*pixelSize, centerRow, centerCol, leapct.setAngleArray(numAngles, angularRange), sod, sdd, tau, pitch)
    elif igeom == 3:
        leapct.set_conebeam(numAngles, numRows, numCols, pixelSize, pixelSize, centerRow, centerCol, leapct.setAngleArray(numAngles, angularRange), sod, sdd, tau, pitch)
    elif igeom == 4:
        leapct.set_conebeam(numAngles, numRows, numCols, pixelSize, pixelSize, centerRow, centerCol, leapct.setAngleArray(numAngles, angularRange), sod, sdd, tau, pitch)
        leapct.set_curvedDetector()
    elif igeom == 5:
        leapct.set_conebeam(numAngles, numRows, numCols, pixelSize, pixelSize, centerRow, centerCol, leapct.setAngleArray(numAngles, angularRange), sod, sdd, tau)
        leapct.convert_to_modularbeam()

    #leapct.set_offsetScan(True)
    for iscale in range(len(voxelScales)):
        # Set true phantom
        leapct.set_default_volume(voxelScales[iscale])
        f_true = leapct.allocate_volume()
        leapct.set_FORBILD(f_true, includeEar)
        
        leapct.print_parameters()
        #leapct.display(f_true)
        #leapct.sketch_system()
        #quit()
        
        # Set true projection data
        g_true = leapct.allocate_projections()
        leapct.set_FORBILD(None, includeEar)
        st = time.time()
        leapct.rayTrace(g_true)
        print('ray trace time: ' + str(time.time()-st) + ' sec')
        
        #leapct.display(g_true)
        #leapct.display(f_true)

        for imethod in range(len(projection_methods)):

            leapct.set_projector(projection_methods[imethod])

            # Test projection
            g = leapct.allocate_projections()
            st = time.time()
            leapct.project(g,f_true)
            print('project time: ' + str(time.time()-st) + ' sec')
            
            # Test FBP
            f = leapct.allocate_volume()
            st = time.time()
            leapct.FBP(g_true, f)
            print('FBP time: ' + str(time.time()-st) + ' sec')
            
            #diff_g = g
            #diff_f = f
            diff_g = np.clip(100.0*(g-g_true)/g_true, -10.0, 10.0)
            diff_f = np.clip(100.0*(f-f_true)/f_true, -10.0, 10.0)

            
            #"""
            plt.figure()
            plt.subplot(2, 2, 1)
            plt.title('projection')
            plt.imshow(np.squeeze(diff_g[0,:,:]), cmap='gray')
            plt.subplot(2, 2, 2)
            plt.title('sinogram')
            plt.imshow(np.squeeze(diff_g[:,numRows//2,:]), cmap='gray')
            plt.subplot(2, 2, 3)
            plt.title('z-slice')
            plt.imshow(np.squeeze(diff_f[diff_f.shape[0]//2,:,:]), cmap='gray')
            plt.subplot(2, 2, 4)
            plt.title('x-slice')
            plt.imshow(np.squeeze(diff_f[:,diff_f.shape[1]//2,:]), cmap='gray')
            plt.show()
            #"""
            #quit()
            
