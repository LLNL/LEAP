#import sys
#sys.path.append(r'..\utils')
#from leap_filter_sequence import *
import matplotlib.pyplot as plt
from leapctype import *
leapct = tomographicModels()

numCols = 512
numAngles = 2*2*int(360*numCols/1024)
#numAngles = 24
pixelSize = 0.65*512/numCols
numRows = 1
#leapct.set_parallelbeam(numAngles=numAngles, numRows=numRows, numCols=numCols, pixelHeight=pixelSize, pixelWidth=pixelSize, centerRow=0.5*(numRows-1), centerCol=0.5*(numCols-1), phis=leapct.setAngleArray(numAngles, 360.0))
leapct.set_fanbeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0), 1100, 1400)
leapct.set_default_volume()
g = leapct.allocateProjections() # shape is numAngles, numRows, numCols
f_true = leapct.allocateVolume() # shape is numZ, numY, numX

#leapct.addObject(f_true, 4, np.array([0.0, 0.0, 0.0]), 120.0*np.array([1.0, 1.0, 1.0]), 0.02, None, None, 3)
#leapct.addObject(f_true, 4, np.array([0.0, 0.0, 0.0]), 110.0*np.array([1.0, 1.0, 1.0]), 0.0, None, None, 3)
#leapct.addObject(f_true, 4, np.array([0.0, 115.0, 0.0]), 2.5*np.array([1.0, 1.0, 1.0]), 0.0, None, None, 3)

leapct.set_FORBILD(f_true,True)
leapct.project(g,f_true)
I_0 = 5000.0
g[:] = -np.log(np.random.poisson(I_0*np.exp(-g))/I_0)
f=leapct.FBP(g)
f[:] = 0.0

#f = leapct.allocate_volume()
filters = filterSequence(4e1)
filters.append(TV(leapct, delta=0.02/20.0))
#filters.append(BilateralFilter(leapct, 4.0, 0.01))
#filters.append(MedianFilter(leapct, 0.0, 3))
#filters.append(histogramSparsity(leapct, mus=np.array([0.0, 0.021, 0.036]),weight=0.0001))
#if type(filters) is filterSequence:
#    print('YES')
#leapct.ASDPOCS(g,f,50,2,5,filters)
#leapct.ASDPOCS(g,f,50,1,5)
#leapct.RWLS(g, f, 50, filters)
filters.beta *= 1.0e-2 * 0.0
#leapct.RDLS(g, f, 50, filters, nonnegativityConstraint=True)
leapct.MLTR(g, f, 500, 1, filters)
filters.append(histogramSparsity(leapct, mus=np.array([0.0, 0.021, 0.036]),weight=0.002))
#leapct.RWLS(g, f, 50, filters)
#leapct.ASDPOCS(g,f,50,2,2,0.02/20.0)

#filter = TV(leapct, delta=0.02/20.0, alpha=1.0, f_0=f_true)
#for n in range(50):
#    f=filter.apply(f)
#filter = BilateralFilter(leapct, 3.0, 0.02, 1.0)
#filter.apply(f)
'''
denoiser = filterSequence()
denoiser.append(TV(leapct, delta=0.02/20.0, alpha=1.0, f_0=None))
denoiser.append(TV(leapct, delta=0.02/20.0, alpha=1.0, f_0=None))
denoiser.append(TV(leapct, delta=0.02/20.0, alpha=1.0, f_0=None))
denoiser.append(TV(leapct, delta=0.02/20.0, alpha=1.0, f_0=None))
denoiser.append(BilateralFilter(leapct, 3.0, 0.02, 1.0))
denoiser.apply(f)
#'''

'''
denoiser = filterSequence()
denoiser.append(histogramSparsity(leapct, mus=np.array([0.0, 0.021, 0.036])))
denoiser.append(BlurFilter(leapct, 2.0))
f=denoiser.apply(f)
denoiser.clear()
denoiser.append(histogramSparsity(leapct, mus=np.array([0.0, 0.021, 0.036])))
f=denoiser.apply(f)
f=denoiser.apply(f)
f=denoiser.apply(f)
#'''

'''
f_0 = leapct.copyData(f)
denoiser = filterSequence()
denoiser.append(TV(leapct, 100.0, 500.0))
denoiser.append(histogramSparsity(leapct, mus=np.array([0.0, 0.021, 0.036]), weight=1.0))
denoiser.append(supportSparsity(leapct, 2.0, 1.0, f_0))
for n in range(5):
    #f = denoiser.apply(f)
    d = denoiser.gradient(f)
    num = leapct.sum(d**2)
    denom = denoiser.quadForm(f, d)
    if denom <= 1.0e-16:
        break
    stepSize = num / denom
    f -= stepSize * d
#'''

#filter = histogramSparsity(leapct, mus=np.array([0.0, 0.021, 0.036]))
#f=filter.apply(f)
#f=filter.apply(f)
#leapct.MedianFilter(f,0.0,3)

#filter = supportSparsity(leapct, 1.2, 1.0, f_true)
#for n in range(50):
#    f = filter.apply(f)

#f = f-f_true
#filter = supportSparsity(leapct, 1.2)
#for n in range(50):
#    f = filter.apply(f)
#f+=f_true

#filter = azimuthalFilter(leapct, 10.0, 2.0)
#for n in range(10):
#    f = filter.apply(f)
#f = filter.gradient(f)
#leapct.AzimuthalBlur(f,10.0)

plt.imshow(np.squeeze(f[f.shape[0]//2,:,:]), cmap=plt.get_cmap('gray'))
plt.show()
