import matplotlib.pyplot as plt
from leapctype import *
leapct = tomographicModels()

'''
This script demonstrates some applications of the so-called "filter sequence" in LEAP.  The filter sequence
is an extensible feature for which to add multiple regularization filters or "priors" to LEAP iterative reconstruction algorithms.

The types of filters used in this demo script are just to demonstrate the features in LEAP and are not necessarily recommended
for any application.  In addition, there are many parameter included in these filters and these parameters must be tuned
to your specific application.  The default values given here might actually result in very poor performance.  Don't judge
the quality of a specific reconstruction algorithm until you have tuned its parameters for best performance.

This feature has many applications and we will slowly add these to this demon script over time.
'''

# This script can be applied to any CT geometry, but here we will demonstrate this for fan-beam data
numCols = 512
angularRange = 360.0
numAngles = 2*2*int(angularRange*numCols/1024)
#numAngles = 100
pixelSize = 0.65*512/numCols
numRows = 1
#leapct.set_parallelbeam(numAngles=numAngles, numRows=numRows, numCols=numCols, pixelHeight=pixelSize, pixelWidth=pixelSize, centerRow=0.5*(numRows-1), centerCol=0.5*(numCols-1), phis=leapct.setAngleArray(numAngles, 360.0))
leapct.set_fanbeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, angularRange), 1100, 1400)
leapct.set_default_volume()
g = leapct.allocateProjections() # shape is numAngles, numRows, numCols
#f_true = leapct.allocateVolume() # shape is numZ, numY, numX

# Simulate projection data using analytic ray tracing methods
#leapct.set_FORBILD()
leapct.addObject(None, 4, np.array([0.0, 0.0, 0.0]), 120.0*np.array([1.0, 1.0, 1.0]), 0.02)
for i in range(100):
    leapct.addObject(None, 4, 100*np.array([np.random.uniform(-1,1), np.random.uniform(-1,1), 0.0]), 5.0*np.array([1.0, 1.0, 1.0]), 0.0)
leapct.rayTrace(g)
I_0 = 1000.0
t = np.random.poisson(I_0*np.exp(-g))
t[t<=1.0] = 1.0
g[:] = -np.log(t/I_0)

# Choose which method you'd like to test
#whichMethod = 1
whichMethod = 2
#whichMethod = 3

if whichMethod == 1:
    # First, we will just try RWLS, seeded with an FBP reconstruction
    f = leapct.FBP(g)
    
    # Initialize the filter sequence and give it a strength of 2e2.  Larger strengths will enforce a stronger regularization.
    # You will need to tune this regularization strength to your specific application
    filters = filterSequence(2e2)
    
    # Add a Total Variation (TV) term.  Typical TV uses an L1 norm, but LEAP allows
    # one to use any Lp norm.  Note that if p < 1, then it will be non-convex.  This is really effective in certain
    # situations, but if you are not careful you can get trapped in a non desirable local minimum
    # The delta parameter must be tuned to your specific application
    filters.append(TV(leapct, delta=0.01/100.0, p=1.0))
    
    leapct.RWLS(g,f,200,filters,preconditioner='SQS')
    
elif whichMethod == 2:
    '''
    Now, let's try histogram sparsity. This method is simular to DART
    This method is not convex, so you must start with a solution that is "close" before you apply this type of regularization
    '''

    f = leapct.FBP(g)
    
    # Initialize the filter sequence and give it a strength of 2e2.  Larger strengths will enforce a stronger regularization.
    filters = filterSequence(2e2)
    
    # First add a Total Variation (TV) term.
    filters.append(TV(leapct, delta=0.01/100.0, p=1.0))
    
    # Now let's add the histogram sparsity term.  Give it a low weight because this is a powerful regularizer.
    # This regularizer requires one to specify the target values of the reconstruction.  These don't have to be perfect, but close.
    # We know the true object should only have values of 0.0 and 0.02, so let's use these.  This regularizer can take any number of
    # target values, i.e., "mus", but the computational complexity of the algorithm increases with more values.
    filters.append(histogramSparsity(leapct, mus=[0.0, 0.02],weight=0.00001))
    
    # Run RWLS reconstruction
    leapct.RWLS(g,f,200,filters,preconditioner='SQS')
    
elif whichMethod == 3:
    '''
    Now let's try a filter sequence with non-differentiable terms.  For this we will need to use an iterative algorithm that is not
    based on gradients.  The only algorithm in LEAP of this type is ASDPOCS.  Note that this is not exactly mathematically valid, but 
    it definitely works in practice.
    '''

    # Start with zeros
    f = leapct.allocate_volume()
    
    # Initialize the filter sequence.  Don't need to give it a strength here, because ASDPOCS does not work this way
    filters = filterSequence()
    
    # Let's add a non-differentiable term, like a median filter
    # Note that for ASDPOCS, the filters are applied in sequence, so the order does matter.
    # With this in mind, it is better to do the median filter first, otherwise TV may preserve some of the spikes
    filters.append(MedianFilter(leapct, 0.0, 5))
    
    # Let's also add a Total Variation (TV) term.
    filters.append(TV(leapct, delta=0.01/100.0, p=1.0))
    
    leapct.ASDPOCS(g,f,50,10,10,filters)

# Display the result
plt.imshow(np.squeeze(f[f.shape[0]//2,:,:]), cmap=plt.get_cmap('gray'), vmin=0.0, vmax=0.025)
plt.show()
