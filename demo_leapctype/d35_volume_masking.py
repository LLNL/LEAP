import matplotlib.pyplot as plt
from leapctype import *
leapct = tomographicModels()

'''
This script demonstrates some applications of using masking on the reconstruction volume.

Applying masks reduces the number of unknowns in the reconstruction which may help in solving ill-posed reconstruction problems.

The mask we use in this script is produced using the LEAP "space_carving" algorithm.  This algorithm is a robust method to estimate the
support of reconstructed object and can also be used to identify the high density foreground regions.

Note that LEAP also applies a circular mask to the volume based on the field of view of the projections.  The size of this mask
is automatically determined by the geometry, but one can shrink or expend this circular mask with the leapct.set_diameterFOV(d) function

Note that applying masks to the projection data can also be done.  See the "W" or "mask" arguments to the various iterative reconstruction algorithms.
'''

# This script can be applied to any CT geometry, but here we will demonstrate this for fan-beam data
numCols = 512
angularRange = 360.0
numAngles = 10
pixelSize = 0.65*512/numCols
numRows = 1
leapct.set_fanbeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, angularRange), 1100, 1400)
leapct.set_default_volume()
g = leapct.allocateProjections() # shape is numAngles, numRows, numCols

# Simulate projection data using analytic ray tracing methods and add noise
for i in range(10):
    while (True):
        x_c = np.random.uniform(-1,1)
        y_c = np.random.uniform(-1,1)
        if x_c**2 + y_c**2 <= 1.0:
            break
    leapct.addObject(None, 4, 100*np.array([x_c, y_c, 0.0]), 5.0*np.array([1.0, 1.0, 1.0]), 0.02)
leapct.rayTrace(g)
I_0 = 10000.0
t = np.random.poisson(I_0*np.exp(-g))
t[t<=1.0] = 1.0
g[:] = -np.log(t/I_0)
g[g<0.0] = 0.0

# Now we use the space carving algorithm to estimate the support of the object
# To do this we first identify the rays in the projections that pass through the object
# We do this by simply thresholding the projection data where the attenuation exceeds 0.01
ind = g>0.01
proj_mask = leapct.allocate_projections()
proj_mask[ind] = 1.0
vol_mask = leapct.allocate_volume()
leapct.space_carving(proj_mask, vol_mask)

# Now we dilate this mask so that we don't clip the edges of the objects
leapct.BlurFilter(vol_mask, 3.0)
vol_mask[vol_mask>0.0] = 1.0

# Now set the parameter that tracks this mask.  This mask will be applied before every
# forward projection and after every backprojection.
# To see the effect of this masking, try commenting out the following line
# If you are doing computations with torch tensors, make sure the volume mask
# is also a torch tensor and copied to the GPU
leapct.set_volume_mask(vol_mask)

# Perform RLS reconstruction with the help of the volume mask
f = leapct.allocate_volume()
filters = filterSequence(1.0e0)
filters.append(TV(leapct, delta=0.02/20.0, p=1.0))
leapct.RLS(g,f,100,filters=filters,preconditioner='SQS')

# Display the result
plt.imshow(np.squeeze(f[f.shape[0]//2,:,:]), cmap=plt.get_cmap('gray'))
plt.show()
