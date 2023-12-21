import sys
import os
import time
import numpy as np
from leapctype import *
leapct = tomographicModels()
# Make sure you add: .../LEAP/src to your python path

# Set the scanner geometry
numCols = 512
numAngles = 2*2*int(360*numCols/1024)
pixelSize = 0.65*512/numCols
numRows = 1
leapct.set_conebeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, 360.0), 1100, 1400)

# Set the volume parameters.
leapct.set_default_volume()

# Allocate space for the projections and the volume
g = leapct.allocateProjections()
f = leapct.allocateVolume()

# Specify simplified FORBILD head phantom
leapct.set_FORBILD(f,True)

# "Simulate" projection data
leapct.project(g,f)

# Reconstruct the data
leapct.FBP(g,f)

# Set the output directory and create it if it does not exist
output_dir = 'my_ct_data_folder'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save the parameters and data
# You can save/load data as a tif sequence, npy, or nrrd files
# To use nrrd, you'll need to install the pynrrd library like this: pip install pynrrd
leapct.save_parameters(os.path.join(output_dir, 'params.txt'))
leapct.save_projections(os.path.join(output_dir, 'projections.tif'), g)
leapct.save_volume(os.path.join(output_dir, 'recon.tif'), f)

# OK, now let's reset all parameters to demonstrate that the load_parameters function work
leapct.reset()

# Print the parameters to show that they are indeed all zeros
leapct.print_parameters()

# Now load the parameters and print them to show they loaded correctly
leapct.load_parameters(os.path.join(output_dir, 'params.txt'))
leapct.print_parameters()

# Now load the data
f[:] = 0.0
g[:] = 0.0
g = leapct.load_projections(os.path.join(output_dir, 'projections.tif'))
f = leapct.load_volume(os.path.join(output_dir, 'recon.tif'))

# Display the volume to verify it loaded correctly
#leapct.display(g)
#leapct.display(f)
