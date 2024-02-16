import sys
import os
import time
import numpy as np
from leapctype import *
leapct = tomographicModels()

'''
This demo script shows you how to use some of LEAP's file I/O functions to save and load
the projection data, volume data, and a text file which tracks the CT geometry and CT volume parameters (i.e., the metadata)
This example below sets the data as a sequence of tif files which is the most common ways to deal with CT data
We also recommend using the nrrd format which is a file format for N-dimension data which is readable by many common 3D image viewing software
such as ImageJ and 3D slicer

Of course you can also use npy files, but these are only good for python and are not always supported by 3D viewers
'''

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
