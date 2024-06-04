# To use this manual install script, go to:
# https://github.com/LLNL/LEAP/releases and download
#   libleapct.dll if on Windows
# or
#   libleapct.so if on Linux
# and copy it to the LEAP folder (the same folder as this script)
# Then run the following command:
# python manual_install.py
#
# This will copy the necessary files to your python site-packages folder

import os
import site
import shutil
from sys import platform as _platform

if _platform == "linux" or _platform == "linux2":
    fname_list = ['libleapct.so']
    #dst_folder = site.getsitepackages()[0]
    dst_folder = site.getusersitepackages()
    
elif _platform == "win32":
    fname_list = ['libleapct.dll']
    dst_folder = site.getsitepackages()[1]
    
elif _platform == "darwin":
    fname_list = ['libleapct.dylib']
    dst_folder = site.getsitepackages()[1]

fname_list.append('src/leaptorch.py')
fname_list.append('src/leapctype.py')
fname_list.append('src/leap_filter_sequence.py')
fname_list.append('src/leap_preprocessing_algorithms.py')
    
current_dir = os.path.dirname(os.path.realpath(__file__))
if os.path.isfile(os.path.join(current_dir, fname_list[0])) == False:
    print('Cannot find: ' + str(fname_list[0]))
    print('Please compile this file or download from: ' + str('https://github.com/LLNL/LEAP/releases'))
    print('and copy it into the same folder as this script and then try again')
else:
    print('Copying LEAP-CT files to: ' + str(dst_folder))
    for n in range(len(fname_list)):
        shutil.copy(fname_list[n], dst_folder)
