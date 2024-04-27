import os
import site
from sys import platform as _platform

if _platform == "linux" or _platform == "linux2":
    lib_fname = 'libleapct.so'
    
    copy_text = 'cp ' + str(lib_fname) + ' ' + str(os.path.join(site.getsitepackages()[1], lib_fname))
    os.system(copy_text)
    
elif _platform == "win32":
    lib_fname = 'libleapct.dll'
    
    
    copy_text = 'copy ' + str(lib_fname) + ' ' + str(os.path.join(site.getsitepackages()[1], lib_fname))
    os.system(copy_text)
    
elif _platform == "darwin":
    lib_fname = 'libleapct.dylib'

    copy_text = 'cp ' + str(lib_fname) + ' ' + str(os.path.join(site.getsitepackages()[1], lib_fname))
    os.system(copy_text)
    