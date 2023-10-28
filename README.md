# LivermorE AI Projector for Computed Tomography (LEAP)
Differentiable forward and backward projectors for AI/ML-driven computed tomography applications  


## Installation
To install LEAP package, use pip command: 

$ pip install .    

It is strongly recommended to run "pip uninstall leapct" if you have installed the previous version.  


## Installation on Livermore Computing (Intel/Linux)

To install LEAP on Livermore Computing, proper modules should be loaded first. To enable GPU features, the installation should be performed under the compute node where nvidia-smi is available. For example,  

$ salloc --partition=pbatch --time=1:00:00  
$ module load gcc/8.3.0  
$ module load cuda/11.7.0   
$ pip install .  


## Installation on Livermore Computing (IBM PowerAI)

$ bsub -G mlct -W 1:00 -Is bash  
$ module load gcc/8.3.0  
$ module load cuda/11.7.0   
$ pip install .  


## Usage
Please see our example code in "demo" directory: test_library.py and test_recon.py   

below is an example run (assuming that FORBILD_head_64.npy exists in "sample_data" directory)  

$ python test_recon_NN.py --proj-fn sample_data/FORBILD_head_64_sino.npy --param-fn sample_data/FORBILD_head_64_param.cfg  



## CPP-CUDA Library

This is c++ with CUDA library that can be used in both C++ and python with the provided wrapper class. It can be compiled with cmake without the use of pytorch.

# Linux
```
> cd LEAP
> ./etc/build.sh
> cd build
> cmake ..
> make clean
> make -j24
```

# Windows
The requires Visual Studio 2019.  Run the command below and then open the solution file LEAP\win_build\leap.sln
```
> .\etc\win_build.bat
```

Note that cuda compiler (nvcc) does not support gcc higher than 7.5.   


## Python-binding
In addition to our provided python library using pybind11, you can make a separate ctype python library using setup_ctype.py. Rename it to setup.py, and then run:  

$ python setup.py install  

Note that this binding option provides cpu-to-gpu copy option only, i.e., numpy array data as input and output (f, g) and they will be moved to GPU memory internally  


## Source code list
* src/CMakeLists.txt: CMake for GPU ctype projector  
* src/main_projector_ctype.cpp, .h: main code for ctype binding   
* src/main_projector.cpp: main code for pybind11  
* src/parameters.h .cpp: projector parameters used in main_projector and projectors  
* src/projectors_cpu.cpp: CPU projector (forward and backproject) for multiple scanner geometry types   
* src/projectors.cu: GPU projector (forward and backproject) for multiple scanner geometry types  
* src/leapctype.py: python wrapper class for standard ctype package  
* src/leaptorch.py: python wrapper class for pytorch nn.module package  
* setup.py: setup.py for torch projector  
* setup_ctype.py: setup.py for ctype projector  


## Resource
Information about python-c++ binding: https://realpython.com/python-bindings-overview/  
https://pytorch.org/tutorials/advanced/cpp_extension.html  


## Authors
Hyojin Kim (hkim@llnl.gov)  
Kyle Champley (champley@gmail.com)  


## Other Contributors
Jiaming Liu (jiaming.liu@wustl.edu) for reconstruction sample code  


## License
LEAP is distributed under the terms of the MIT license. All new contributions must be made under this license. See LICENSE in this directory for the terms of the license.  
See [LICENSE](LICENSE) for more details.  
SPDX-License-Identifier: MIT  
LLNL-CODE-848657  

