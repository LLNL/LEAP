# LivermorE AI Projector for Computed Tomography (LEAP)
Differentiable forward and backward projectors for AI/ML-driven computed tomography applications.

This software contains C++/CUDA source code for Computed Tomography forward and back projectors (and some other related algorithms) and compiles into a dynamic library which can be bound to Python with the included Python bindings.  One binding (leaptorch.py) allows this library to be used within PyTorch and the other is just a basic Python binding (leapctype.py).

It assumes that all data (a copy of the projections and a copy of the volume) fit into the memory of a single GPU.

See LTT (https://www.sciencedirect.com/science/article/abs/pii/S0963869521001948) for a more general-purpose and full functioning CT software package that does not have any memory limitations (it does not, however, work with PyTorch).

## Installation and Usage

Installation and usage information is posted on the wiki page here: (https://github.com/LLNL/LEAP/wiki)


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


## Authors
Hyojin Kim (hkim@llnl.gov)  
Kyle Champley (champley@gmail.com)  


## License
LEAP is distributed under the terms of the MIT license. All new contributions must be made under this license. See LICENSE in this directory for the terms of the license.  
See [LICENSE](LICENSE) for more details.  
SPDX-License-Identifier: MIT  
LLNL-CODE-848657  

