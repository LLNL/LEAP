# LEAP: Differentiable forward and backward projectors for AI/ML-driven computed tomography applications


## Python API

Projector class is provided to enable PyTorch integeration. It is a wrapper class that uses Python LEAP API.
This section describes the Python API functions. 
 
#### project_cpu
forward project on CPU

#### backproject_cpu
back project on CPU

#### project_gpu
forward project on GPU

#### backproject_gpu
back project on GPU

#### print_param
print all current projection and image parameters

#### save_param
save all current parameters to a file

#### set_gpu
set GPU device id for forward and back projection operations. begin with 0.

#### set_projector

#### set_symmetry_axis

#### set_cone_beam
set up cone beam projection parameters

#### set_parallel_beam
set up parallel beam projection parameters

#### set_modular_beam
set up modular beam projection parameters

#### set_volume
set up image volume dimension to be reconstructed
 
#### get_volume_dim
get image volume dimension

#### get_projection_dim
get projection data (sinogram) dimension

#### create_param
create a new parameter set, in addition to the default parameter set

#### clear_param_all
remove all parameter sets in memory, the main default parameter set will not be removed. 

#### project_cone_cpu
forward project on CPU by cone beam geometry parameters

#### project_parallel_cpu
forward project on CPU by parallel beam geometry parameters

#### backproject_cone_cpu
back project on CPU by cone beam geometry parameters

#### backproject_parallel_cpu
back project on CPU by parallel beam geometry parameters

#### project_cone_gpu
forward project on GPU by cone beam geometry parameters

#### project_parallel_gpu
forward project on GPU by parallel beam geometry parameters

#### backproject_cone_gpu
back project on GPU by cone beam geometry parameters

#### backproject_parallel_gpu
back project on GPU by parallel beam geometry parameters


## Backend C++ API

The core backend of the LEAP library is written in C++ and CUDA. It is dynamically linked in the Python environment and used with the Python binding mechanism. 

