////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2022 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// main c++ module
////////////////////////////////////////////////////////////////////////////////

//*
#include <torch/torch.h>
#include <torch/extension.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include "tomographic_models.h"
#include "parameters.h"
//#include "projectors_cpu.h"

#ifdef __USE_GPU
//#include "projectors.h"
//#include "projectors_SF.h"

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#endif


// this params instance should not be used for pytorch class
//parameters g_params;
//std::vector<parameters> g_params_list;
tomographicModels default_model;
std::vector<tomographicModels> list_models;


bool project_cpu(int param_id, torch::Tensor& g_tensor, torch::Tensor& f_tensor)
{
    tomographicModels* p_model;
    if (param_id == -1)
        p_model = &default_model;
    else if (param_id >= 0 && param_id < (int)list_models.size())
        p_model = &list_models[param_id];
    else
        return false;
    
    //CHECK_CONTIGUOUS(g_tensor);
    //CHECK_CONTIGUOUS(f_tensor);
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    
    return p_model->project_cpu(g, f);
}

bool backproject_cpu(int param_id, torch::Tensor& g_tensor, torch::Tensor& f_tensor)
{
    tomographicModels* p_model;
    if (param_id == -1)
        p_model = &default_model;
    else if (param_id >= 0 && param_id < (int)list_models.size())
        p_model = &list_models[param_id];
    else
        return false;
        
    //CHECK_CONTIGUOUS(g_tensor);
    //CHECK_CONTIGUOUS(f_tensor);
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    
    return p_model->backproject_cpu(g, f);
}

bool fbp_cpu(int param_id, torch::Tensor& g_tensor, torch::Tensor& f_tensor)
{
    tomographicModels* p_model;
    if (param_id == -1)
        p_model = &default_model;
    else if (param_id >= 0 && param_id < (int)list_models.size())
        p_model = &list_models[param_id];
    else
        return false;
        
    CHECK_INPUT(g_tensor);
    CHECK_INPUT(f_tensor);
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    
    int whichGPU_save = p_model->getGPU();
    p_model->setGPU(-1)
    bool retVal = p_model->FBP(g, f, false);
    p_model->setGPU(whichGPU_save);
    
    return retVal;
}

#ifdef __USE_GPU
bool project_gpu(int param_id, torch::Tensor& g_tensor, torch::Tensor& f_tensor)
{
    tomographicModels* p_model;
    if (param_id == -1)
        p_model = &default_model;
    else if (param_id >= 0 && param_id < (int)list_models.size())
        p_model = &list_models[param_id];
    else
        return false;
        
    CHECK_INPUT(g_tensor);
    CHECK_INPUT(f_tensor);
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    
    return p_model->project_gpu(g, f);
}

bool backproject_gpu(int param_id, torch::Tensor& g_tensor, torch::Tensor& f_tensor)
{
    tomographicModels* p_model;
    if (param_id == -1)
        p_model = &default_model;
    else if (param_id >= 0 && param_id < (int)list_models.size())
        p_model = &list_models[param_id];
    else
        return false;
        
    CHECK_INPUT(g_tensor);
    CHECK_INPUT(f_tensor);
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    
    return p_model->backproject_gpu(g, f);
}

bool fbp_gpu(int param_id, torch::Tensor& g_tensor, torch::Tensor& f_tensor)
{
    tomographicModels* p_model;
    if (param_id == -1)
        p_model = &default_model;
    else if (param_id >= 0 && param_id < (int)list_models.size())
        p_model = &list_models[param_id];
    else
        return false;
        
    CHECK_INPUT(g_tensor);
    CHECK_INPUT(f_tensor);
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    
    return p_model->FBP(g, f, false);
}

#endif

bool setGPU(int param_id, int whichGPU)
{
    tomographicModels* p_model;
    if (param_id == -1)
        p_model = &default_model;
    else if (param_id >= 0 && param_id < (int)list_models.size())
        p_model = &list_models[param_id];
    else
        return false;
    
    return p_model->setGPU(whichGPU);
}

bool setProjector(int param_id, int which)
{
    tomographicModels* p_model;
    if (param_id == -1)
        p_model = &default_model;
    else if (param_id >= 0 && param_id < (int)list_models.size())
        p_model = &list_models[param_id];
    else
        return false;
    
    return p_model->setProjector(which);
}

bool setVolumeDimensionOrder(int param_id, int which)
{
    tomographicModels* p_model;
    if (param_id == -1)
        p_model = &default_model;
    else if (param_id >= 0 && param_id < (int)list_models.size())
        p_model = &list_models[param_id];
    else
        return false;

    if (which == parameters::ZYX)
        p_model->setVolumeDimensionOrder(parameters::ZYX);
    else
        p_model->setVolumeDimensionOrder(parameters::XYZ);
    return true;
}

bool set_axisOfSymmetry(int param_id, float axisOfSymmetry)
{
    tomographicModels* p_model;
    if (param_id == -1)
        p_model = &default_model;
    else if (param_id >= 0 && param_id < (int)list_models.size())
        p_model = &list_models[param_id];
    else
        return false;
    
    return p_model->set_axisOfSymmetry(axisOfSymmetry);
}

bool printParameters(int param_id)
{
    tomographicModels* p_model;
    if (param_id == -1)
        p_model = &default_model;
    else if (param_id >= 0 && param_id < (int)list_models.size())
        p_model = &list_models[param_id];
    else
        return false;
    
    return p_model->printParameters();
}

bool saveParamsToFile(int param_id, std::string param_fn)
{
    tomographicModels* p_model;
    if (param_id == -1)
        p_model = &default_model;
    else if (param_id >= 0 && param_id < (int)list_models.size())
        p_model = &list_models[param_id];
    else
        return false;
    
    parameters* params = &(p_model->params);

    std::string phis_strs;
    for (int i = 0; i < params->numAngles; i++) {
        float phis = (params->phis[i] + 0.5*PI) * 180.0 / PI;
        char phis_str[64];
        sprintf(phis_str, " %f", phis);
        phis_strs += phis_str;
        if (i != params->numAngles - 1)
            phis_strs += ",";
    }

    std::ofstream param_file;
    param_file.open(param_fn.c_str());
    param_file << "img_dimx = " << params->numX << std::endl;
    param_file << "img_dimy = " << params->numY << std::endl;
    param_file << "img_dimz = " << params->numZ << std::endl;
    param_file << "img_pwidth = " << params->voxelWidth << std::endl;
    param_file << "img_pheight = " << params->voxelHeight << std::endl;
    param_file << "img_offsetx = " << params->offsetX << std::endl;
    param_file << "img_offsety = " << params->offsetY << std::endl;
    param_file << "img_offsetz = " << params->offsetZ << std::endl;

    if (params->geometry == parameters::CONE)
        param_file << "proj_geometry = " << "cone" << std::endl;
    else if (params->geometry == parameters::PARALLEL)
        param_file << "proj_geometry = " << "parallel" << std::endl;
    else if (params->geometry == parameters::FAN)
        param_file << "proj_geometry = " << "fan" << std::endl;
    else if (params->geometry == parameters::MODULAR)
        param_file << "proj_geometry = " << "modular" << std::endl;
    param_file << "proj_arange = " << params->angularRange << std::endl;
    param_file << "proj_nangles = " << params->numAngles << std::endl;
    param_file << "proj_nrows = " << params->numRows << std::endl;
    param_file << "proj_ncols = " << params->numCols << std::endl;
    param_file << "proj_pheight = " << params->pixelHeight << std::endl;
    param_file << "proj_pwidth = " << params->pixelWidth << std::endl;
    param_file << "proj_crow = " << params->centerRow << std::endl;
    param_file << "proj_ccol = " << params->centerCol << std::endl;
    param_file << "proj_phis = " << phis_strs << std::endl;
    param_file << "proj_sod = " << params->sod << std::endl;
    param_file << "proj_sdd = " << params->sdd << std::endl;
    param_file.close();

    return true;
}

bool reset(int param_id)
{
    tomographicModels* p_model;
    if (param_id == -1)
        p_model = &default_model;
    else if (param_id >= 0 && param_id < (int)list_models.size())
        p_model = &list_models[param_id];
    else
        return false;
    
    return p_model->reset();
}

bool getVolumeDim(int param_id, torch::Tensor& dim_tensor)
{
    tomographicModels* p_model;
    if (param_id == -1)
        p_model = &default_model;
    else if (param_id >= 0 && param_id < (int)list_models.size())
        p_model = &list_models[param_id];
    else
        return false;
    
    parameters* params = &(p_model->params);
    
    int* dim = dim_tensor.data_ptr<int>();
    dim[0] = params->numX;
    dim[1] = params->numY;
    dim[2] = params->numZ;
    return true;
}

bool getProjectionDim(int param_id, torch::Tensor& dim_tensor)
{
    tomographicModels* p_model;
    if (param_id == -1)
        p_model = &default_model;
    else if (param_id >= 0 && param_id < (int)list_models.size())
        p_model = &list_models[param_id];
    else
        return false;
    
    parameters* params = &(p_model->params);

    int* dim = dim_tensor.data_ptr<int>();
    dim[0] = params->numAngles;
    dim[1] = params->numRows;
    dim[2] = params->numCols;
    return true;
}

bool getVolumeDimensionOrder(int param_id, int& which)
{
    tomographicModels* p_model;
    if (param_id == -1)
        p_model = &default_model;
    else if (param_id >= 0 && param_id < (int)list_models.size())
        p_model = &list_models[param_id];
    else
        return false;

    which = p_model->getVolumeDimensionOrder();
    return true;
}

bool setConeBeamParams(int param_id, int numAngles, int numRows, int numCols, 
                       float pixelHeight, float pixelWidth, float centerRow, float centerCol, 
                       torch::Tensor& phis_tensor, float sod, float sdd)
{
    tomographicModels* p_model;
    if (param_id == -1)
        p_model = &default_model;
    else if (param_id >= 0 && param_id < (int)list_models.size())
        p_model = &list_models[param_id];
    else
        return false;
    
    float* phis = phis_tensor.data_ptr<float>();
    return p_model->setConeBeamParams(numAngles, numRows, numCols, 
                       pixelHeight, pixelWidth, centerRow, centerCol, 
                       phis, sod, sdd);
}

bool setParallelBeamParams(int param_id, int numAngles, int numRows, int numCols, 
                           float pixelHeight, float pixelWidth, float centerRow, float centerCol, 
                           torch::Tensor& phis_tensor)
{
    tomographicModels* p_model;
    if (param_id == -1)
        p_model = &default_model;
    else if (param_id >= 0 && param_id < (int)list_models.size())
        p_model = &list_models[param_id];
    else
        return false;
    
    float* phis = phis_tensor.data_ptr<float>();
    return p_model->setParallelBeamParams(numAngles, numRows, numCols, 
                           pixelHeight, pixelWidth, centerRow, centerCol, phis);
}

bool setFanBeamParams(int param_id, int numAngles, int numRows, int numCols, 
                      float pixelHeight, float pixelWidth, float centerRow, float centerCol, 
                      torch::Tensor& phis_tensor, float sod, float sdd)
{
    tomographicModels* p_model;
    if (param_id == -1)
        p_model = &default_model;
    else if (param_id >= 0 && param_id < (int)list_models.size())
        p_model = &list_models[param_id];
    else
        return false;
    
    //float* phis = phis_tensor.data_ptr<float>();
    //return p_model->setFanBeamParams(numAngles, numRows, numCols, 
    //                  pixelHeight, pixelWidth, centerRow, centerCol, 
    //                  phis, sod, sdd);
    return false;
}

bool setModularBeamParams(int param_id, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, 
                          torch::Tensor& sourcePositions_in_tensor, torch::Tensor& moduleCenters_in_tensor, 
                          torch::Tensor& rowVectors_in_tensor, torch::Tensor& colVectors_in_tensor)
{
    tomographicModels* p_model;
    if (param_id == -1)
        p_model = &default_model;
    else if (param_id >= 0 && param_id < (int)list_models.size())
        p_model = &list_models[param_id];
    else
        return false;
    
    float* sourcePositions_in = sourcePositions_in_tensor.data_ptr<float>();
	float* moduleCenters_in = moduleCenters_in_tensor.data_ptr<float>();
	float* rowVectors_in = rowVectors_in_tensor.data_ptr<float>();
	float* colVectors_in = colVectors_in_tensor.data_ptr<float>();
    return p_model->setModularBeamParams(numAngles, numRows, numCols, pixelHeight, pixelWidth, 
                          sourcePositions_in, moduleCenters_in, 
                          rowVectors_in, colVectors_in);
}

bool setVolumeParams(int param_id, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
    tomographicModels* p_model;
    if (param_id == -1)
        p_model = &default_model;
    else if (param_id >= 0 && param_id < (int)list_models.size())
        p_model = &list_models[param_id];
    else
        return false;
    
    return p_model->setVolumeParams(numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ);
}

bool setDefaultVolumeParams(float scale)
{
    tomographicModels* p_model;
    if (param_id == -1)
        p_model = &default_model;
    else if (param_id >= 0 && param_id < (int)list_models.size())
        p_model = &list_models[param_id];
    else
        return false;
    
    return p_model->setDefaultVolumeParameters(scale);
}

int createParams()
{
    tomographicModels new_model;
    list_models.push_back(new_model);
    return (int)list_models.size()-1;
}

bool clearParamsList()
{
    list_models.clear();
    return true;
}

// for multiple independent projector instances
bool project_cpu_ConeBeam(torch::Tensor& g_tensor, torch::Tensor& f_tensor, int whichProjector, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, torch::Tensor& phis_tensor, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    float* phis = phis_tensor.data_ptr<float>();
    
    tomographicModels tempModel;
    tempModel.setConeBeamParams(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd);
    tempModel.setVolumeParams(numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ);
    tempModel.setProjector(whichProjector);
    return tempModel.project_cpu(g, f);
}

bool project_cpu_ParallelBeam(torch::Tensor& g_tensor, torch::Tensor& f_tensor, int whichProjector, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, torch::Tensor& phis_tensor, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    float* phis = phis_tensor.data_ptr<float>();
    
    tomographicModels tempModel;
    tempModel.setParallelBeamParams(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis);
    tempModel.setVolumeParams(numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ);
    tempModel.setProjector(whichProjector);
    return tempModel.project_cpu(g, f);
}

bool backproject_cpu_ConeBeam(torch::Tensor& g_tensor, torch::Tensor& f_tensor, int whichProjector, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, torch::Tensor& phis_tensor, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    float* phis = phis_tensor.data_ptr<float>();
    
    tomographicModels tempModel;
    tempModel.setConeBeamParams(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd);
    tempModel.setVolumeParams(numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ);
    tempModel.setProjector(whichProjector);
    return tempModel.backproject_cpu(g, f);
}


bool backproject_cpu_ParallelBeam(torch::Tensor& g_tensor, torch::Tensor& f_tensor, int whichProjector, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, torch::Tensor& phis_tensor, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    float* phis = phis_tensor.data_ptr<float>();
    
    tomographicModels tempModel;
    tempModel.setParallelBeamParams(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis);
    tempModel.setVolumeParams(numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ);
    tempModel.setProjector(whichProjector);
    return tempModel.backproject_cpu(g, f);
}

#ifdef __USE_GPU
bool project_gpu_ConeBeam(torch::Tensor& g_tensor, torch::Tensor& f_tensor, int whichGPU, int whichProjector, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, torch::Tensor& phis_tensor, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    float* phis = phis_tensor.data_ptr<float>();
    
    tomographicModels tempModel;
    tempModel.setGPU(whichGPU);
    tempModel.setConeBeamParams(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd);
    tempModel.setVolumeParams(numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ);
    tempModel.setProjector(whichProjector);
    return tempModel.project_gpu(g, f);
}

bool project_gpu_ParallelBeam(torch::Tensor& g_tensor, torch::Tensor& f_tensor, int whichGPU, int whichProjector, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, torch::Tensor& phis_tensor, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    float* phis = phis_tensor.data_ptr<float>();
    
    tomographicModels tempModel;
    tempModel.setGPU(whichGPU);
    tempModel.setParallelBeamParams(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis);
    tempModel.setVolumeParams(numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ);
    tempModel.setProjector(whichProjector);
    return tempModel.project_gpu(g, f);
}

bool backproject_gpu_ConeBeam(torch::Tensor& g_tensor, torch::Tensor& f_tensor, int whichGPU, int whichProjector, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, torch::Tensor& phis_tensor, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    float* phis = phis_tensor.data_ptr<float>();
    
    tomographicModels tempModel;
    tempModel.setGPU(whichGPU);
    tempModel.setConeBeamParams(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd);
    tempModel.setVolumeParams(numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ);
    tempModel.setProjector(whichProjector);
    return tempModel.backproject_gpu(g, f);
}


bool backproject_gpu_ParallelBeam(torch::Tensor& g_tensor, torch::Tensor& f_tensor, int whichGPU, int whichProjector, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, torch::Tensor& phis_tensor, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    float* phis = phis_tensor.data_ptr<float>();
    
    tomographicModels tempModel;
    tempModel.setGPU(whichGPU);
    tempModel.setParallelBeamParams(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis);
    tempModel.setVolumeParams(numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ);
    tempModel.setProjector(whichProjector);
    return tempModel.backproject_gpu(g, f);
}

#endif


//*
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("project_cpu", &project_cpu, "forward project on CPU");
    m.def("backproject_cpu", &backproject_cpu, "back project on CPU");
    m.def("fbp_cpu", &fbp_cpu, "FBP on CPU");
#ifdef __USE_GPU
    m.def("project_gpu", &project_gpu, "forward project on GPU");
    m.def("backproject_gpu", &backproject_gpu, "back project on GPU");
    m.def("fbp_gpu", &fbp_gpu, "FBP on GPU");
#endif
    m.def("print_param", &printParameters, "print current parameters");
    m.def("save_param", &saveParamsToFile, "save current parameters to file");
    m.def("set_gpu", &setGPU, "set GPU device (0, 1, ...) ");
    m.def("set_projector", &setProjector, "");
    m.def("set_dim_order", &setVolumeDimensionOrder, "");
	m.def("set_symmetry_axis", &set_axisOfSymmetry, "");
    m.def("set_cone_beam", &setConeBeamParams, "");
	m.def("set_parallel_beam", &setParallelBeamParams, "");
	m.def("set_fan_beam", &setFanBeamParams, "");
    m.def("set_modular_beam", &setModularBeamParams, "");
	m.def("set_volume", &setVolumeParams, "");
    m.def("set_default_volume", &setDefaultVolumeParams, "");
    m.def("get_volume_dim", &getVolumeDim, "");
    m.def("get_projection_dim", &getProjectionDim, "");
    m.def("get_dim_order", &getVolumeDimensionOrder, "");
    m.def("create_param", &createParams, "");
    m.def("clear_param_all", &clearParamsList, "");
    m.def("project_cone_cpu", &project_cpu_ConeBeam, "");
    m.def("project_parallel_cpu", &project_cpu_ParallelBeam, "");
    m.def("backproject_cone_cpu", &backproject_cpu_ConeBeam, "");
    m.def("backproject_parallel_cpu", &backproject_cpu_ParallelBeam, "");
#ifdef __USE_GPU
    m.def("project_cone_gpu", &project_gpu_ConeBeam, "");
    m.def("project_parallel_gpu", &project_gpu_ParallelBeam, "");
    m.def("backproject_cone_gpu", &backproject_gpu_ConeBeam, "");
    m.def("backproject_parallel_gpu", &backproject_gpu_ParallelBeam, "");
#endif
}
//*/

