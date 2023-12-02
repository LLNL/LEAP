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

#ifdef __USE_GPU

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#endif


// this params instance should not be used for pytorch class
//parameters g_params;
//std::vector<parameters> g_params_list;
tomographicModels default_model;
std::vector<tomographicModels> list_models;

tomographicModels* get_model(int param_id)
{
    if (param_id == -1)
        return &default_model;
    else if (param_id >= 0 && param_id < (int)list_models.size())
        return &list_models[param_id];
    else
        return NULL;
}

bool project_cpu(int param_id, torch::Tensor& g_tensor, torch::Tensor& f_tensor)
{
    tomographicModels* p_model = get_model(param_id);
    if (p_model == NULL)
        return false;
    
    //CHECK_CONTIGUOUS(g_tensor);
    //CHECK_CONTIGUOUS(f_tensor);
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    
    return p_model->project_cpu(g, f);
}

bool backproject_cpu(int param_id, torch::Tensor& g_tensor, torch::Tensor& f_tensor)
{
    tomographicModels* p_model = get_model(param_id);
    if (p_model == NULL)
        return false;
        
    //CHECK_CONTIGUOUS(g_tensor);
    //CHECK_CONTIGUOUS(f_tensor);
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    
    return p_model->backproject_cpu(g, f);
}

bool filterProjections_cpu(int param_id, torch::Tensor& g_tensor)
{
    tomographicModels* p_model = get_model(param_id);
    if (p_model == NULL)
        return false;
        
    CHECK_INPUT(g_tensor);
    float* g = g_tensor.data_ptr<float>();
    
    int whichGPU_save = p_model->get_GPU();
    p_model->set_GPU(-1);
    bool retVal = p_model->filterProjections(g, false);
    p_model->set_GPU(whichGPU_save);
    
    return retVal;
}

bool FBP_cpu(int param_id, torch::Tensor& g_tensor, torch::Tensor& f_tensor)
{
    tomographicModels* p_model = get_model(param_id);
    if (p_model == NULL)
        return false;
        
    CHECK_INPUT(g_tensor);
    CHECK_INPUT(f_tensor);
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    
    int whichGPU_save = p_model->get_GPU();
    p_model->set_GPU(-1);
    bool retVal = p_model->doFBP(g, f, false);
    p_model->set_GPU(whichGPU_save);
    
    return retVal;
}

#ifdef __USE_GPU
bool project_gpu(int param_id, torch::Tensor& g_tensor, torch::Tensor& f_tensor)
{
    tomographicModels* p_model = get_model(param_id);
    if (p_model == NULL)
        return false;
        
    CHECK_INPUT(g_tensor);
    CHECK_INPUT(f_tensor);
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    
    return p_model->project_gpu(g, f);
}

bool backproject_gpu(int param_id, torch::Tensor& g_tensor, torch::Tensor& f_tensor)
{
    tomographicModels* p_model = get_model(param_id);
    if (p_model == NULL)
        return false;
        
    CHECK_INPUT(g_tensor);
    CHECK_INPUT(f_tensor);
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    
    return p_model->backproject_gpu(g, f);
}

bool filterProjections_gpu(int param_id, torch::Tensor& g_tensor)
{
    tomographicModels* p_model = get_model(param_id);
    if (p_model == NULL)
        return false;
        
    CHECK_INPUT(g_tensor);
    float* g = g_tensor.data_ptr<float>();
    
    return p_model->filterProjections(g, false);
}

bool FBP_gpu(int param_id, torch::Tensor& g_tensor, torch::Tensor& f_tensor)
{
    tomographicModels* p_model = get_model(param_id);
    if (p_model == NULL)
        return false;
        
    CHECK_INPUT(g_tensor);
    CHECK_INPUT(f_tensor);
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    
    return p_model->doFBP(g, f, false);
}

#endif

bool set_gpu(int param_id, int whichGPU)
{
    tomographicModels* p_model = get_model(param_id);
    if (p_model == NULL)
        return false;
    
    return p_model->set_GPU(whichGPU);
}

bool set_gpus(int param_id, float* whichGPUs)
{
    tomographicModels* p_model = get_model(param_id);
    if (p_model == NULL)
        return false;
    
    return p_model->set_GPUs(whichGPUs);
}

bool set_projector(int param_id, int which)
{
    tomographicModels* p_model = get_model(param_id);
    if (p_model == NULL)
        return false;
    
    return p_model->set_projector(which);
}

bool set_volumeDimensionOrder(int param_id, int which)
{
    tomographicModels* p_model = get_model(param_id);
    if (p_model == NULL)
        return false;

    if (which == parameters::ZYX)
        p_model->set_volumeDimensionOrder(parameters::ZYX);
    else
        p_model->set_volumeDimensionOrder(parameters::XYZ);
    return true;
}

bool set_axisOfSymmetry(int param_id, float axisOfSymmetry)
{
    tomographicModels* p_model = get_model(param_id);
    if (p_model == NULL)
        return false;
    
    return p_model->set_axisOfSymmetry(axisOfSymmetry);
}

bool print_parameters(int param_id)
{
    tomographicModels* p_model = get_model(param_id);
    if (p_model == NULL)
        return false;
    
    return p_model->print_parameters();
}

bool saveParamsToFile(int param_id, std::string param_fn)
{
    tomographicModels* p_model = get_model(param_id);
    if (p_model == NULL)
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
    //*
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
    //*/
    /*
    param_file << "numX = " << params->numX << std::endl;
    param_file << "numY = " << params->numY << std::endl;
    param_file << "numZ = " << params->numZ << std::endl;
    param_file << "voxelWidth = " << params->voxelWidth << std::endl;
    param_file << "voxelHeight = " << params->voxelHeight << std::endl;
    param_file << "offsetX = " << params->offsetX << std::endl;
    param_file << "offsetY = " << params->offsetY << std::endl;
    param_file << "offsetZ = " << params->offsetZ << std::endl;

    if (params->geometry == parameters::CONE)
        param_file << "geometry = " << "CONE" << std::endl;
    else if (params->geometry == parameters::PARALLEL)
        param_file << "geometry = " << "PARALLEL" << std::endl;
    else if (params->geometry == parameters::FAN)
        param_file << "geometry = " << "FAN" << std::endl;
    else if (params->geometry == parameters::MODULAR)
        param_file << "geometry = " << "MODULAR" << std::endl;
    param_file << "angularRange = " << params->angularRange << std::endl;
    param_file << "numAngles = " << params->numAngles << std::endl;
    param_file << "numRows = " << params->numRows << std::endl;
    param_file << "numCols = " << params->numCols << std::endl;
    param_file << "pixelHeight = " << params->pixelHeight << std::endl;
    param_file << "pixelWidth = " << params->pixelWidth << std::endl;
    param_file << "centerRow = " << params->centerRow << std::endl;
    param_file << "centerCol = " << params->centerCol << std::endl;
    param_file << "phis = " << phis_strs << std::endl;
    param_file << "sod = " << params->sod << std::endl;
    param_file << "sdd = " << params->sdd << std::endl;
    //*/
    param_file.close();

    return true;
}

bool reset(int param_id)
{
    tomographicModels* p_model = get_model(param_id);
    if (p_model == NULL)
        return false;
    
    return p_model->reset();
}

bool get_volumeDim(int param_id, torch::Tensor& dim_tensor)
{
    tomographicModels* p_model = get_model(param_id);
    if (p_model == NULL)
        return false;
    
    parameters* params = &(p_model->params);
    
    int* dim = dim_tensor.data_ptr<int>();
    dim[0] = params->numX;
    dim[1] = params->numY;
    dim[2] = params->numZ;
    return true;
}

bool get_projectionDim(int param_id, torch::Tensor& dim_tensor)
{
    tomographicModels* p_model = get_model(param_id);
    if (p_model == NULL)
        return false;
    
    parameters* params = &(p_model->params);

    int* dim = dim_tensor.data_ptr<int>();
    dim[0] = params->numAngles;
    dim[1] = params->numRows;
    dim[2] = params->numCols;
    return true;
}

bool get_volumeDimensionOrder(int param_id, int& which)
{
    tomographicModels* p_model = get_model(param_id);
    if (p_model == NULL)
        return false;

    which = p_model->get_volumeDimensionOrder();
    return true;
}

bool set_conebeam(int param_id, int numAngles, int numRows, int numCols, 
                       float pixelHeight, float pixelWidth, float centerRow, float centerCol, 
                       torch::Tensor& phis_tensor, float sod, float sdd, float tau, float helicalPitch)
{
    tomographicModels* p_model = get_model(param_id);
    if (p_model == NULL)
        return false;
    
    float* phis = phis_tensor.data_ptr<float>();
    return p_model->set_conebeam(numAngles, numRows, numCols, 
                       pixelHeight, pixelWidth, centerRow, centerCol, 
                       phis, sod, sdd, tau, helicalPitch);
}

bool set_fanbeam(int param_id, int numAngles, int numRows, int numCols, 
                      float pixelHeight, float pixelWidth, float centerRow, float centerCol, 
                      torch::Tensor& phis_tensor, float sod, float sdd, float tau)
{
    tomographicModels* p_model = get_model(param_id);
    if (p_model == NULL)
        return false;
    
    float* phis = phis_tensor.data_ptr<float>();
    return p_model->set_fanbeam(numAngles, numRows, numCols, 
                      pixelHeight, pixelWidth, centerRow, centerCol, 
                      phis, sod, sdd, tau);
    return false;
}

bool set_parallelbeam(int param_id, int numAngles, int numRows, int numCols, 
                           float pixelHeight, float pixelWidth, float centerRow, float centerCol, 
                           torch::Tensor& phis_tensor)
{
    tomographicModels* p_model = get_model(param_id);
    if (p_model == NULL)
        return false;
    
    float* phis = phis_tensor.data_ptr<float>();
    return p_model->set_parallelbeam(numAngles, numRows, numCols, 
                           pixelHeight, pixelWidth, centerRow, centerCol, phis);
}

bool set_modularbeam(int param_id, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, 
                          torch::Tensor& sourcePositions_in_tensor, torch::Tensor& moduleCenters_in_tensor, 
                          torch::Tensor& rowVectors_in_tensor, torch::Tensor& colVectors_in_tensor)
{
    tomographicModels* p_model = get_model(param_id);
    if (p_model == NULL)
        return false;
    
    float* sourcePositions_in = sourcePositions_in_tensor.data_ptr<float>();
	float* moduleCenters_in = moduleCenters_in_tensor.data_ptr<float>();
	float* rowVectors_in = rowVectors_in_tensor.data_ptr<float>();
	float* colVectors_in = colVectors_in_tensor.data_ptr<float>();
    return p_model->set_modularbeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, 
                          sourcePositions_in, moduleCenters_in, 
                          rowVectors_in, colVectors_in);
}

bool set_volume(int param_id, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
    tomographicModels* p_model = get_model(param_id);
    if (p_model == NULL)
        return false;
    
    return p_model->set_volume(numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ);
}

bool set_default_volume(int param_id, float scale)
{
    tomographicModels* p_model = get_model(param_id);
    if (p_model == NULL)
        return false;
    
    return p_model->set_default_volume(scale);
}

int create_param()
{
    tomographicModels new_model;
    list_models.push_back(new_model);
    return (int)list_models.size()-1;
}

bool clearParamList()
{
    list_models.clear();
    return true;
}

// for multiple independent projector instances
bool project_coneBeam_cpu(torch::Tensor& g_tensor, torch::Tensor& f_tensor, int whichProjector, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, torch::Tensor& phis_tensor, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    float* phis = phis_tensor.data_ptr<float>();
    
    tomographicModels tempModel;
    tempModel.set_coneBeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd);
    tempModel.set_volume(numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ);
    tempModel.set_projector(whichProjector);
    return tempModel.project_cpu(g, f);
}

bool project_fanBeam_cpu(torch::Tensor& g_tensor, torch::Tensor& f_tensor, int whichProjector, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, torch::Tensor& phis_tensor, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    float* phis = phis_tensor.data_ptr<float>();
    
    tomographicModels tempModel;
    tempModel.set_fanBeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd);
    tempModel.set_volume(numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ);
    tempModel.set_projector(whichProjector);
    return tempModel.project_cpu(g, f);
}

bool project_parallelBeam_cpu(torch::Tensor& g_tensor, torch::Tensor& f_tensor, int whichProjector, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, torch::Tensor& phis_tensor, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    float* phis = phis_tensor.data_ptr<float>();
    
    tomographicModels tempModel;
    tempModel.set_parallelBeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis);
    tempModel.set_volume(numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ);
    tempModel.set_projector(whichProjector);
    return tempModel.project_cpu(g, f);
}

bool backproject_coneBeam_cpu(torch::Tensor& g_tensor, torch::Tensor& f_tensor, int whichProjector, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, torch::Tensor& phis_tensor, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    float* phis = phis_tensor.data_ptr<float>();
    
    tomographicModels tempModel;
    tempModel.set_coneBeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd);
    tempModel.set_volume(numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ);
    tempModel.set_projector(whichProjector);
    return tempModel.backproject_cpu(g, f);
}

bool backproject_fanBeam_cpu(torch::Tensor& g_tensor, torch::Tensor& f_tensor, int whichProjector, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, torch::Tensor& phis_tensor, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    float* phis = phis_tensor.data_ptr<float>();
    
    tomographicModels tempModel;
    tempModel.set_fanBeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd);
    tempModel.set_volume(numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ);
    tempModel.set_projector(whichProjector);
    return tempModel.backproject_cpu(g, f);
}

bool backproject_parallelBeam_cpu(torch::Tensor& g_tensor, torch::Tensor& f_tensor, int whichProjector, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, torch::Tensor& phis_tensor, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    float* phis = phis_tensor.data_ptr<float>();
    
    tomographicModels tempModel;
    tempModel.set_parallelBeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis);
    tempModel.set_volume(numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ);
    tempModel.set_projector(whichProjector);
    return tempModel.backproject_cpu(g, f);
}

#ifdef __USE_GPU
bool project_coneBeam_gpu(torch::Tensor& g_tensor, torch::Tensor& f_tensor, int whichGPU, int whichProjector, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, torch::Tensor& phis_tensor, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    float* phis = phis_tensor.data_ptr<float>();
    
    tomographicModels tempModel;
    tempModel.set_GPU(whichGPU);
    tempModel.set_coneBeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd);
    tempModel.set_volume(numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ);
    tempModel.set_projector(whichProjector);
    return tempModel.project_gpu(g, f);
}

bool project_fanBeam_gpu(torch::Tensor& g_tensor, torch::Tensor& f_tensor, int whichGPU, int whichProjector, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, torch::Tensor& phis_tensor, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    float* phis = phis_tensor.data_ptr<float>();
    
    tomographicModels tempModel;
    tempModel.set_GPU(whichGPU);
    tempModel.set_fanBeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd);
    tempModel.set_volume(numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ);
    tempModel.set_projector(whichProjector);
    return tempModel.project_gpu(g, f);
}

bool project_parallelBeam_gpu(torch::Tensor& g_tensor, torch::Tensor& f_tensor, int whichGPU, int whichProjector, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, torch::Tensor& phis_tensor, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    float* phis = phis_tensor.data_ptr<float>();
    
    tomographicModels tempModel;
    tempModel.set_GPU(whichGPU);
    tempModel.set_parallelBeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis);
    tempModel.set_volume(numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ);
    tempModel.set_projector(whichProjector);
    return tempModel.project_gpu(g, f);
}

bool backproject_coneBeam_gpu(torch::Tensor& g_tensor, torch::Tensor& f_tensor, int whichGPU, int whichProjector, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, torch::Tensor& phis_tensor, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    float* phis = phis_tensor.data_ptr<float>();
    
    tomographicModels tempModel;
    tempModel.set_GPU(whichGPU);
    tempModel.set_coneBeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd);
    tempModel.set_volume(numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ);
    tempModel.set_projector(whichProjector);
    return tempModel.backproject_gpu(g, f);
}

bool backproject_fanBeam_gpu(torch::Tensor& g_tensor, torch::Tensor& f_tensor, int whichGPU, int whichProjector, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, torch::Tensor& phis_tensor, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    float* phis = phis_tensor.data_ptr<float>();
    
    tomographicModels tempModel;
    tempModel.set_GPU(whichGPU);
    tempModel.set_fanBeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd);
    tempModel.set_volume(numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ);
    tempModel.set_projector(whichProjector);
    return tempModel.backproject_gpu(g, f);
}

bool backproject_parallelBeam_gpu(torch::Tensor& g_tensor, torch::Tensor& f_tensor, int whichGPU, int whichProjector, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, torch::Tensor& phis_tensor, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    float* phis = phis_tensor.data_ptr<float>();
    
    tomographicModels tempModel;
    tempModel.set_GPU(whichGPU);
    tempModel.set_parallelBeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis);
    tempModel.set_volume(numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ);
    tempModel.set_projector(whichProjector);
    return tempModel.backproject_gpu(g, f);
}

#endif


//*
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("project_cpu", &project_cpu, "forward project on CPU");
    m.def("backproject_cpu", &backproject_cpu, "back project on CPU");
    m.def("filterProjections_cpu", &filterProjections_cpu, "filter projections on CPU");
    m.def("FBP_cpu", &FBP_cpu, "FBP on CPU");
#ifdef __USE_GPU
    m.def("project_gpu", &project_gpu, "forward project on GPU");
    m.def("backproject_gpu", &backproject_gpu, "back project on GPU");
    m.def("filterProjections_gpu", &filterProjections_gpu, "filter projections on GPU");
    m.def("FBP_gpu", &FBP_gpu, "FBP on GPU");
#endif
    m.def("print_parameters", &print_parameters, "print current parameters");
    m.def("save_param", &saveParamsToFile, "save current parameters to file");
    m.def("set_gpu", &set_gpu, "set GPU device (0, 1, ...) ");
    m.def("set_gpus", &set_gpus, "set GPU device (0, 1, ...) ");
    m.def("set_projector", &set_projector, "");
    m.def("set_volumeDimensionOrder", &set_volumeDimensionOrder, "");
	m.def("set_axisOfSymmetry", &set_axisOfSymmetry, "");
    m.def("set_conebeam", &set_conebeam, "");
    m.def("set_fanbeam", &set_fanbeam, "");
	m.def("set_parallelbeam", &set_parallelbeam, "");
    m.def("set_modularbeam", &set_modularbeam, "");
	m.def("set_volume", &set_volume, "");
    m.def("set_default_volume", &set_default_volume, "");
    m.def("get_volumeDimensionOrder", &get_volumeDimensionOrder, "");
    m.def("get_projectionDim", &get_projectionDim, "");
    m.def("get_volumeDim", &get_volumeDim, "");
    m.def("create_param", &create_param, "");
    m.def("clearParamList", &clearParamList, "");
    m.def("project_coneBeam_cpu", &project_coneBeam_cpu, "");
    m.def("project_fanBeam_cpu", &project_fanBeam_cpu, "");
    m.def("project_parallelBeam_cpu", &project_parallelBeam_cpu, "");
    m.def("backproject_coneBeam_cpu", &backproject_coneBeam_cpu, "");
    m.def("backproject_fanBeam_cpu", &backproject_fanBeam_cpu, "");
    m.def("backproject_parallelBeam_cpu", &backproject_parallelBeam_cpu, "");
#ifdef __USE_GPU
    m.def("project_coneBeam_gpu", &project_coneBeam_gpu, "");
    m.def("project_fanBeam_gpu", &project_fanBeam_gpu, "");
    m.def("project_parallelBeam_gpu", &project_parallelBeam_gpu, "");
    m.def("backproject_coneBeam_gpu", &backproject_coneBeam_gpu, "");
    m.def("backproject_fanBeam_gpu", &backproject_fanBeam_gpu, "");
    m.def("backproject_parallelBeam_gpu", &backproject_parallelBeam_gpu, "");
#endif
}
//*/

