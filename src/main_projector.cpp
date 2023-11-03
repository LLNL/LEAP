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
#include "parameters.h"
#include "projectors_cpu.h"

#ifdef __USE_GPU
#include "projectors.h"
#include "projectors_SF.h"

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#endif


// this params instance should not be used for pytorch class
parameters g_params;
std::vector<parameters> g_params_list;


bool project_cpu(int param_id, torch::Tensor& g_tensor, torch::Tensor& f_tensor)
{
    parameters* params;
    if (param_id == -1)
        params = &g_params;
    else if (param_id >= 0 && param_id < (int)g_params_list.size())
        params = &g_params_list[param_id];
    else
        return false;

    //CHECK_CONTIGUOUS(g_tensor);
    //CHECK_CONTIGUOUS(f_tensor);
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    
    if (params->geometry == parameters::CONE) {
    	if (params->useSF())
    		return CPUproject_SF_cone(g, f, params);
    	else
        	return CPUproject_cone(g, f, params);
	}
    else if (params->geometry == parameters::PARALLEL) {
		if (params->useSF())
			return CPUproject_SF_parallel(g, f, params);
		else
        	return CPUproject_parallel(g, f, params);
	}
    else if (params->geometry == parameters::FAN) {
        if (params->useSF()) {
        	//return CPUproject_SF_fan(g, f, params);
            return CPUproject_fan(g, f, params);
        }
		else
        	return CPUproject_fan(g, f, params);
    }
    else if (params->geometry == parameters::MODULAR) {
        return CPUproject_modular(g, f, params);
	}
    else {
        return false;
    }
	
    return true;
}

bool backproject_cpu(int param_id, torch::Tensor& g_tensor, torch::Tensor& f_tensor)
{
    parameters* params;
    if (param_id == -1)
        params = &g_params;
    else if (param_id >= 0 && param_id < (int)g_params_list.size())
        params = &g_params_list[param_id];
    else
        return false;
        
    //CHECK_CONTIGUOUS(g_tensor);
    //CHECK_CONTIGUOUS(f_tensor);
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
	
    if (params->geometry == parameters::CONE) {
    	if (params->useSF())
    		return CPUbackproject_SF_cone(g, f, params);
    	else
        	return CPUbackproject_cone(g, f, params);
	}
    else if (params->geometry == parameters::PARALLEL) {
    	if (params->useSF())
    		return CPUbackproject_SF_parallel(g, f, params);
    	else
	        return CPUbackproject_parallel(g, f, params);
	}
    else if (params->geometry == parameters::FAN) {
        return false;
    }
    else if (params->geometry == parameters::MODULAR) {
        return CPUbackproject_modular(g, f, params);
	}
    else {
        return false;
    }

    return true;
}

#ifdef __USE_GPU
bool project_gpu(int param_id, torch::Tensor& g_tensor, torch::Tensor& f_tensor)
{
    parameters* params;
    if (param_id == -1)
        params = &g_params;
    else if (param_id >= 0 && param_id < (int)g_params_list.size())
        params = &g_params_list[param_id];
    else
        return false;
        
    CHECK_INPUT(g_tensor);
    CHECK_INPUT(f_tensor);
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
	
    if (params->geometry == parameters::CONE)
    {
        if (params->useSF())
            return project_SF_cone(g, f, params, false);
        else
            return project_cone(g, f, params, false);
    }
    else if (params->geometry == parameters::PARALLEL)
    {
        if (params->useSF())
            return project_SF_parallel(g, f, params, false);
        else
            return project_parallel(g, f, params, false);
    }
    else if (params->geometry == parameters::FAN) {
        return false;
    }
    else if (params->geometry == parameters::MODULAR) {
        return project_modular(g, f, params, false);
    }
    else {
        return false;
    }

}

bool backproject_gpu(int param_id, torch::Tensor& g_tensor, torch::Tensor& f_tensor)
{
    parameters* params;
    if (param_id == -1)
        params = &g_params;
    else if (param_id >= 0 && param_id < (int)g_params_list.size())
        params = &g_params_list[param_id];
    else
        return false;
        
    CHECK_INPUT(g_tensor);
    CHECK_INPUT(f_tensor);
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();

    if (params->geometry == parameters::CONE)
    {
        if (params->useSF())
            return backproject_SF_cone(g, f, params, false);
        else
            return backproject_cone(g, f, params, false);
    }
    else if (params->geometry == parameters::PARALLEL)
    {
        if (params->useSF())
            return backproject_SF_parallel(g, f, params, false);
        else
            return backproject_parallel(g, f, params, false);
    }
    else if (params->geometry == parameters::FAN) {
        return false;
    }
    else if (params->geometry == parameters::MODULAR) {
        return backproject_modular(g, f, params, false);
    }
    else {
        return false;
    }
}

#endif

bool setGPU(int param_id, int whichGPU)
{
    parameters* params;
    if (param_id == -1)
        params = &g_params;
    else if (param_id >= 0 && param_id < (int)g_params_list.size())
        params = &g_params_list[param_id];
    else
        return false;
        
    params->whichGPU = whichGPU;
    return true;
}

bool setProjector(int param_id, int which)
{
    parameters* params;
    if (param_id == -1)
        params = &g_params;
    else if (param_id >= 0 && param_id < (int)g_params_list.size())
        params = &g_params_list[param_id];
    else
        return false;
        
    if (which == parameters::SEPARABLE_FOOTPRINT)
        params->whichProjector = parameters::SEPARABLE_FOOTPRINT;
    else
        params->whichProjector = 0;
    return true;
}

bool setVolumeDimensionOrder(int param_id, int which)
{
    parameters* params;
    if (param_id == -1)
        params = &g_params;
    else if (param_id >= 0 && param_id < (int)g_params_list.size())
        params = &g_params_list[param_id];
    else
        return false;
    
    if (which == parameters::ZYX)
        params->volumeDimensionOrder = parameters::ZYX;
    else
        params->volumeDimensionOrder = parameters::XYZ;
    return true;
}

bool set_axisOfSymmetry(int param_id, float axisOfSymmetry)
{
    parameters* params;
    if (param_id == -1)
        params = &g_params;
    else if (param_id >= 0 && param_id < (int)g_params_list.size())
        params = &g_params_list[param_id];
    else
        return false;
        
    params->axisOfSymmetry = axisOfSymmetry;
    return true;
}

bool printParameters(int param_id)
{
    parameters* params;
    if (param_id == -1)
        params = &g_params;
    else if (param_id >= 0 && param_id < (int)g_params_list.size())
        params = &g_params_list[param_id];
    else
        return false;
        
    params->printAll();
    return true;
}

bool saveParamsToFile(int param_id, std::string param_fn)
{
    parameters* params;
    if (param_id == -1)
        params = &g_params;
    else if (param_id >= 0 && param_id < (int)g_params_list.size())
        params = &g_params_list[param_id];
    else
        return false;

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
    parameters* params;
    if (param_id == -1)
        params = &g_params;
    else if (param_id >= 0 && param_id < (int)g_params_list.size())
        params = &g_params_list[param_id];
    else
        return false;
        
	params->clearAll();
	params->setDefaults(1);
	return true;
}

bool getVolumeDim(int param_id, torch::Tensor& dim_tensor)
{
    parameters* params;
    if (param_id == -1)
        params = &g_params;
    else if (param_id >= 0 && param_id < (int)g_params_list.size())
        params = &g_params_list[param_id];
    else
        return false;

    int* dim = dim_tensor.data_ptr<int>();
    dim[0] = params->numX;
    dim[1] = params->numY;
    dim[2] = params->numZ;
    return true;
}

bool getProjectionDim(int param_id, torch::Tensor& dim_tensor)
{
    parameters* params;
    if (param_id == -1)
        params = &g_params;
    else if (param_id >= 0 && param_id < (int)g_params_list.size())
        params = &g_params_list[param_id];
    else
        return false;

    int* dim = dim_tensor.data_ptr<int>();
    dim[0] = params->numAngles;
    dim[1] = params->numRows;
    dim[2] = params->numCols;
    return true;
}

bool getVolumeDimensionOrder(int param_id, int& which)
{
    parameters* params;
    if (param_id == -1)
        params = &g_params;
    else if (param_id >= 0 && param_id < (int)g_params_list.size())
        params = &g_params_list[param_id];
    else
        return false;

    which = params->volumeDimensionOrder;
    return true;
}

bool setConeBeamParams(int param_id, int numAngles, int numRows, int numCols, 
                       float pixelHeight, float pixelWidth, float centerRow, float centerCol, 
                       float angularRange, torch::Tensor& phis_tensor, float sod, float sdd)
{
    parameters* params;
    if (param_id == -1)
        params = &g_params;
    else if (param_id >= 0 && param_id < (int)g_params_list.size())
        params = &g_params_list[param_id];
    else
        return false;

	float* phis = phis_tensor.data_ptr<float>();

    params->geometry = parameters::CONE;
    params->detectorType = parameters::FLAT;
    params->numAngles = numAngles;
    params->numRows = numRows;
    params->numCols = numCols;
    params->pixelHeight = pixelHeight;
    params->pixelWidth = pixelWidth;
    params->centerRow = centerRow;
    params->centerCol = centerCol;
    params->angularRange = angularRange;
    params->setAngles(phis, numAngles);
    params->sod = sod;
    params->sdd = sdd;
    return params->geometryDefined();
}

bool setParallelBeamParams(int param_id, int numAngles, int numRows, int numCols, 
                           float pixelHeight, float pixelWidth, float centerRow, float centerCol, 
                           float angularRange, torch::Tensor& phis_tensor)
{
    parameters* params;
    if (param_id == -1)
        params = &g_params;
    else if (param_id >= 0 && param_id < (int)g_params_list.size())
        params = &g_params_list[param_id];
    else
        return false;
        
	float* phis = phis_tensor.data_ptr<float>();

    params->geometry = parameters::PARALLEL;
    params->numAngles = numAngles;
    params->numRows = numRows;
    params->numCols = numCols;
    params->pixelHeight = pixelHeight;
    params->pixelWidth = pixelWidth;
    params->centerRow = centerRow;
    params->centerCol = centerCol;
    params->angularRange = angularRange;
    params->setAngles(phis, numAngles);
    return params->geometryDefined();
}

bool setFanBeamParams(int param_id, int numAngles, int numRows, int numCols, 
                      float pixelHeight, float pixelWidth, float centerRow, float centerCol, 
                      float angularRange, torch::Tensor& phis_tensor, float sod, float sdd)
{
    parameters* params;
    if (param_id == -1)
        params = &g_params;
    else if (param_id >= 0 && param_id < (int)g_params_list.size())
        params = &g_params_list[param_id];
    else
        return false;
        
	float* phis = phis_tensor.data_ptr<float>();

    params->geometry = parameters::FAN;
    params->detectorType = parameters::FLAT;
    params->numAngles = numAngles;
    params->numRows = numRows;
    params->numCols = numCols;
    params->pixelHeight = pixelHeight;
    params->pixelWidth = pixelWidth;
    params->centerRow = centerRow;
    params->centerCol = centerCol;
    params->angularRange = angularRange;
    params->setAngles(phis, numAngles);
    params->sod = sod;
    params->sdd = sdd;
    return params->geometryDefined();
}

bool setModularBeamParams(int param_id, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, 
                          torch::Tensor& sourcePositions_in_tensor, torch::Tensor& moduleCenters_in_tensor, 
                          torch::Tensor& rowVectors_in_tensor, torch::Tensor& colVectors_in_tensor)
{
    parameters* params;
    if (param_id == -1)
        params = &g_params;
    else if (param_id >= 0 && param_id < (int)g_params_list.size())
        params = &g_params_list[param_id];
    else
        return false;
        
	float* sourcePositions_in = sourcePositions_in_tensor.data_ptr<float>();
	float* moduleCenters_in = moduleCenters_in_tensor.data_ptr<float>();
	float* rowVectors_in = rowVectors_in_tensor.data_ptr<float>();
	float* colVectors_in = colVectors_in_tensor.data_ptr<float>();

    params->geometry = parameters::MODULAR;
    params->numAngles = numAngles;
    params->numRows = numRows;
    params->numCols = numCols;
    params->pixelHeight = pixelHeight;
    params->pixelWidth = pixelWidth;
    params->setSourcesAndModules(sourcePositions_in, moduleCenters_in, rowVectors_in, colVectors_in, numAngles);
    return params->geometryDefined();
}

bool setVolumeParams(int param_id, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
    parameters* params;
    if (param_id == -1)
        params = &g_params;
    else if (param_id >= 0 && param_id < (int)g_params_list.size())
        params = &g_params_list[param_id];
    else
        return false;
        
    params->numX = numX;
    params->numY = numY;
    params->numZ = numZ;
    params->voxelWidth = voxelWidth;
    params->voxelHeight = voxelHeight;
    params->offsetX = offsetX;
    params->offsetY = offsetY;
    params->offsetZ = offsetZ;
    return params->volumeDefined();
}

int createParams()
{
    parameters params;
    g_params_list.push_back(params);
    return (int)g_params_list.size()-1;
}

bool clearParamsList()
{
    g_params_list.clear();
    return true;
}

// for multiple independent projector instances
bool project_cpu_ConeBeam(torch::Tensor& g_tensor, torch::Tensor& f_tensor, int whichProjector, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, torch::Tensor& phis_tensor, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    float* phis = phis_tensor.data_ptr<float>();

    parameters tempParams;
    if (whichProjector == parameters::SEPARABLE_FOOTPRINT)
        tempParams.whichProjector = parameters::SEPARABLE_FOOTPRINT;
    else
        tempParams.whichProjector = 0;
    tempParams.geometry = parameters::CONE;
    tempParams.detectorType = parameters::FLAT;
    tempParams.sod = sod;
    tempParams.sdd = sdd;
    tempParams.pixelWidth = pixelWidth;
    tempParams.pixelHeight = pixelHeight;
    tempParams.numCols = numCols;
    tempParams.numRows = numRows;
    tempParams.numAngles = numAngles;
    tempParams.centerCol = centerCol;
    tempParams.centerRow = centerRow;
    tempParams.setAngles(phis, numAngles);
    
    tempParams.numX = numX;
    tempParams.numY = numY;
    tempParams.numZ = numZ;
    tempParams.voxelWidth = voxelWidth;
    tempParams.voxelHeight = voxelHeight;
    tempParams.offsetX = offsetX;
    tempParams.offsetY = offsetY;
    tempParams.offsetZ = offsetZ;
    
    if (tempParams.allDefined() == false || g == NULL || f == NULL)
        return false;
    
    if (tempParams.useSF())
        return CPUproject_SF_cone(g, f, &tempParams);
    else
        return CPUproject_cone(g, f, &tempParams);
}

bool project_cpu_ParallelBeam(torch::Tensor& g_tensor, torch::Tensor& f_tensor, int whichProjector, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, torch::Tensor& phis_tensor, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    float* phis = phis_tensor.data_ptr<float>();
    
    parameters tempParams;
    if (whichProjector == parameters::SEPARABLE_FOOTPRINT)
        tempParams.whichProjector = parameters::SEPARABLE_FOOTPRINT;
    else
        tempParams.whichProjector = 0;
    tempParams.geometry = parameters::PARALLEL;
    tempParams.detectorType = parameters::FLAT;
    tempParams.pixelWidth = pixelWidth;
    tempParams.pixelHeight = pixelHeight;
    tempParams.numCols = numCols;
    tempParams.numRows = numRows;
    tempParams.numAngles = numAngles;
    tempParams.centerCol = centerCol;
    tempParams.centerRow = centerRow;
    tempParams.setAngles(phis, numAngles);

    tempParams.numX = numX;
    tempParams.numY = numY;
    tempParams.numZ = numZ;
    tempParams.voxelWidth = voxelWidth;
    tempParams.voxelHeight = voxelHeight;
    tempParams.offsetX = offsetX;
    tempParams.offsetY = offsetY;
    tempParams.offsetZ = offsetZ;

    if (tempParams.allDefined() == false || g == NULL || f == NULL)
        return false;

    if (tempParams.useSF())
        return CPUproject_SF_parallel(g, f, &tempParams);
    else
        return CPUproject_parallel(g, f, &tempParams);
}

bool backproject_cpu_ConeBeam(torch::Tensor& g_tensor, torch::Tensor& f_tensor, int whichProjector, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, torch::Tensor& phis_tensor, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    float* phis = phis_tensor.data_ptr<float>();
    
    parameters tempParams;
    if (whichProjector == parameters::SEPARABLE_FOOTPRINT)
        tempParams.whichProjector = parameters::SEPARABLE_FOOTPRINT;
    else
        tempParams.whichProjector = 0;
    tempParams.geometry = parameters::CONE;
    tempParams.detectorType = parameters::FLAT;
    tempParams.sod = sod;
    tempParams.sdd = sdd;
    tempParams.pixelWidth = pixelWidth;
    tempParams.pixelHeight = pixelHeight;
    tempParams.numCols = numCols;
    tempParams.numRows = numRows;
    tempParams.numAngles = numAngles;
    tempParams.centerCol = centerCol;
    tempParams.centerRow = centerRow;
    tempParams.setAngles(phis, numAngles);
    
    tempParams.numX = numX;
    tempParams.numY = numY;
    tempParams.numZ = numZ;
    tempParams.voxelWidth = voxelWidth;
    tempParams.voxelHeight = voxelHeight;
    tempParams.offsetX = offsetX;
    tempParams.offsetY = offsetY;
    tempParams.offsetZ = offsetZ;
    
    if (tempParams.allDefined() == false || g == NULL || f == NULL)
        return false;
        
    if (tempParams.useSF())
        return CPUbackproject_SF_cone(g, f, &tempParams);
    else
        return CPUbackproject_cone(g, f, &tempParams);
}


bool backproject_cpu_ParallelBeam(torch::Tensor& g_tensor, torch::Tensor& f_tensor, int whichProjector, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, torch::Tensor& phis_tensor, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    float* phis = phis_tensor.data_ptr<float>();
    
    parameters tempParams;
    if (whichProjector == parameters::SEPARABLE_FOOTPRINT)
        tempParams.whichProjector = parameters::SEPARABLE_FOOTPRINT;
    else
        tempParams.whichProjector = 0;
    tempParams.geometry = parameters::PARALLEL;
    tempParams.detectorType = parameters::FLAT;
    tempParams.pixelWidth = pixelWidth;
    tempParams.pixelHeight = pixelHeight;
    tempParams.numCols = numCols;
    tempParams.numRows = numRows;
    tempParams.numAngles = numAngles;
    tempParams.centerCol = centerCol;
    tempParams.centerRow = centerRow;
    tempParams.setAngles(phis, numAngles);

    tempParams.numX = numX;
    tempParams.numY = numY;
    tempParams.numZ = numZ;
    tempParams.voxelWidth = voxelWidth;
    tempParams.voxelHeight = voxelHeight;
    tempParams.offsetX = offsetX;
    tempParams.offsetY = offsetY;
    tempParams.offsetZ = offsetZ;

    if (tempParams.allDefined() == false || g == NULL || f == NULL)
        return false;

    if (tempParams.useSF())
        return CPUbackproject_SF_parallel(g, f, &tempParams);
    else
        return CPUbackproject_parallel(g, f, &tempParams);
}

#ifdef __USE_GPU
bool project_gpu_ConeBeam(torch::Tensor& g_tensor, torch::Tensor& f_tensor, int whichGPU, int whichProjector, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, torch::Tensor& phis_tensor, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    float* phis = phis_tensor.data_ptr<float>();
    
    parameters tempParams;
    tempParams.whichGPU = whichGPU;
    if (whichProjector == parameters::SEPARABLE_FOOTPRINT)
        tempParams.whichProjector = parameters::SEPARABLE_FOOTPRINT;
    else
        tempParams.whichProjector = 0;
    tempParams.geometry = parameters::CONE;
    tempParams.detectorType = parameters::FLAT;
    tempParams.sod = sod;
    tempParams.sdd = sdd;
    tempParams.pixelWidth = pixelWidth;
    tempParams.pixelHeight = pixelHeight;
    tempParams.numCols = numCols;
    tempParams.numRows = numRows;
    tempParams.numAngles = numAngles;
    tempParams.centerCol = centerCol;
    tempParams.centerRow = centerRow;
    tempParams.setAngles(phis, numAngles);
    
    tempParams.numX = numX;
    tempParams.numY = numY;
    tempParams.numZ = numZ;
    tempParams.voxelWidth = voxelWidth;
    tempParams.voxelHeight = voxelHeight;
    tempParams.offsetX = offsetX;
    tempParams.offsetY = offsetY;
    tempParams.offsetZ = offsetZ;
    
    if (tempParams.allDefined() == false || g == NULL || f == NULL)
        return false;
        
    if (tempParams.useSF())
        return project_SF_cone(g, f, &tempParams, false);
    else
        return project_cone(g, f, &tempParams, false);
}

bool project_gpu_ParallelBeam(torch::Tensor& g_tensor, torch::Tensor& f_tensor, int whichGPU, int whichProjector, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, torch::Tensor& phis_tensor, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    float* phis = phis_tensor.data_ptr<float>();
    
    parameters tempParams;
    tempParams.whichGPU = whichGPU;
    if (whichProjector == parameters::SEPARABLE_FOOTPRINT)
        tempParams.whichProjector = parameters::SEPARABLE_FOOTPRINT;
    else
        tempParams.whichProjector = 0;
    tempParams.geometry = parameters::PARALLEL;
    tempParams.detectorType = parameters::FLAT;
    tempParams.pixelWidth = pixelWidth;
    tempParams.pixelHeight = pixelHeight;
    tempParams.numCols = numCols;
    tempParams.numRows = numRows;
    tempParams.numAngles = numAngles;
    tempParams.centerCol = centerCol;
    tempParams.centerRow = centerRow;
    tempParams.setAngles(phis, numAngles);

    tempParams.numX = numX;
    tempParams.numY = numY;
    tempParams.numZ = numZ;
    tempParams.voxelWidth = voxelWidth;
    tempParams.voxelHeight = voxelHeight;
    tempParams.offsetX = offsetX;
    tempParams.offsetY = offsetY;
    tempParams.offsetZ = offsetZ;

    if (tempParams.allDefined() == false || g == NULL || f == NULL)
        return false;
    
    if (tempParams.useSF())
        return project_SF_parallel(g, f, &tempParams, false);
    else
        return project_parallel(g, f, &tempParams, false);
}

bool backproject_gpu_ConeBeam(torch::Tensor& g_tensor, torch::Tensor& f_tensor, int whichGPU, int whichProjector, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, torch::Tensor& phis_tensor, float sod, float sdd, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    float* phis = phis_tensor.data_ptr<float>();
    
    parameters tempParams;
    tempParams.whichGPU = whichGPU;
    if (whichProjector == parameters::SEPARABLE_FOOTPRINT)
        tempParams.whichProjector = parameters::SEPARABLE_FOOTPRINT;
    else
        tempParams.whichProjector = 0;
    tempParams.geometry = parameters::CONE;
    tempParams.detectorType = parameters::FLAT;
    tempParams.sod = sod;
    tempParams.sdd = sdd;
    tempParams.pixelWidth = pixelWidth;
    tempParams.pixelHeight = pixelHeight;
    tempParams.numCols = numCols;
    tempParams.numRows = numRows;
    tempParams.numAngles = numAngles;
    tempParams.centerCol = centerCol;
    tempParams.centerRow = centerRow;
    tempParams.setAngles(phis, numAngles);
    
    tempParams.numX = numX;
    tempParams.numY = numY;
    tempParams.numZ = numZ;
    tempParams.voxelWidth = voxelWidth;
    tempParams.voxelHeight = voxelHeight;
    tempParams.offsetX = offsetX;
    tempParams.offsetY = offsetY;
    tempParams.offsetZ = offsetZ;
    
    if (tempParams.allDefined() == false || g == NULL || f == NULL)
        return false;
    
    if (tempParams.useSF())
        return backproject_SF_cone(g, f, &tempParams, false);
    else
        return backproject_cone(g, f, &tempParams, false);
}


bool backproject_gpu_ParallelBeam(torch::Tensor& g_tensor, torch::Tensor& f_tensor, int whichGPU, int whichProjector, int numAngles, int numRows, int numCols, float pixelHeight, float pixelWidth, float centerRow, float centerCol, torch::Tensor& phis_tensor, int numX, int numY, int numZ, float voxelWidth, float voxelHeight, float offsetX, float offsetY, float offsetZ)
{
    float* g = g_tensor.data_ptr<float>();
    float* f = f_tensor.data_ptr<float>();
    float* phis = phis_tensor.data_ptr<float>();
    
    parameters tempParams;
    tempParams.whichGPU = whichGPU;
    if (whichProjector == parameters::SEPARABLE_FOOTPRINT)
        tempParams.whichProjector = parameters::SEPARABLE_FOOTPRINT;
    else
        tempParams.whichProjector = 0;
    tempParams.geometry = parameters::PARALLEL;
    tempParams.detectorType = parameters::FLAT;
    tempParams.pixelWidth = pixelWidth;
    tempParams.pixelHeight = pixelHeight;
    tempParams.numCols = numCols;
    tempParams.numRows = numRows;
    tempParams.numAngles = numAngles;
    tempParams.centerCol = centerCol;
    tempParams.centerRow = centerRow;
    tempParams.setAngles(phis, numAngles);

    tempParams.numX = numX;
    tempParams.numY = numY;
    tempParams.numZ = numZ;
    tempParams.voxelWidth = voxelWidth;
    tempParams.voxelHeight = voxelHeight;
    tempParams.offsetX = offsetX;
    tempParams.offsetY = offsetY;
    tempParams.offsetZ = offsetZ;

    if (tempParams.allDefined() == false || g == NULL || f == NULL)
        return false;
    
    if (tempParams.useSF())
        return backproject_SF_parallel(g, f, &tempParams, false);
    else
        return backproject_parallel(g, f, &tempParams, false);
}
#endif


//*
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("project_cpu", &project_cpu, "forward project on CPU");
    m.def("backproject_cpu", &backproject_cpu, "back project on CPU");
#ifdef __USE_GPU
    m.def("project_gpu", &project_gpu, "forward project on GPU");
    m.def("backproject_gpu", &backproject_gpu, "back project on GPU");
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

