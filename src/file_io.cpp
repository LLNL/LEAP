////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// c++ header for some file I/O routines
////////////////////////////////////////////////////////////////////////////////
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <math.h>
#include <iostream>
#include <fstream>
#include "file_io.h"

bool saveParametersToFile(const char* param_fn, parameters* params)
{
	if (param_fn == NULL || params == NULL)
		return false;

#ifdef USE_OLD_PARAM_NAMES
	std::string numX = "img_dimx";
	std::string numY = "img_dimy";
	std::string numZ = "img_dimz";
	std::string voxelWidth = "img_pwidth";
	std::string voxelHeight = "img_pheight";
	std::string offsetX = "img_offsetx";
	std::string offsetY = "img_offsety";
	std::string offsetZ = "img_offsetz";

	std::string geometry = "proj_geometry";
	std::string numAngles = "proj_nangles";
	std::string numRows = "proj_nrows";
	std::string numCols = "proj_ncols";
	std::string pixelHeight = "proj_pheight";
	std::string pixelWidth = "proj_pwidth";
	std::string centerRow = "proj_crow";
	std::string centerCol = "proj_ccol";
	std::string angularRange = "proj_arange";
	std::string phis = "proj_phis";
	std::string sod = "proj_sod";
	std::string sdd = "proj_sdd";
	std::string tau = "proj_tau";
	std::string helicalPitch = "proj_helicalpitch";
	std::string axisOfSymmetry = "proj_axisofsymmetry";

	std::string sourcePositions = "proj_srcpos";
	std::string moduleCenters = "proj_modcenter";
	std::string rowVectors = "proj_rowvec";
	std::string colVectors = "proj_colvec";

	std::string muCoeff = "mucoeff";
	std::string muRadius = "muradius";
#else
	std::string numX = "numX";
	std::string numY = "numY";
	std::string numZ = "numZ";
	std::string voxelWidth = "voxelWidth";
	std::string voxelHeight = "voxelHeight";
	std::string offsetX = "offsetX";
	std::string offsetY = "offsetY";
	std::string offsetZ = "offsetZ";

	std::string geometry = "geometry";
	std::string numAngles = "numAngles";
	std::string numRows = "numRows";
	std::string numCols = "numCols";
	std::string pixelHeight = "pixelHeight";
	std::string pixelWidth = "pixelWidth";
	std::string centerRow = "centerRow";
	std::string centerCol = "centerCol";
	std::string angularRange = "angularRange";
	std::string phis = "phis";
	std::string sod = "sod";
	std::string sdd = "sdd";
	std::string tau = "tau";
	std::string helicalPitch = "helicalPitch";
	std::string axisOfSymmetry = "axisOfSymmetry";

	std::string sourcePositions = "sourcePositions";
	std::string moduleCenters = "moduleCenters";
	std::string rowVectors = "rowVectors";
	std::string colVectors = "colVectors";

	std::string muCoeff = "muCoeff";
	std::string muRadius = "muRadius";
#endif

	std::string phis_strs;
	if (params->phis != NULL)
	{
		for (int i = 0; i < params->numAngles; i++)
		{
			float phis = (params->phis[i] + 0.5 * PI) * 180.0 / PI;
			char phis_str[64];
#ifdef WIN32
			sprintf_s(phis_str, " %e", phis);
#else
			sprintf(phis_str, " %e", phis);
#endif
			phis_strs += phis_str;
			if (i != params->numAngles - 1)
				phis_strs += ",";
		}
	}
	else
	{
		phis_strs = "";
	}

	std::ofstream param_file;
	param_file.open(param_fn);

	// Save CT Volume Parameters
	param_file << "# CT volume parameters" << std::endl;
	param_file << numX << " = " << params->numX << std::endl;
	param_file << numY << " = " << params->numY << std::endl;
	param_file << numZ << " = " << params->numZ << std::endl;
	param_file << voxelWidth << " = " << std::scientific << params->voxelWidth << std::endl;
	param_file << voxelHeight << " = " << std::scientific << params->voxelHeight << std::endl;
	param_file << offsetX << " = " << std::scientific << params->offsetX << std::endl;
	param_file << offsetY << " = " << std::scientific << params->offsetY << std::endl;
	param_file << offsetZ << " = " << std::scientific << params->offsetZ << std::endl;

	param_file << std::endl;

	// Save CT Geometry Parameters
	param_file << "# CT geometry parameters" << std::endl;
	if (params->geometry == parameters::CONE)
		param_file << geometry << " = " << "cone" << std::endl;
	else if (params->geometry == parameters::PARALLEL)
		param_file << geometry << " = " << "parallel" << std::endl;
	else if (params->geometry == parameters::FAN)
		param_file << geometry << " = " << "fan" << std::endl;
	else if (params->geometry == parameters::MODULAR)
		param_file << geometry << " = " << "modular" << std::endl;

	if (params->geometry == parameters::CONE && params->detectorType == parameters::CURVED)
		param_file << "detectorType = curved" << std::endl;

	param_file << numAngles << " = " << params->numAngles << std::endl;
	param_file << numRows << " = " << params->numRows << std::endl;
	param_file << numCols << " = " << params->numCols << std::endl;
	param_file << pixelHeight << " = " << std::scientific << params->pixelHeight << std::endl;
	param_file << pixelWidth << " = " << std::scientific << params->pixelWidth << std::endl;
	param_file << centerRow << " = " << params->centerRow << std::endl;
	param_file << centerCol << " = " << params->centerCol << std::endl;
	if (params->anglesAreEquispaced())
	{
		param_file << angularRange << " = " << params->angularRange << std::endl;
	}
	else
	{
		param_file << phis << " = " << phis_strs << std::endl;
	}
	param_file << sod << " = " << params->sod << std::endl;
	param_file << sdd << " = " << params->sdd << std::endl;
	if (params->geometry == parameters::CONE || params->geometry == parameters::FAN)
	{
		param_file << tau << " = " << params->tau << std::endl;
	}
	if (params->geometry == parameters::CONE)
	{
		if (fabs(params->helicalPitch) < 1.0e-16)
		{
			param_file << helicalPitch << " = " << 0.0 << std::endl;
		}
		else
		{
			param_file << helicalPitch << " = " << params->helicalPitch << std::endl;
		}
	}
	if (params->isSymmetric())
	{
		param_file << axisOfSymmetry << " = " << params->axisOfSymmetry << std::endl;
	}
	if (params->geometry == parameters::MODULAR)
	{
		std::string sourcePositions_strs;
		std::string moduleCenters_strs;
		std::string rowVectors_strs;
		std::string colVectors_strs;

		for (int i = 0; i < params->numAngles; i++)
		{
			char temp_str[256];

#ifdef WIN32
			sprintf_s(temp_str, "%e, %e, %e", params->sourcePositions[i * 3 + 0], params->sourcePositions[i * 3 + 1], params->sourcePositions[i * 3 + 2]);
#else
			sprintf(temp_str, "%e, %e, %e", params->sourcePositions[i * 3 + 0], params->sourcePositions[i * 3 + 1], params->sourcePositions[i * 3 + 2]);
#endif
			sourcePositions_strs += temp_str;
			if (i != params->numAngles - 1)
				sourcePositions_strs += ", ";

#ifdef WIN32
			sprintf_s(temp_str, "%e, %e, %e", params->moduleCenters[i * 3 + 0], params->moduleCenters[i * 3 + 1], params->moduleCenters[i * 3 + 2]);
#else
			sprintf(temp_str, "%e, %e, %e", params->moduleCenters[i * 3 + 0], params->moduleCenters[i * 3 + 1], params->moduleCenters[i * 3 + 2]);
#endif
			moduleCenters_strs += temp_str;
			if (i != params->numAngles - 1)
				moduleCenters_strs += ", ";

#ifdef WIN32
			sprintf_s(temp_str, "%e, %e, %e", params->rowVectors[i * 3 + 0], params->rowVectors[i * 3 + 1], params->rowVectors[i * 3 + 2]);
#else
			sprintf(temp_str, "%e, %e, %e", params->rowVectors[i * 3 + 0], params->rowVectors[i * 3 + 1], params->rowVectors[i * 3 + 2]);
#endif
			rowVectors_strs += temp_str;
			if (i != params->numAngles - 1)
				rowVectors_strs += ", ";

#ifdef WIN32
			sprintf_s(temp_str, "%e, %e, %e", params->colVectors[i * 3 + 0], params->colVectors[i * 3 + 1], params->colVectors[i * 3 + 2]);
#else
			sprintf(temp_str, "%e, %e, %e", params->colVectors[i * 3 + 0], params->colVectors[i * 3 + 1], params->colVectors[i * 3 + 2]);
#endif
			colVectors_strs += temp_str;
			if (i != params->numAngles - 1)
				colVectors_strs += ", ";
		}

		param_file << sourcePositions << " = " << sourcePositions_strs << std::endl;
		param_file << moduleCenters << " = " << moduleCenters_strs << std::endl;
		param_file << rowVectors << " = " << rowVectors_strs << std::endl;
		param_file << colVectors << " = " << colVectors_strs << std::endl;
	}
	if (params->geometry == parameters::PARALLEL)
	{
		if (params->muCoeff != 0.0 && params->muRadius > 0.0)
		{
			param_file << muCoeff << " = " << params->muCoeff << std::endl;
			param_file << muRadius << " = " << params->muRadius << std::endl;
		}
	}

	param_file.close();

	return true;
}

bool tif_image_size(const char* fileName, int& numRows, int& numCols)
{
    return false;
}

bool read_tif(const char* fileName, float* data, int firstRow, int lastRow)
{
    return false;
}

bool save_tif(const char* fileName, float* data, int numRows, int numCols, bool quantize)
{
	return false;
}
