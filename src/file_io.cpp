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
#include <string.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "file_io.h"
#include "cpu_utils.h"

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
	std::string tiltAngle = "proj_tiltAngle";
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
	std::string tiltAngle = "tiltAngle";
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
	else if (params->geometry == parameters::CONE_PARALLEL)
		param_file << geometry << " = " << "cone_parallel" << std::endl;

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
		if (params->T_phi() < 0.0)
			param_file << angularRange << " = " << -params->angularRange << std::endl;
		else
			param_file << angularRange << " = " << params->angularRange << std::endl;
	}
	else
	{
		param_file << phis << " = " << phis_strs << std::endl;
	}
	param_file << sod << " = " << params->sod << std::endl;
	param_file << sdd << " = " << params->sdd << std::endl;
	if (params->geometry == parameters::CONE || params->geometry == parameters::CONE_PARALLEL || params->geometry == parameters::FAN)
	{
		param_file << tau << " = " << params->tau << std::endl;
	}
	if (params->geometry == parameters::CONE && params->detectorType == parameters::FLAT)
	{
		param_file << tiltAngle << " = " << params->tiltAngle << std::endl;
	}
	else
	{
		param_file << tiltAngle << " = " << 0.0 << std::endl;
	}
	if (params->geometry == parameters::CONE || params->geometry == parameters::CONE_PARALLEL)
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

bool read_header(char* fileName, int* shape, float* size, float* slope_and_offset)
{
	if (fileName == NULL || fileName[0] == 0 || shape == NULL || size == NULL || slope_and_offset == NULL)
		return false;

	ImageHeader H;
	ImageHeader* h = &H;
	sprintf(h->fileName, "%s", fileName);
	FILE* fdes = read_header_leave_open(h);
	if (fdes == NULL)
		return false;

	shape[0] = h->numRows;
	shape[1] = h->numCols;
	size[0] = h->pixelHeight;
	size[1] = h->pixelWidth;
	slope_and_offset[0] = h->slope;
	slope_and_offset[1] = h->offset;

	fclose(fdes);
	return true;
}

FILE* read_header_leave_open(ImageHeader* h)
{
	char cdata[255], cslope[32], coffset[32], cequals[32];
	//2-byte integers
	uint16 n42, nde, tag[50], field_type[50], BitsPerSample = 32, SampleFormat = 3;
	//4-byte integers
	uint32 ifd_offset, ft_num_vals[50], val_offset[50];
	uint32 StripOffset = 0, RowsPerStrip, StripByteCounts = 0;
	uint32 XRes[2], YRes[2];
	double XResolution = 1.0, YResolution = 1.0;
	FILE* fptr;

	//start with some default units, in case they aren't explicitly defined
	h->numFrames = 1;
	h->bigEndian = false;
	h->datatype = 4;
	h->slope = 1.0;
	h->offset = 0.0;
	h->compressionType = 1; // no compression

	if ((fptr = fopen(h->fileName, "rb")) == NULL)
	{
		printf("error opening file %s.\n", h->fileName);
		return NULL;
	}

	fread(cdata, sizeof(char), (size_t)2, fptr);
	if (cdata[0] != 'I')
		h->bigEndian = true;

	fread(&n42, sizeof(short), (size_t)1, fptr);
	if (h->bigEndian)
		n42 = swapEndian(short(n42));
	if (n42 != 42)
		printf("ERROR:  Not a tif file (%s)!!!\n", h->fileName);

	fread(&ifd_offset, sizeof(int), (size_t)1, fptr); //bytes 5-7: byte offset for first IFD
	if (h->bigEndian)
		ifd_offset = swapEndian(int(ifd_offset));

	if ((fseek(fptr, ifd_offset, SEEK_SET)) != 0)
	{
		printf("error in fseek\n");
		fclose(fptr);
		return NULL;
	}

	fread(&nde, sizeof(short), (size_t)1, fptr); //2 byte count; # of directory entries
	if (h->bigEndian)
		nde = swapEndian(short(nde));

	for (int i = 0; i < nde; i++) //read through the directory entries
	{
		fread(&tag[i], sizeof(short), (size_t)1, fptr);
		fread(&field_type[i], sizeof(short), (size_t)1, fptr);
		fread(&ft_num_vals[i], sizeof(int), (size_t)1, fptr);
		fread(&val_offset[i], sizeof(int), (size_t)1, fptr);
		if (h->bigEndian)
		{
			tag[i] = swapEndian(short(tag[i]));
			field_type[i] = swapEndian(short(field_type[i]));
			ft_num_vals[i] = swapEndian(int(ft_num_vals[i]));

			if (field_type[i] == 3)
				val_offset[i] = swapEndian(short(val_offset[i]));
			else if (field_type[i] == 4 || field_type[i] == 5)
				val_offset[i] = swapEndian(val_offset[i]);
		}
	}

	/*
	fread(&val_offset[i], sizeof(int), (size_t)1, fptr);  //check for more than 1 image
	if (h->bigEndian)
		val_offset[i] = swapEndian(val_offset[i]);
	if (val_offset[i] != 0)
	{
		printf("Error - more data than 1 image.\n");
		fclose(fptr);
		return NULL;
	}
	//*/

	for (int i = 0; i < nde; i++)
	{
		if (tag[i] == 256)
			h->numCols = val_offset[i];
		if (tag[i] == 257)
			h->numRows = val_offset[i];
		if (tag[i] == 258)
		{
			BitsPerSample = (uint16)val_offset[i];
			//assume these in case no 339 tag
			if (BitsPerSample == 8)
			{
				h->datatype = 0;
				SampleFormat = 1;
			}
			if (BitsPerSample == 16)
			{
				h->datatype = 1;
				SampleFormat = 1;
			}
			if (BitsPerSample == 32)
			{
				h->datatype = 3;
				SampleFormat = 3;
			}
		}
		if (tag[i] == 259)
			h->compressionType = val_offset[i];
		if (tag[i] == 270)  //the comment line where slope offset is stored
		{
			fseek(fptr, val_offset[i], SEEK_SET);
			fread(cdata, sizeof(char), (size_t)ft_num_vals[i], fptr);
			//this is something like: slope = 2.3346E-005 '\13' offset = 0.00000E000
			sscanf(cdata, "%s %s %f %s %s %f", cslope, cequals, &h->slope, coffset, cequals, &h->offset);
			if (strcmp(cslope, "slope") != 0)
				h->slope = 0.0;

			if (h->bigEndian)
			{
				h->slope = swapEndian(h->slope);
				h->offset = swapEndian(h->offset);
			}
		}
		if (tag[i] == 273)
			StripOffset = val_offset[i];
		if (tag[i] == 278)
			RowsPerStrip = val_offset[i];
		if (tag[i] == 279)
			StripByteCounts = val_offset[i];
		if (tag[i] == 282)
		{
			fseek(fptr, val_offset[i], SEEK_SET);
			fread(XRes, sizeof(unsigned int), (size_t)2, fptr);//tiff rational type; two ints

			if (h->bigEndian)
			{
				XRes[0] = swapEndian(XRes[0]);
				XRes[1] = swapEndian(XRes[1]);
			}

			XResolution = 10.0 * float(XRes[1]) / float(XRes[0]);
		}
		if (tag[i] == 283)
		{
			fseek(fptr, val_offset[i], SEEK_SET);
			fread(YRes, sizeof(unsigned int), (size_t)2, fptr);

			if (h->bigEndian)
			{
				YRes[0] = swapEndian(YRes[0]);
				YRes[1] = swapEndian(YRes[1]);
			}

			YResolution = 10.0 * float(YRes[1]) / float(YRes[0]);
		}
		if (tag[i] == 339)
			SampleFormat = uint16(val_offset[i]);
	}

	//Determine datatype from SampleFormat and BitsPerSample
	if (SampleFormat == 1)
	{
		if (BitsPerSample == 8)
			h->datatype = 0; // 0: 8-bit unsigned char
		else if (BitsPerSample == 16)
			h->datatype = 1; // 1: 16-bit unsigned short
		else
		{
			printf("error: SampleFormat: %d, BitsPerSample: %d not yet supported.\n", SampleFormat, BitsPerSample);
			fclose(fptr);
			return NULL;
		}
	}
	else if (SampleFormat == 2)
	{
		if (BitsPerSample == 32)
			h->datatype = 2; // 2: 32-bit signed int
		else
		{
			printf("error: SampleFormat: %d, BitsPerSample: %d not yet supported.\n", SampleFormat, BitsPerSample);
			fclose(fptr);
			return NULL;
		}
	}
	else if (SampleFormat == 3)
	{
		if (BitsPerSample == 32)
			h->datatype = 3; // 3: 32-bit float
		else if (BitsPerSample == 64)
			h->datatype = 4; // 4: 64-bit float
		else
		{
			printf("error: SampleFormat: %d, BitsPerSample: %d not yet supported.\n", SampleFormat, BitsPerSample);
			fclose(fptr);
			return NULL;
		}
	}
	else
	{
		printf("error: SampleFormat: %d, BitsPerSample: %d not yet supported.\n", SampleFormat, BitsPerSample);
		fclose(fptr);
		return NULL;
	}

	h->pixelWidth = XResolution;
	h->pixelHeight = YResolution;
	h->bytes_of_data = StripByteCounts;
	h->offset_to_data = StripOffset;

	return fptr;
}

float* load_tif(char* fileName, float* data)
{
	return load_tif_rows(fileName, 0, -1, data);
}

float* load_tif_roi(char* fileName, int firstRow, int lastRow, int firstCol, int lastCol, float* data)
{
	if (fileName == NULL || fileName[0] == 0)
		return NULL;

	ImageHeader H;
	ImageHeader* h = &H;
	sprintf(h->fileName, "%s", fileName);

	float* data_all_cols = load_tif_rows(h, firstRow, lastRow);
	if (data_all_cols == NULL)
		return NULL;
	else
	{
		int numCols_source = h->numCols;
		int numCols_target = lastCol - firstCol + 1;
		if (data == NULL)
			data = (float*)malloc(sizeof(float) * size_t((lastRow - firstRow + 1) * numCols_target));
		for (int iRow = 0; iRow < lastRow - firstRow + 1; iRow++)
		{
			for (int iCol = firstCol; iCol <= lastCol; iCol++)
				data[iRow * numCols_target + iCol - firstCol] = data_all_cols[iRow * numCols_source + iCol];
		}
		free(data_all_cols);
		return data;
	}
}

float* load_tif_rows(char* fileName, int firstRow, int lastRow, float* data)
{
	if (fileName == NULL || fileName[0] == 0)
		return NULL;

	ImageHeader H;
	ImageHeader* h = &H;
	sprintf(h->fileName, "%s", fileName);
	return load_tif_rows(h, firstRow, lastRow, data);
}

float* load_tif_rows(ImageHeader* h, int firstRow, int lastRow, float* data)
{
	if (h == NULL)
		return NULL;

	FILE* fdes = read_header_leave_open(h);
	if (fdes == NULL)
		return NULL;

	if (lastRow < 0)
		lastRow = h->numRows - 1;

	if (lastRow >= int(h->numRows) || firstRow > lastRow || firstRow < 0)
	{
		printf("error: tried to one-line read outside of file: %s\n", h->fileName);
		return NULL;
	}

	if (data == NULL)
		data = (float*)malloc(sizeof(float) * size_t((lastRow - firstRow + 1) * h->numCols));

	if (h->slope <= 0.0 || h->slope >= 1.0e16 || h->offset <= -1.0e16 || h->offset >= 1.0e16)
	{
		// slope and offset seem invalid, so ignore these values
		h->slope = 1.0;
		h->offset = 0.0;
	}

	// offset_to_data must be at least 8 and no bigger than 8+size of the data
	int numBytesOfImage = 4;
	if (h->datatype == 1)
		numBytesOfImage = 2;
	if (h->datatype == 0)
		numBytesOfImage = 1;
	if (h->offset_to_data < 8 || h->offset_to_data > 512 || h->offset_to_data > numBytesOfImage * h->numCols * h->numRows + 8)
	{
		//printf("offset_to_data: %d -> 8\n", h->offset_to_data);
		h->offset_to_data = 8;
	}

	if (h->datatype == 3) // tif data is 32-bit float
	{
		size_t offset = sizeof(float) * h->numCols * firstRow + h->offset_to_data;
		if (fseek(fdes, offset, SEEK_SET) != 0)
			printf("error in fseek\n");
		size_t length = sizeof(float) * h->numCols * (lastRow - firstRow + 1);
		if (!fread(data, length, 1, fdes))
			printf("warning: fread in read_tif_32f may have failed (firstRow=%d, lastRow=%d, numCols=%d, offset_to_data=%d)\n", int(firstRow), int(lastRow), int(h->numCols), int(h->offset_to_data));

		if (h->bigEndian)
		{
			for (int i = 0; i < int(h->numCols * (lastRow - firstRow + 1)); i++)
				data[i] = swapEndian(data[i]);
		}

	}
	if (h->datatype == 1) //tif data is 16-bit unsigned short (can be slope/offset)
	{
		size_t offset = sizeof(uint16) * h->numCols * firstRow + h->offset_to_data;
		//printf("offset = %d\n", int(h->offset_to_data));
		if (fseek(fdes, offset, SEEK_SET) != 0)
			printf("error in fseek\n");
		int vxw = h->numCols * (lastRow - firstRow + 1);
		size_t length = sizeof(uint16) * vxw;
		uint16* data16r = (uint16*)malloc(length);
		if (!fread(data16r, length, 1, fdes))
			printf("warning: fread in read_tif32f_rows, 16-bit may have failed\n");

		if (h->bigEndian)
		{
			for (int i = 0; i < vxw; i++)
				data16r[i] = swapEndian(data16r[i]);
		}

		if (h->slope != 1.0 || h->offset != 0.0)
		{
			for (int i = 0; i < vxw; i++)
				data[i] = (float)(h->slope * float(data16r[i]) + h->offset);
		}
		else
		{
			for (int i = 0; i < vxw; i++)
				data[i] = float(data16r[i]);
		}
		free(data16r);
	}

	if (h->datatype == 0)
	{
		size_t offset = sizeof(uint8) * h->numCols * firstRow + h->offset_to_data;
		if (fseek(fdes, offset, SEEK_SET) != 0)
			printf("error in fseek\n");
		int vxw = h->numCols * (lastRow - firstRow + 1); //read first to last lines inclusive
		size_t length = sizeof(uint8) * vxw;
		uint8* data8r = (uint8*)malloc(length);
		if (!fread(data8r, length, 1, fdes))
			printf("warning: fread in read_tif32f_rows, 8-bit may have failed\n");

		if (h->bigEndian)
		{
			for (int i = 0; i < vxw; i++)
				data8r[i] = swapEndian(data8r[i]);
		}

		if (h->slope != 1.0 || h->offset != 0.0)
		{
			for (int i = 0; i < vxw; i++)
				data[i] = h->slope * float(data8r[i]) + h->offset;
		}
		else
		{
			for (int i = 0; i < vxw; i++)
				data[i] = float(data8r[i]);
		}
		free(data8r);
	}

	fclose(fdes);
	return data;
}

float* load_tif_cols(char* fileName, int firstCol, int lastCol, float* data)
{
	if (fileName == NULL || fileName[0] == 0)
		return NULL;

	ImageHeader H;
	ImageHeader* h = &H;
	sprintf(h->fileName, "%s", fileName);
	FILE* fdes = read_header_leave_open(h);
	if (fdes == NULL)
		return NULL;

	if (lastCol < 0)
		lastCol = h->numCols - 1;

	if (lastCol >= int(h->numCols) || firstCol > lastCol || firstCol < 0)
	{
		printf("error: tried to one-line read outside of file: %s\n", h->fileName);
		return NULL;
	}

	if (data == NULL)
		data = (float*)malloc(sizeof(float) * size_t((lastCol - firstCol + 1) * h->numRows));

	if (h->slope <= 0.0 || h->slope >= 1.0e16 || h->offset <= -1.0e16 || h->offset >= 1.0e16)
	{
		// slope and offset seem invalid, so ignore these values
		h->slope = 1.0;
		h->offset = 0.0;
	}

	// offset_to_data must be at least 8 and no bigger than 8+size of the data
	int numBytesOfImage = 4;
	if (h->datatype == 1)
		numBytesOfImage = 2;
	if (h->datatype == 0)
		numBytesOfImage = 1;
	if (h->offset_to_data < 8 || h->offset_to_data > 512 || h->offset_to_data > numBytesOfImage * h->numCols * h->numRows + 8)
	{
		//printf("offset_to_data: %d -> 8\n", h->offset_to_data);
		h->offset_to_data = 8;
	}

	if (h->datatype == 3)
	{
		size_t offset = sizeof(float) * firstCol + h->offset_to_data;
		size_t length = lastCol - firstCol + 1;
		uint16 skipLength = sizeof(float) * (h->numCols - length);

		if (fseek(fdes, offset, SEEK_SET) != 0)
			printf("error in fseek\n");

		for (int rowNum = 0; rowNum < int(h->numRows); rowNum++)
		{
			if (rowNum > 0)
			{
				if (fseek(fdes, skipLength, SEEK_CUR) != 0)
					printf("error in fseek\n");
			}
			fread(&data[rowNum * length], sizeof(float), length, fdes);
		}

		if (h->bigEndian)
		{
			for (int i = 0; i < int(h->numRows * (lastCol - firstCol + 1)); i++)
				data[i] = swapEndian(data[i]);
		}
	}
	if (h->datatype == 1)
	{
		size_t offset = sizeof(uint16) * firstCol + h->offset_to_data;
		size_t length = lastCol - firstCol + 1;
		uint16 skipLength = sizeof(uint16) * (h->numCols - length);

		if (fseek(fdes, offset, SEEK_SET) != 0)
			printf("error in fseek\n");

		uint16* data16r = (uint16*)malloc(sizeof(uint16) * h->numRows * length);
		for (int rowNum = 0; rowNum < int(h->numRows); rowNum++)
		{
			if (rowNum > 0)
			{
				if (fseek(fdes, skipLength, SEEK_CUR) != 0)
					printf("error in fseek\n");
			}
			fread(&data16r[rowNum * length], sizeof(uint16), length, fdes);
		}
		int vxw = h->numRows * length;
		if (h->bigEndian)
		{
			for (int i = 0; i < vxw; i++)
				data16r[i] = swapEndian(data16r[i]);
		}

		if (h->slope != 1.0 || h->offset != 0.0)
		{
			for (int i = 0; i < vxw; i++)
				data[i] = h->slope * float(data16r[i]) + h->offset;
		}
		else
		{
			for (int i = 0; i < vxw; i++)
				data[i] = float(data16r[i]);
		}
		free(data16r);
	}
	if (h->datatype == 0)
	{
		size_t offset = sizeof(uint8) * firstCol + h->offset_to_data;
		size_t length = lastCol - firstCol + 1;
		uint16 skipLength = sizeof(uint8) * (h->numCols - length);

		if (fseek(fdes, offset, SEEK_SET) != 0)
			printf("error in fseek\n");

		uint8* data8r = (uint8*)malloc(sizeof(uint8) * h->numRows * length);
		for (int rowNum = 0; rowNum < int(h->numRows); rowNum++)
		{
			if (rowNum > 0)
			{
				if (fseek(fdes, skipLength, SEEK_CUR) != 0)
					printf("error in fseek\n");
			}
			fread(&data8r[rowNum * length], sizeof(uint8), length, fdes);
		}
		int vxw = h->numRows * length;
		if (h->bigEndian)
		{
			for (int i = 0; i < vxw; i++)
				data8r[i] = swapEndian(data8r[i]);
		}

		if (h->slope != 1.0 || h->offset != 0.0)
		{
			for (int i = 0; i < vxw; i++)
				data[i] = h->slope * float(data8r[i]) + h->offset;
		}
		else
		{
			for (int i = 0; i < vxw; i++)
				data[i] = float(data8r[i]);
		}
		free(data8r);
	}

	fclose(fdes);
	return data;
}

bool write_tif(char* fileName, float* data, int numRows, int numCols, float pixelHeight, float pixelWidth, int dtype, float wmin, float wmax)
{
	if (fileName == NULL || fileName[0] == 0 || data == NULL || numRows <= 0 || numCols <= 0)
		return false;

	if (wmax <= wmin)
		wmax = wmin + 0.02;

	ImageHeader H;
	ImageHeader* h = &H;
	h->numRows = numRows;
	h->numCols = numCols;
	//h->slope = slope;
	//h->offset = offset;
	h->datatype = dtype;
	h->pixelHeight = pixelHeight;
	h->pixelWidth = pixelWidth;
	sprintf(h->fileName, "%s", fileName);

	char cdata[255];
	//2-byte integers
	uint16 n42 = 42;
	uint16 nde, tag[50], field_type[50];
	uint16 BitsPerSample, BytesPerSample;
	uint16 SampleFormat; // ResolutionUnit;
	//4-byte integers
	uint32 ifd_offset, ft_num_vals[50], val_offset[50];
	uint32 StripOffset;
	uint32 StripByteCounts; //, RowsPerStrip;
	uint32 XRes[2], YRes[2];
	int zero = 0;
	uint32 vxw;
	FILE* wptr;

	//get tif data and defaults set up
	double XResolution = h->pixelWidth;
	double YResolution = h->pixelHeight;
	vxw = h->numCols * h->numRows;
	if (h->datatype == 3)
	{
		SampleFormat = 3; BitsPerSample = 32; BytesPerSample = 4;
		h->slope = 1.0;
		h->offset = 0.0;
	}
	else if (h->datatype == 1)
	{
		SampleFormat = 1; BitsPerSample = 16; BytesPerSample = 2;

		// (x - wmin) / (wmax - wmin) * max_dtype
		h->slope = (wmax - wmin) / 65535.0;
		h->offset = wmin;
	}
	else if (h->datatype == 0)
	{
		SampleFormat = 1; BitsPerSample = 8; BytesPerSample = 1;
		h->slope = (wmax - wmin) / 255.0;
		h->offset = wmin;
	}
	else
	{
		printf("error: unknown write type (%d) in image header\n", h->datatype);
		return false;
	}

	remove(h->fileName);

	//open the file
	//printf("opening %s\n",write_filename);
	if ((wptr = fopen(h->fileName, "wb")) == NULL)
	{
		fprintf(stderr, "error opening %s.\n", h->fileName);
		return false;
	}

	cdata[0] = 'I'; cdata[1] = 'I';
	//top few bytes of the tif file, II and 42
	fwrite(&cdata[0], sizeof(cdata[0]), (size_t)2, wptr);
	fwrite(&n42, sizeof(uint16), (size_t)1, wptr);
	//ifd_offset we'll do this on the fly later.  for now, dump in estimate
	ifd_offset = (uint32)(h->numCols * h->numRows * sizeof(uint16) + 8);
	fwrite(&ifd_offset, sizeof(ifd_offset), (size_t)1, wptr);
	//now write the data.  First of all, we are at byte 8, so tag 273, StripOffset, is 8.
	StripOffset = (uint32)ftell(wptr);

	// now write the image data
	if (h->datatype == 1) // uint16
	{
		uint16* data16 = (uint16*)malloc(vxw * sizeof(uint16));
		for (int i = 0; i < int(vxw); i++)
			data16[i] = uint16(std::max(0.0f, std::min(65535.0f, (data[i] - h->offset) / h->slope)));
		StripByteCounts = vxw * (uint32)sizeof(uint16);
		fwrite(data16, 1, StripByteCounts, wptr);
		free(data16);
	}
	else if (h->datatype == 0) // uint8
	{
		uint8* data8 = (uint8*)malloc(vxw * sizeof(uint8));
		for (int i = 0; i < int(vxw); i++)
			data8[i] = uint8(std::max(0.0f, std::min(255.0f, (data[i] - h->offset) / h->slope)));
		StripByteCounts = vxw * (uint32)sizeof(uint8);
		fwrite(data8, 1, StripByteCounts, wptr);
		free(data8);
	}
	else //if (h->datatype == 3)
	{
		StripByteCounts = vxw * (uint32)sizeof(float);
		fwrite(data, 1, StripByteCounts, wptr);
	}

	//write the values that are pointed to in the directory list
	//tag270 type 2 vals 44 val offset?
	val_offset[5] = (uint32)ftell(wptr); tag[5] = 270;
	sprintf(cdata, "slope = %13.6E\13offset = %13.6E", h->slope, h->offset);
	fwrite(&cdata[0], sizeof(cdata[0]), (size_t)(strlen(cdata)), wptr);

	// Xres then Yres values
	val_offset[10] = (uint32)ftell(wptr); tag[10] = 282;
	XRes[0] = 1000000000;  //FIXME:  These are 32-bit unsigned int; we can go out to 4 billion if we want.
	XRes[1] = (uint32)((0.1 * XResolution * (float)XRes[0]));//0.1 to convert mm to cm
	fwrite(&XRes[0], sizeof(uint32), 2, wptr);
	val_offset[11] = (uint32)ftell(wptr); tag[11] = 283;
	YRes[0] = 1000000000;
	YRes[1] = (uint32)((0.1 * YResolution * (float)YRes[0]));//0.1 to convert mm to cm
	fwrite(&YRes[0], sizeof(uint32), 2, wptr);
	
	//Now make sure ifd_offset is aligned to a word.  (Is it 2-byte or 4-byte?)
	while (ftell(wptr) % 4) fwrite(&zero, sizeof(char), 1, wptr);
	//We need to write this as the ifd_offset in the beginning of the file
	ifd_offset = (uint32)ftell(wptr);
	
	//nde
	nde = 14; fwrite(&nde, sizeof(uint16), (size_t)1, wptr); // byte count; # of directory entries
	
	//0 width
	tag[0] = 256; fwrite(&tag[0], sizeof(uint16), (size_t)1, wptr);
	field_type[0] = 4; fwrite(&field_type[0], sizeof(uint16), (size_t)1, wptr);
	ft_num_vals[0] = 1; fwrite(&ft_num_vals[0], sizeof(uint32), (size_t)1, wptr);
	val_offset[0] = h->numCols; fwrite(&val_offset[0], sizeof(uint32), (size_t)1, wptr);

	//1 height
	tag[0] = 257; fwrite(&tag[0], sizeof(uint16), (size_t)1, wptr);
	field_type[0] = 4; fwrite(&field_type[0], sizeof(uint16), (size_t)1, wptr);
	ft_num_vals[0] = 1; fwrite(&ft_num_vals[0], sizeof(uint32), (size_t)1, wptr);
	val_offset[0] = h->numRows; fwrite(&val_offset[0], sizeof(uint32), (size_t)1, wptr);

	//Is it 32, 16, or 8-bit per sample?
	tag[0] = 258; fwrite(&tag[0], sizeof(uint16), (size_t)1, wptr);
	field_type[0] = 3; fwrite(&field_type[0], sizeof(uint16), (size_t)1, wptr);
	ft_num_vals[0] = 1; fwrite(&ft_num_vals[0], sizeof(uint32), (size_t)1, wptr);
	val_offset[0] = BitsPerSample; fwrite(&val_offset[0], sizeof(uint32), (size_t)1, wptr);

	//3 compression 1
	tag[0] = 259; fwrite(&tag[0], sizeof(uint16), (size_t)1, wptr);
	field_type[0] = 3; fwrite(&field_type[0], sizeof(uint16), (size_t)1, wptr);
	ft_num_vals[0] = 1; fwrite(&ft_num_vals[0], sizeof(uint32), (size_t)1, wptr);
	val_offset[0] = 1; fwrite(&val_offset[0], sizeof(uint32), (size_t)1, wptr);

	//4 photo black is zero 1
	tag[0] = 262; fwrite(&tag[0], sizeof(uint16), (size_t)1, wptr);
	field_type[0] = 3; fwrite(&field_type[0], sizeof(uint16), (size_t)1, wptr);
	ft_num_vals[0] = 1; fwrite(&ft_num_vals[0], sizeof(uint32), (size_t)1, wptr);
	val_offset[0] = 1; fwrite(&val_offset[0], sizeof(uint32), (size_t)1, wptr);

	//5 270 description this is slope and offset
	//tag[5] and val_offset[5] defined above
	fwrite(&tag[5], sizeof(uint16), (size_t)1, wptr);
	field_type[5] = 2; fwrite(&field_type[5], sizeof(uint16), (size_t)1, wptr);
	ft_num_vals[5] = 44; fwrite(&ft_num_vals[5], sizeof(uint32), (size_t)1, wptr);
	fwrite(&val_offset[5], sizeof(uint32), (size_t)1, wptr);

	//6 273 StripOffset
	tag[0] = 273; fwrite(&tag[0], sizeof(uint16), (size_t)1, wptr);
	field_type[0] = 4; fwrite(&field_type[0], sizeof(uint16), (size_t)1, wptr);
	ft_num_vals[0] = 1; fwrite(&ft_num_vals[0], sizeof(uint32), (size_t)1, wptr);
	val_offset[0] = 8; fwrite(&val_offset[0], sizeof(uint32), (size_t)1, wptr);

	//7 277 SamplesPerPixel
	tag[0] = 277; fwrite(&tag[0], sizeof(uint16), (size_t)1, wptr);
	field_type[0] = 3; fwrite(&field_type[0], sizeof(uint16), (size_t)1, wptr);
	ft_num_vals[0] = 1; fwrite(&ft_num_vals[0], sizeof(uint32), (size_t)1, wptr);
	val_offset[0] = 1; fwrite(&val_offset[0], sizeof(uint32), (size_t)1, wptr);

	//8 278 RowsPerStrip
	tag[0] = 278; fwrite(&tag[0], sizeof(uint16), (size_t)1, wptr);
	field_type[0] = 4; fwrite(&field_type[0], sizeof(uint16), (size_t)1, wptr);
	ft_num_vals[0] = 1; fwrite(&ft_num_vals[0], sizeof(uint32), (size_t)1, wptr);
	//The length, number of rows, is equal to the height of the image
	val_offset[0] = h->numRows; fwrite(&val_offset[0], sizeof(uint32), (size_t)1, wptr);

	//9 279 StripByteCounts
	tag[0] = 279; fwrite(&tag[0], sizeof(uint16), (size_t)1, wptr);
	field_type[0] = 4; fwrite(&field_type[0], sizeof(uint16), (size_t)1, wptr);
	ft_num_vals[0] = 1; fwrite(&ft_num_vals[0], sizeof(uint32), (size_t)1, wptr);
	val_offset[0] = StripByteCounts; fwrite(&val_offset[0], sizeof(uint32), (size_t)1, wptr);

	//10 282 XResolution
	fwrite(&tag[10], sizeof(uint16), (size_t)1, wptr);
	field_type[10] = 5; fwrite(&field_type[10], sizeof(uint16), (size_t)1, wptr);
	ft_num_vals[10] = 1; fwrite(&ft_num_vals[10], sizeof(uint32), (size_t)1, wptr);
	fwrite(&val_offset[10], sizeof(uint32), (size_t)1, wptr);

	//11 283 YResolution
	fwrite(&tag[11], sizeof(uint16), (size_t)1, wptr);
	field_type[11] = 5; fwrite(&field_type[11], sizeof(uint16), (size_t)1, wptr);
	ft_num_vals[11] = 1; fwrite(&ft_num_vals[11], sizeof(uint32), (size_t)1, wptr);
	fwrite(&val_offset[11], sizeof(uint32), (size_t)1, wptr);

	//12 296 ResUnit: go with cm
	tag[0] = 296; fwrite(&tag[0], sizeof(uint16), (size_t)1, wptr);
	field_type[0] = 3; fwrite(&field_type[0], sizeof(uint16), (size_t)1, wptr);
	ft_num_vals[0] = 1; fwrite(&ft_num_vals[0], sizeof(uint32), (size_t)1, wptr);
	val_offset[0] = 3; fwrite(&val_offset[0], sizeof(uint32), (size_t)1, wptr);

	//13 339 SampleFormat: 1 for uint, 2 for int, 3 for float, and 4 for undef
	tag[0] = 339; fwrite(&tag[0], sizeof(uint16), (size_t)1, wptr);
	field_type[0] = 3; fwrite(&field_type[0], sizeof(uint16), (size_t)1, wptr);
	ft_num_vals[0] = 1; fwrite(&ft_num_vals[0], sizeof(uint32), (size_t)1, wptr);
	val_offset[0] = SampleFormat; fwrite(&val_offset[0], sizeof(uint32), (size_t)1, wptr);

	//write out one more zero at the end of the ifd.
	val_offset[0] = 0;
	fwrite(&val_offset[0], sizeof(uint32), (size_t)1, wptr);

	//go back and write the correct ifd_offset
	fseek(wptr, 4, SEEK_SET);
	fwrite(&ifd_offset, sizeof(ifd_offset), (size_t)1, wptr);

	//fseek(wptr, 0, SEEK_SET);

	fclose(wptr);
	return true;
}
