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
			if (strcmpi(cslope, "slope") != 0)
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

float* read_tif(char* fileName, float* data)
{
	return read_tif_rows(fileName, 0, -1, data);
}

float* read_tif_rows(char* fileName, int firstRow, int lastRow, float* data)
{
	if (fileName == NULL || fileName[0] == 0)
		return NULL;

	ImageHeader H;
	ImageHeader* h = &H;
	sprintf(h->fileName, "%s", fileName);
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

	size_t offset;
	size_t length;
	uint16* data16r;
	uint8* data8r;
	int vxw;

	if (h->datatype == 3) // tif data is 32-bit float
	{
		offset = sizeof(float) * h->numCols * firstRow + h->offset_to_data;
		if (fseek(fdes, offset, SEEK_SET) != 0)
			printf("error in fseek\n");
		length = sizeof(float) * h->numCols * (lastRow - firstRow + 1);
		if (!fread(data, length, 1, fdes))
			printf("warning: fread in read_tif_32f may have failed (numCols=%d, offset_to_data=%d)\n", int(h->numCols), int(h->offset_to_data));

		if (h->bigEndian)
		{
			for (int i = 0; i < int(h->numCols * (lastRow - firstRow + 1)); i++)
				data[i] = swapEndian(data[i]);
		}

	}
	if (h->datatype == 1) //tif data is 16-bit unsigned short (can be slope/offset)
	{
		offset = sizeof(uint16) * h->numCols * firstRow + h->offset_to_data;
		//printf("offset = %d\n", int(h->offset_to_data));
		if (fseek(fdes, offset, SEEK_SET) != 0)
			printf("error in fseek\n");
		vxw = h->numCols * (lastRow - firstRow + 1);
		length = sizeof(uint16) * vxw;
		data16r = (uint16*)malloc(length);
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
		offset = sizeof(uint8) * h->numCols * firstRow + h->offset_to_data;
		if (fseek(fdes, offset, SEEK_SET) != 0)
			printf("error in fseek\n");
		vxw = h->numCols * (lastRow - firstRow + 1); //read first to last lines inclusive
		length = sizeof(uint8) * vxw;
		data8r = (uint8*)malloc(length);
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

float* read_tif_cols(char* fileName, int firstCol, int lastCol, float* data)
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

	size_t offset;
	size_t length;
	uint16* data16r;
	uint8* data8r;
	int vxw;

	if (h->datatype == 3)
	{
		offset = sizeof(float) * firstCol + h->offset_to_data;
		length = lastCol - firstCol + 1;
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
		offset = sizeof(uint16) * firstCol + h->offset_to_data;
		length = lastCol - firstCol + 1;
		uint16 skipLength = sizeof(uint16) * (h->numCols - length);

		if (fseek(fdes, offset, SEEK_SET) != 0)
			printf("error in fseek\n");

		data16r = (uint16*)malloc(sizeof(uint16) * h->numRows * length);
		for (int rowNum = 0; rowNum < int(h->numRows); rowNum++)
		{
			if (rowNum > 0)
			{
				if (fseek(fdes, skipLength, SEEK_CUR) != 0)
					printf("error in fseek\n");
			}
			fread(&data16r[rowNum * length], sizeof(uint16), length, fdes);
		}
		vxw = h->numRows * length;
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
		offset = sizeof(uint8) * firstCol + h->offset_to_data;
		length = lastCol - firstCol + 1;
		uint16 skipLength = sizeof(uint8) * (h->numCols - length);

		if (fseek(fdes, offset, SEEK_SET) != 0)
			printf("error in fseek\n");

		data8r = (uint8*)malloc(sizeof(uint8) * h->numRows * length);
		for (int rowNum = 0; rowNum < int(h->numRows); rowNum++)
		{
			if (rowNum > 0)
			{
				if (fseek(fdes, skipLength, SEEK_CUR) != 0)
					printf("error in fseek\n");
			}
			fread(&data8r[rowNum * length], sizeof(uint8), length, fdes);
		}
		vxw = h->numRows * length;
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

bool write_tif(char* fileName, float* data, int numRows, int numCols, int dtype, float slope, float offset)
{
	// see: ::write_tif(float *data, IMG_HEADER *h, char write_filename[])
	return false;
}
