////////////////////////////////////////////////////////////////////////////////
// Copyright 2016-2025 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// c++ header for inpainting
////////////////////////////////////////////////////////////////////////////////
#ifndef __INPAINTING_H
#define __INPAINTING_H

#ifdef WIN32
#pragma once
#endif

//#include "parameters.h"
#include <stdlib.h>
#include <vector>

class xyCoord
{
public:
	xyCoord();
	xyCoord(int, int);
	~xyCoord();
	int y;
	int x;
};

class image
{
public:
	image();
	image(int, int);
	~image();

	void clearAll();
	bool malloc();
	void free();
	image* copy(bool setToZero = false);
	image* equal(image* lhs);

	float get(int, int);
	bool set(int, int, float);
	bool set(float* data_in, int M, int N);

	float* data;
	int numRows;
	int numCols;
private:
	bool ownsData;
};

class imageHeap
{
public:
	imageHeap();
	~imageHeap();
	bool init(image*);

	bool isEmpty();
	xyCoord pop();
	bool push(xyCoord);

	bool printAll();

	bool compare_larger(const xyCoord& a, const xyCoord& b);

	std::vector<xyCoord> coords;
	image* I;
};

// BAND: the pixel belongs to the narrow band.  Its T value undergoes update
// KNOWN: the pixel is outside ∂Ω, in the known image area. Its T and I values are known.
// INSIDE: the pixel is inside ∂Ω, in the region to inpaint. Its T and I values are not yet known.
enum pixelType_list { TOOFAR = 0, KNOWN, BAND, INSIDE };

//*
class inpaint2D
{
public:
	inpaint2D();
	~inpaint2D();

	bool execute(image*, image*);
	bool setWindowSize(int);
	bool setGradAtten(double);

private:
	bool clearAll();
	bool init(image* theInput, image* theMask);

	image* I; // image to be corrected
	image* f; // a Flag that records labels of TOOFAR, KNOWN, BAND, or INSIDE

	// Temporary images
	image* T; // solution to the Eikonal equation
	image* gI_x; // image gradient in x direction
	image* gI_y; // image gradient in y direction
	image* gT_x; // gradient of T in x direction
	image* gT_y; // gradient of T in y direction

	imageHeap heap;

	bool isBAND(image* J, int i, int j);

	void get_gI(int i, int j);
	void init_gI();
	void get_gT(int i, int j);
	void init_gT();
	void renew_gI_gT(int i, int j);
	void smoothT();
	void smoothI();

	void inpaint(int i, int j);
	void inpaint_other(int i, int j);
	void insideFMM();
	void outsideFMM();
	double solve(int i1, int j1, int i2, int j2, int TYPE1, int TYPE2);
	double min4(double, double, double, double);

	int EPSILON_PIXEL;
	int T_MAX;
	bool useGradients;
	double strangeFactor;
};
//*/

bool inpaint3D(float* I, int N_1, int N_2, int N_3);

#endif
