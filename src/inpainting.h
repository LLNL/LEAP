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

#include <stdlib.h>
#include <vector>

/*
 * This class implements 2D Telea inpainting which is essentially equivalent to:
 * import cv2 as cv
 * cv.inpaint(img, mask, window_size, cv.INPAINT_TELEA)
 * Except these routines work on floating point images (the cv2 method does not)
 * and uses OpenMP parallelization to inpaint a series of images (projections or volume slices)
 * in parallel.  This code is just a simple C++ translation of the pseudo-code in the reference page:
 * Telea, Alexandru. "An image inpainting technique based on the fast marching method." Journal of graphics tools 9, no. 1 (2004): 23-34.
 */

// This class just tracks image coordinates of a pixel
class xyCoord
{
public:
	xyCoord();
	xyCoord(int, int);
	~xyCoord();
	int y;
	int x;
};

// This class is just for convenience to track 2D floating point data (images)
class image
{
public:
	image();
	image(int, int);
	~image();

	/**
	 * \fn          clearAll
	 * \brief       sets numRows and numCols to zero, frees data is ownsData == true
	 */
	void clearAll();

	/**
	 * \fn          malloc
	 * \brief       allocates image data and sets ownsData = true
	 * \return      true if operation is successful, false otherwise (if numRows == 0 or numCols == 0)
	 */
	bool malloc();

	/**
	 * \fn          free
	 * \brief       if ownsData is true, then deletes data and sets it to NULL, otherwise just sets data to NULL
	 */
	void free();

	/**
	 * \fn          copy
	 * \brief       makes a deep copy of this class
	 * \param[in]   setToZero: if true, new image will be all zeros, otherwise values will be copied to the new image object
	 * \return      pointer to the new image object
	 */
	image* copy(bool setToZero = false);

	/**
	 * \fn          equal
	 * \brief       sets the values of data to the values of the input argument
	 * \param[in]   lhs: point to image object for which to copy the values
	 * \return      this
	 */
	image* equal(image* lhs);

	/**
	 * \fn          get
	 * \brief       returns data[iRow*numCols+iCol], if iRow, iCol are out of bounds returns 0.0
	 * \param[in]   iRow: row index
	 * \param[in]	iCol: column index
	 * \return      data[iRow*numCols+iCol]
	 */
	float get(int iRow, int iCol);

	/**
	 * \fn          set
	 * \brief       data[iRow*numCols+iCol] = val
	 * \param[in]   iRow: image row index
	 * \param[in]	iCol: image column index
	 * \param[in]	val: value to set
	 * \return      true if operation is succcessful, false otherwise
	 */
	bool set(int iRow, int iCol, float val);

	/**
	 * \fn          set
	 * \brief       sets the class member variables (with ownsData = false)
	 * \param[in]	data_in: 2D image data, for which we do: data = data_in and ownsData = false
	 * \param[in]   M: number of rows
	 * \param[in]	N: number of columns
	 * \return      true if operation is succcessful, false otherwise
	 */
	bool set(float* data_in, int M, int N);

	float* data; // pointer to 2D array
	int numRows; // number of rows
	int numCols; // number of columns
private:
	bool ownsData; // if true, this class will be responsible to allocating and freeing, otherwise not
};

// This class stores a stack of image coordinates for processing
class imageHeap
{
public:
	imageHeap();
	~imageHeap();

	/**
	 * \fn          init
	 * \brief       does coords.clear(); I = input;
	 * \param[in]   input: image to inpaint
	 * \return      true if operation is successful, false otherwise
	 */
	bool init(image* input);

	/**
	 * \fn          init
	 * \return      coords.empty()
	 */
	bool isEmpty();

	/**
	 * \fn          pop
	 * \brief		pops the coord stack and returns the popped value
	 * \return      coords.back()
	 */
	xyCoord pop();

	/**
	 * \fn          push
	 * \brief		pushed new image coords onto stack
	 * \return      true is successful, false otherwise
	 */
	bool push(xyCoord);

	/**
	 * \fn          printAll
	 * \brief		prints all coordinates on the stack (for debugging purposes)
	 * \return      true is successful, false otherwise
	 */
	bool printAll();

	/**
	 * \fn          compare_larger
	 * \brief		returns whether the image value at a is bigger than the value at b
	 * \param[in]	a: image coordinate
	 * \param[in]	b: image coordinate
	 * \return      (I->get(a.y, a.x) > I->get(b.y, b.x))
	 */
	bool compare_larger(const xyCoord& a, const xyCoord& b);

	// Stack of image coords to process for inpainting
	std::vector<xyCoord> coords;

	// pointer to image (not owned by this class)
	image* I;
};

// BAND: the pixel belongs to the narrow band.  Its T value undergoes update
// KNOWN: the pixel is outside ∂Ω, in the known image area. Its T and I values are known.
// INSIDE: the pixel is inside ∂Ω, in the region to inpaint. Its T and I values are not yet known.
enum pixelType_list { TOOFAR = 0, KNOWN, BAND, INSIDE };

// This class performs 2D Telea inpainting
class inpaint2D
{
public:
	inpaint2D();
	~inpaint2D();

	/**
	 * \fn          executre
	 * \brief       performs 2D Telea inpainting
	 * \param[in]   theInput: 2D data to inpaint
	 * \param[in]	theMask: the mask which specifies which pixels to inpaint (positive values mark pixels to be inpainted)
	 * \return      true if operation is successful, false otherwise
	 */
	bool execute(image* theInput, image* theMask);

	/**
	 * \fn          setWindowSize
	 * \brief       Sets the window size used in the algorithm which is the main input parameter
	 * \param[in]   theWindowSize: the size of the window in number of pixels
	 * \return      true if operation is successful, false otherwise
	 */
	bool setWindowSize(int theWindowSize);

	/**
	 * \fn          setGradAtten
	 * \brief       Sets a parameter which helps mitigate artifacts caused by large gradients in the image
	 * \param[in]   a: the value to be multiplied to the gradient
	 * \return      true if operation is successful, false otherwise
	 */
	bool setGradAtten(double a);

private:

	/**
	 * \fn          clearAll
	 * \brief       frees all memory of member variables
	 * \return      true if operation is successful, false otherwise
	 */
	bool clearAll();

	/**
	 * \fn          init
	 * \brief       Initializes temporary variables for the algorithm
	 * \param[in]   theInput: 2D data to inpaint
	 * \param[in]	theMask: the mask which specifies which pixels to inpaint (positive values mark pixels to be inpainted)
	 * \return      true if operation is successful, false otherwise
	 */
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

	/**
	 * \fn          isBAND
	 * \brief       returns whether the given pixel is on the boundary of the inpainted region
	 * \param[in]   J: pointer to the image which labels pixels of pixelType_list type
	 * \param[in]   i: image row index
	 * \param[in]   j: image column index
	 * \return      true if pixel is on the boundary of the inpainted region, false otherwise
	 */
	bool isBAND(image* J, int i, int j);

	/**
	 * \fn          get_gI
	 * \brief       sets the gradient images gI_x and gI_y at the given pixel indices
	 * \param[in]   i: image row index
	 * \param[in]   j: image column index
	 */
	void get_gI(int i, int j);

	/**
	 * \fn       	init_gI   
	 * \brief       initializes the gradient images, gI_x and gI_y
	 */
	void init_gI();

	/**
	 * \fn          get_gT
	 * \brief       sets the gradient images gT_x and gT_y at the given pixel indices
	 * \param[in]   i: image row index
	 * \param[in]   j: image column index
	 */
	void get_gT(int i, int j);

	/**
	 * \fn          init_gT
	 * \brief       initializes the gradient images, gT_x and gT_y
	 */
	void init_gT();

	/**
	 * \fn          renew_gI_gT
	 * \brief       recalculates all four gradient images
	 * \param[in]   i: image row index
	 * \param[in]   j: image column index
	 */
	void renew_gI_gT(int i, int j);

	/**
	 * \fn          smoothT
	 * \brief       applies a low-pass filter to the image T
	 */
	void smoothT();

	/**
	 * \fn          smoothI
	 * \brief       applies a low-pass filter to the image I
	 */
	void smoothI();

	/**
	 * \fn          inpaint
	 * \brief       inpaints the given pixel
	 * \param[in]   i: image row index
	 * \param[in]   j: image column index
	 */
	void inpaint(int i, int j);

	/**
	 * \fn          inpaint_other
	 * \brief       inpaints the given pixel (experimental alternature method)
	 * \param[in]   i: image row index
	 * \param[in]   j: image column index
	 */
	void inpaint_other(int i, int j);

	/**
	 * \fn          insideFMM
	 * \brief       runs the fast marching method on regions inside the inpainted region
	 */
	void insideFMM();

	/**
	 * \fn          insideFMM
	 * \brief       runs the fast marching method on regions outside the inpainted region
	 */
	void outsideFMM();

	/**
	 * \fn          solve
	 * \brief       performs the solution to the problem given a pixel and its neighbor
	 * \param[in]   i1: image row index
	 * \param[in]   j1: image column index
	 * \param[in]   i2: image row index
	 * \param[in]   j2: image column index
	 * \param[in]   TYPE1: label for first pixel
	 * \param[in]   TYPE2: label for second pixel
	 * \return		returns the solution
	 */
	double solve(int i1, int j1, int i2, int j2, int TYPE1, int TYPE2);

	/**
	 * \fn          min4
	 * \return      returns the minimum value of the four given inputs
	 */
	double min4(double, double, double, double);

	int EPSILON_PIXEL; // window size
	int T_MAX; // T_MAX = 2 * EPSILON_PIXEL
	bool useGradients; // debug parameter, usually set to true
	double strangeFactor; // debug parameter, strangeFactor = max(0.0, min(1.0, setGradAtten(a)))
};


// This function just kicks off a series of 2D inpainting tasks
// one for each slice and uses OpenMP parallelization over each slice
bool inpaint3D(float* I, int N_1, int N_2, int N_3);

#endif
