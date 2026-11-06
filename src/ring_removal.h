////////////////////////////////////////////////////////////////////////////////
// Copyright 2016-2025 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// c++ header for ring removal algorithms
////////////////////////////////////////////////////////////////////////////////
#ifndef __RING_REMOVAL_H
#define __RING_REMOVAL_H

#ifdef WIN32
#pragma once
#endif

#include <stdlib.h>

class ringRemoval
{
public:
    ringRemoval();
    ~ringRemoval();

    /**
	 * \fn          execute
	 * \brief       Performs minimization of a weighted combination of an L2
     *              loss function (of orginal data and do-ringed data) and a TV functional
     *              with differences over the third dimension only.
     *              A preconditioner is applied to every iteration which takes the mean
     *              of the TV gradient over all angles which ensures the correction is
     *              the same for every view.
     * \param[in]   g:
     * \param[in]   N_1: number of elements in the 1st dimension (numAngles)
     * \param[in]   N_2: number of elements in the 2nd dimension (numRows)
     * \param[in]   N_3: number of elements in the 3rd dimension (numCols)
     * \param[in]   delta: Huber loss function parameter
     * \param[in]   beta: strength of TV functional
     * \param[in]   numIter: number of iterations of gradient descent
     * \param[in]   maxChange: the maximum value allowed for this correction
     * \param[in]   angle_downsampling_factor: parameter affecting algorithm speed vs accuracy
	 * \return      true if operation was sucessful, false otherwise
	 */
    bool execute(float* g, int N_1, int N_2, int N_3, float delta = 0.01, float beta = 1.0e3, int numIter = 30, float maxChange = 0.05, int angle_downsampling_factor = 1);
private:

    /**
	 * \fn          h1
	 * \brief       Calculates the first derivative of the Huber loss function
     * \param[in]   t: argument to Huber loss function derivative
	 * \return      the first derivative of the Huber loss function of the given input
	 */
    float h1(float t);

    /**
	 * \fn          h1
	 * \brief       Calculates the quadratic surrogate of second derivative of the Huber loss function
     *              which is equal to h1(t)/t
     * \param[in]   t: argument to Huber loss function second derivative quadratic surrogate
	 * \return      the second derivative quadratic surrogate of the Huber loss function of the given input
	 */
    float h2(float t);

    float delta; // Huber loss function parameter
    float beta; // strength of TV functional
};

#endif
