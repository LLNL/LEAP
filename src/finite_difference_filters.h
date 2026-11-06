////////////////////////////////////////////////////////////////////////////////
// Copyright 2016-2025 Kyle Champley
//
// c++ header for finite difference filters
////////////////////////////////////////////////////////////////////////////////
#ifndef __FINITE_DIFFERENCE_FILTERS_H
#define __FINITE_DIFFERENCE_FILTERS_H

#ifdef WIN32
#pragma once
#endif

#include <stdlib.h>

/**
 * \fn          first_order_finite_difference_filter
 * \brief       returns first order finite difference filter
 * \param[in]   L: length of the filter (set by this function)
 * \param[in]   order: the accuracy of the finite difference filter
 * \param[in]   shift: -1 for backward difference, 0 for central difference, 1 for forward difference
 * \return      pointer to the filter (calling function needs to free this when done)
 */
float* first_order_finite_difference_filter(int& L, int order = 2, int shift = 1);

/**
 * \fn          second_order_finite_difference_filter
 * \brief       returns second order finite difference filter
 * \param[in]   L: length of the filter (set by this function)
 * \param[in]   order: the accuracy of the finite difference filter
 * \return      pointer to the filter (calling function needs to free this when done)
 */
float* second_order_finite_difference_filter(int&L, int order = 2);

#endif
