////////////////////////////////////////////////////////////////////////////////
// Copyright 2016-2025 Kyle Champley
//
// c++ module for the Maximally Flat Filter (MFF)
////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include "maximally_flat_filter.h"
#include "leap_defines.h"

float mff_filter(float x, int order)
{
    if (order == 0)
    {
        if (fabs(x) >= 1.0)
            return 0.0;
        else
        {
            float retVal = cos(0.5 * PI * x);
            return retVal * retVal;
        }
    }

    order = std::max(2, std::min(16, order));
    const float* coeff = NULL;
    switch (order)
    {
        case 2:
        case 3:
            coeff = mff_coeff2;
            break;
        case 4:
        case 5:
            coeff = mff_coeff4;
            break;
        case 6:
        case 7:
            coeff = mff_coeff6;
            break;
        case 8:
        case 9:
            coeff = mff_coeff8;
            break;
        case 10:
        case 11:
            coeff = mff_coeff10;
            break;
        case 12:
        case 13:
            coeff = mff_coeff12;
            break;
        case 14:
        case 15:
            coeff = mff_coeff14;
            break;
        case 16:
        case 17:
            coeff = mff_coeff16;
            break;
        default:
            coeff = mff_coeff2;
    }

    order = order - (order%2);
    float retVal = 0.0;
    for (int n = 0; n < order; n++)
    {
        if (coeff[n] != 0.0)
            retVal += coeff[n]*(sinc(2.0*x-n) + sinc(2.0*x+n));
    }
    return retVal;
}

float sinc(float x)
{
    if (fabs(x) < 1.0e-4)
    {
        float pix = PI * x;
        float pix2 = pix * pix;
        return 1.0 - pix2 / 6.0 + (pix2 * pix2) / 120.0;
    }
    else
        return sin(PI * x) / (PI * x);
}
