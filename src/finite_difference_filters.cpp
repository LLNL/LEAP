////////////////////////////////////////////////////////////////////////////////
// Copyright 2016-2025 Kyle Champley
//
// c++ module for finite difference filters
////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include "maximally_flat_filter.h"
#include "leap_defines.h"

float* first_order_finite_difference_filter(int& L, int order, int shift)
{
    order = std::max(0, std::min(order, 16));
    
    float* retVal = NULL;
    if (shift == 0)
	{
		switch (order)
		{
			case 0:
				//break;
			case 1:
				//break;
			case 2:
				L = 3;
				retVal = (float*) calloc(size_t(L), sizeof(float));
				retVal[2] = 0.5;
				retVal[1] = 0.0;
				retVal[0] = -0.5;
				break;
			case 3:
			case 4:
                L = 5;
				retVal = (float*) calloc(size_t(L), sizeof(float));
				retVal[4] = -1.0/12.0;
				retVal[3] = 2.0/3.0;
				retVal[2] = 0.0;
				retVal[1] = -2.0/3.0;
				retVal[0] = 1.0/12.0;
				break;
			case 5:
			case 6:
                L = 7;
				retVal = (float*) calloc(size_t(L), sizeof(float));
				retVal[6] = 1.0/60.0;
				retVal[5] = -3.0/20.0;
				retVal[4] = 3.0/4.0;
				retVal[3] = 0.0;
				retVal[2] = -3.0/4.0;
				retVal[1] = 3.0/20.0;
				retVal[0] = -1.0/60.0;
				break;
			case 7:
			case 8:
                L = 9;
				retVal = (float*) calloc(size_t(L), sizeof(float));
				retVal[8] = -1.0/280.0;
				retVal[7] = 4.0/105.0;
				retVal[6] = -1.0/5.0;
				retVal[5] = 4.0/5.0;
				retVal[4] = 0.0;
				retVal[3] = -4.0/5.0;
				retVal[2] = 1.0/5.0;
				retVal[1] = -4.0/105.0;
				retVal[0] = 1.0/280.0;
				break;
			case 9:
			case 10:
                L = 11;
				retVal = (float*) calloc(size_t(L), sizeof(float));
				//[-1/1260 5/504 -5/84 5/21 -5/6 0 5/6 -5/21 5/84 -5/504 1/1260]
				retVal[10] = 1.0/1260.0;
				retVal[9] = -5.0/504.0;
				retVal[8] = 5.0/84.0;
				retVal[7] = -5.0/21.0;
				retVal[6] = 5.0/6.0;
				retVal[5] = 0.0;
				retVal[4] = -5.0/6.0;
				retVal[3] = 5.0/21.0;
				retVal[2] = -5.0/84.0;
				retVal[1] = 5.0/504.0;
				retVal[0] = -1.0/1260.0;
				break;
			default:
                L = 11;
				retVal = (float*) calloc(size_t(L), sizeof(float));
				retVal[10] = 1.0/1260.0;
				retVal[9] = -5.0/504.0;
				retVal[8] = 5.0/84;
				retVal[7] = -5/21.0;
				retVal[6] = 5.0/6.0;
				retVal[5] = 0.0;
				retVal[4] = -5.0/6.0;
				retVal[3] = 5.0/21.0;
				retVal[2] = -5.0/84.0;
				retVal[1] = 5.0/504.0;
				retVal[0] = -1.0/1260.0;
		}
	}
	else
	{
		shift = 0;
		//if (shift < 0)
		//	shift = 1;
		if (order <= 0)
		{
			L = 4;
			retVal = (float*) calloc(size_t(L), sizeof(float));

			retVal[3+shift] = 0.25;
			retVal[2+shift] = 0.25;
			retVal[1+shift] = -0.25;
			retVal[0+shift] = -0.25;
		}
		else if (order <= 2)
		{
			L = 2;
			retVal = (float*) calloc(size_t(L), sizeof(float));
			retVal[1+shift] = 1.0;
			retVal[0+shift] = -1.0;
		}
		else if (order <= 4)
		{
			L = 4;
			retVal = (float*) calloc(size_t(L), sizeof(float));
			retVal[3+shift] = -1.0/24.0;
			retVal[2+shift] = 9.0/8.0;
			retVal[1+shift] = -9.0/8.0;
			retVal[0+shift] = 1.0/24.0;
		}
		else if (order <= 6)
		{
			L = 6;
			retVal = (float*) calloc(size_t(L), sizeof(float));
			retVal[5+shift] = 3.0/640.0;
			retVal[4+shift] = -25.0/384.0;
			retVal[3+shift] = 75.0/64.0;
			retVal[2+shift] = -75.0/64.0;
			retVal[1+shift] = 25.0/384.0;
			retVal[0+shift] = -3.0/640.0;
		}
		else if (order <= 8)
		{
			L = 8;
			retVal = (float*) calloc(size_t(L), sizeof(float));
			retVal[7+shift] = -5.0/7168.0;
			retVal[6+shift] = 49.0/5120.0;
			retVal[5+shift] = -245.0/3072.0;
			retVal[4+shift] = 1225.0/1024.0;
			retVal[3+shift] = -1225.0/1024.0;
			retVal[2+shift] = 245.0/3072.0;
			retVal[1+shift] = -49.0/5120.0;
			retVal[0+shift] = 5.0/7168.0;
		}
		else
		{
            L = 10;
            retVal = (float*) calloc(size_t(L), sizeof(float));
            retVal[9+shift] = 35.0/294912.0;
            retVal[8+shift] = -405.0/229376.0;
            retVal[7+shift] = 567.0/40960.0;
            retVal[6+shift] = -735.0/8192.0;
            retVal[5+shift] = 19845.0/16384.0;
            retVal[4+shift] = -19845.0/16384.0;
            retVal[3+shift] = 735.0/8192.0;
            retVal[2+shift] = -567.0/40960.0;
            retVal[1+shift] = 405.0/229376.0;
            retVal[0+shift] = -35.0/294912.0;
		}
	}

	return retVal;
}

float* second_order_finite_difference_filter(int&L, int order)
{
    order = std::max(0, std::min(order, 16));
    //order += (order % 2);
    float* retVal = NULL;
	switch (order)
	{
		case 0:
		case 1:
		case 2:
			L = 3;
			retVal = (float*) calloc(size_t(L), sizeof(float));
			retVal[0] = 1.0;
			retVal[1] = -2.0;
			retVal[2] = 1.0;
			break;
		case 3:
		case 4:
			L = 5;
			retVal = (float*) calloc(size_t(L), sizeof(float));
			retVal[0] = -1.0/12.0;
			retVal[1] = 4.0/3.0;
			retVal[2] = -5.0/2.0;
			retVal[3] = 4.0/3.0;
			retVal[4] = -1.0/12.0;
			break;
		case 5:
		case 6:
			L = 7;
			retVal = (float*) calloc(size_t(L), sizeof(float));
			retVal[0] = 1.0/90.0;
			retVal[1] = -3.0/20.0;
			retVal[2] = 3.0/2.0;
			retVal[3] = -49.0/18.0;
			retVal[4] = 3.0/2.0;
			retVal[5] = -3.0/20.0;
			retVal[6] = 1.0/90.0;
			break;
		case 7:
		case 8:
			L = 9;
			retVal = (float*) calloc(size_t(L), sizeof(float));
			retVal[0] = -1.0/560.0;
			retVal[1] = 8.0/315.0;
			retVal[2] = -1.0/5.0;
			retVal[3] = 8.0/5.0;
			retVal[4] = -205.0/72.0;
			retVal[5] = 8.0/5.0;
			retVal[6] = -1.0/5.0;
			retVal[7] = 8.0/315.0;
			retVal[8] = -1.0/560.0;
			break;
		case 9:
		case 10:
		default:
			L = 9;
			retVal = (float*) calloc(size_t(L), sizeof(float));
			retVal[0] = -1.0/560.0;
			retVal[1] = 8.0/315.0;
			retVal[2] = -1.0/5.0;
			retVal[3] = 8.0/5.0;
			retVal[4] = -205.0/72.0;
			retVal[5] = 8.0/5.0;
			retVal[6] = -1.0/5.0;
			retVal[7] = 8/315.0;
			retVal[8] = -1.0/560.0;
	}

	return retVal;

}
