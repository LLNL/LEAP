////////////////////////////////////////////////////////////////////////////////
// Copyright 2016-2025 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// c++ module for inpainting
////////////////////////////////////////////////////////////////////////////////

#include "inpainting.h"

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <omp.h>
#include "log.h"
#include "leap_defines.h"
#include "cpu_utils.h"

#define INF_PIXEL 1.0e16

using namespace std;

bool inpaint3D(float* data, int N_1, int N_2, int N_3)
{
	if (data == NULL || N_1 <= 0 || N_2 <= 0 || N_3 <= 0)
		return false;

	omp_set_num_threads(num_cpu_threads());
	#pragma omp parallel for
	for (int i = 0; i < N_1; i++)
	{
		float* anImage = &data[uint64(i) * uint64(N_2 * N_3)];
		image* I = new image;
		I->set(anImage, N_2, N_3);
		image* mask = I->copy(true);

		bool doProcessing = false;
		for (int j = 0; j < N_2 * N_3; j++)
		{
			//*
			if (isnan(anImage[j]))
			{
				mask->data[j] = 1.0;
				doProcessing = true;
			}
			//*/
			/*
			if (isnan(anImage[j]))
				mask->data[j] = 0.0;
			else
				mask->data[j] = 1.0;
			//*/
		}

		if (doProcessing)
		{
			inpaint2D TeleaInpainting;
			TeleaInpainting.execute(I, mask);
			delete mask;
			delete I;
		}
	}
	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// inpaint2D Class
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
inpaint2D::inpaint2D()
{
	I = NULL;
	f = NULL;
	T = NULL;
	gI_x = NULL;
	gI_y = NULL;
	gT_x = NULL;
	gT_y = NULL;

	useGradients = true;
	//useGradients = false;
	//setGradAtten(0.0);
	setGradAtten(1.0 / 3.0); // best?
	//setGradAtten(1.0/2.0);
	//setGradAtten(1.0);


	//setWindowSize(10);
	setWindowSize(4);
	//setWindowSize(2);
}

inpaint2D::~inpaint2D()
{
	clearAll();
}

bool inpaint2D::clearAll()
{
	I = NULL;
	f = NULL;

	if (T != NULL)
		delete T;
	T = NULL;

	if (gI_x != NULL)
		delete gI_x;
	gI_x = NULL;

	if (gI_y != NULL)
		delete gI_y;
	gI_y = NULL;

	if (gT_x != NULL)
		delete gT_x;
	gT_x = NULL;

	if (gT_y != NULL)
		delete gT_y;
	gT_y = NULL;

	return true;
}

bool inpaint2D::setGradAtten(double x)
{
	strangeFactor = max(0.0, min(1.0, x));
	return true;
}

bool inpaint2D::setWindowSize(int n)
{
	if (n > 0)
	{
		EPSILON_PIXEL = n;
		T_MAX = 2 * EPSILON_PIXEL;
		return true;
	}
	else
		return false;
}

bool inpaint2D::execute(image* theInput, image* theMask)
{
	//time_t startTime = time(NULL);
	if (init(theInput, theMask) == false)
		return false;

	//T->write("Tinit.tif");

	outsideFMM();
	smoothT();
	init_gI();
	init_gT();

	//T->write("T2.tif"); exit(1);
	//*
	for (int i = 0; i < f->numRows; i++)
	{
		for (int j = 0; j < f->numCols; j++)
		{
			if (f->get(i, j) == TOOFAR)
				f->set(i, j, KNOWN);
		}
	}
	//*/

	//printf("Setup time: %d s\n", int(time(NULL)-startTime));
	//startTime = time(NULL);

	insideFMM();

	//printf("Inpaint time: %d s\n", int(time(NULL)-startTime));

	//T->write("Tfinal.tif");
	//printf("clearAll...\n");
	clearAll();

	return true;
}

void inpaint2D::inpaint(int i, int j)
{
	//return inpaint_other(i, j);
	// USES: gI_x, gI_y, fine, f, T, image
	// CALLS: -
	// f->data2D[i][j] == INSIDE

	// p: (i,j), value to inpaint
	// q: (k,l), neighbor with known value

	double I_sum = 0.0;
	double w_sum = 0.0;
	double w_sum_x = 0.0;
	double w_sum_y = 0.0;
	double gradI_x = 0.0;
	double gradI_y = 0.0;
	for (int k = i - EPSILON_PIXEL; k <= i + EPSILON_PIXEL; k++)
	{
		for (int l = j - EPSILON_PIXEL; l <= j + EPSILON_PIXEL; l++)
		{
			if (k < 0 || k >= I->numRows || l < 0 || l >= I->numCols)
				continue;

			if (k == i && l == j)
				continue;

			// i - k ==> i - [i-E:i+E] == [E:-E]


			//* r = p-q
			int r_x = j - l;
			int r_y = i - k;
			//*/
			/* r = q-p
			int r_x = l - j;
			int r_y = k - i;
			//*/
			double r_length = sqrt(double(r_x * r_x) + double(r_y * r_y));

			if (r_length > EPSILON_PIXEL)
				continue;

			if (f->get(k,l) != KNOWN)
				continue;

			//printf("inpainting: (%d, %d); known pixel: (%d, %d); gT = (%f, %f); gI = (%f, %f)\n", i, j, k, l, gT_x->data2D[k][l], gT_y->data2D[k][l], gI_x->data2D[k][l], gI_y->data2D[k][l]);
			// [k][l] is a KNOWN pixel
			double dir = (r_x * gT_x->get(k,l) + r_y * gT_y->get(k,l)) / r_length;
			dir = abs(dir);
			double dst = 1.0 / (r_x * r_x + r_y * r_y);
			double lev = 1.0 / (1.0 + abs(T->get(k,l) - T->get(i,j)));
			//double lev = 1.0 / (1.0 + (T->data2D[k][l] - T->data2D[i][j])*(T->data2D[k][l] - T->data2D[i][j]));
			double w = dir * dst * lev;
			//w = dst;
			//double w = dir;
			//w = 1.0;

			//printf("[%d, %d]: dir%f dst%f lev%f\n", k, l, dir, dst, lev);

			I_sum += w * I->get(k,l);
			w_sum += w;

			if (k > 0 && k < f->numRows - 1 && f->get(k - 1,l) != INSIDE && f->get(k + 1,l) != INSIDE)
			{
				gradI_y += w * double(r_y) * strangeFactor * 0.5 * (I->get(k + 1,l) - I->get(k - 1,l));
				w_sum_y += w;
			}
			else if (k > 0 && f->get(k - 1,l) != INSIDE)
			{
				gradI_y += w * double(r_y) * strangeFactor * (I->get(k,l) - I->get(k - 1,l));
				w_sum_y += w;
			}
			else if (k < f->numRows - 1 && f->get(k + 1,l) != INSIDE)
			{
				gradI_y += w * double(r_y) * strangeFactor * (I->get(k + 1,l) - I->get(k,l));
				w_sum_y += w;
			}

			if (l > 0 && l < f->numCols - 1 && f->get(k,l - 1) != INSIDE && f->get(k,l + 1) != INSIDE)
			{
				gradI_x += w * double(r_x) * strangeFactor * 0.5 * (I->get(k,l + 1) - I->get(k,l - 1));
				w_sum_x += w;
			}
			else if (l > 0 && f->get(k,l - 1) != INSIDE)
			{
				gradI_x += w * double(r_x) * strangeFactor * (I->get(k,l) - I->get(k,l - 1));
				w_sum_x += w;
			}
			else if (l < f->numCols - 1 && f->get(k,l + 1) != INSIDE)
			{
				gradI_x += w * double(r_x) * strangeFactor * (I->get(k,l + 1) - I->get(k,l));
				w_sum_x += w;
			}
			/*
			double gradI_x = 0.0;
			double gradI_y = 0.0;
			if (k > 0 && k < f->numRows-1 && l > 0 && l < f->numCols-1 && f->data2D[k-1][l] != INSIDE && f->data2D[k][l-1] != INSIDE &&
				f->data2D[k][l+1] != INSIDE && f->data2D[k+1][l] != INSIDE)
			{
				gradI_x = gI_x->data2D[k][l];
				gradI_y = gI_y->data2D[k][l];
			}
			I_sum += w*(I->data2D[k][l] +
							(gradI_x*double(r_x) + gradI_y*double(r_y)));
			w_sum += w;
			//*/
		}
	}
	if (w_sum == 0.0 || w_sum_x == 0.0 || w_sum_y == 0.0)
	{
		// Cannot inpaint current pixel.  Likely because surrounding pixel gradients are not defined
		// In this case just take average of known neighboring pixels
		if (w_sum == 0.0)
		{
			I->set(i,j, 0.0);
			I_sum = 0.0;
			if (i - 1 >= 0 && f->get(i - 1,j) == KNOWN)
			{
				I_sum += I->get(i - 1,j);
				w_sum += 1.0;
			}
			if (j - 1 >= 0 && f->get(i,j - 1) == KNOWN)
			{
				I_sum += I->get(i,j - 1);
				w_sum += 1.0;
			}
			if (i + 1 < f->numRows && f->get(i + 1,j) == KNOWN)
			{
				I_sum += I->get(i + 1,j);
				w_sum += 1.0;
			}
			if (j + 1 < f->numCols && f->get(i,j + 1) == KNOWN)
			{
				I_sum += I->get(i,j + 1);
				w_sum += 1.0;
			}
		}
		if (w_sum > 0.0)
			I->set(i, j, I_sum / w_sum);
		else
		{
			/*
			printf("inpainting set with a zero denominator (%d, %d)!\n", i, j);
			//printf("flags: TOOFAR=%d, KNOWN=%d, BAND=%d, INSIDE=%d\n", TOOFAR, KNOWN, BAND, INSIDE);
			printf("neighboring flags: %.0f, %.0f, %.0f, %.0f\n", f->data2D[i-1][j], f->data2D[i+1][j], f->data2D[i][j-1], f->data2D[i][j+1]);
			printf("\n");
			//*/
			//exit(1);
		}
	}
	else
	{
		//printf("successful inpaint!\n");
		//I->data2D[i][j] = I_sum / w_sum;

		if (w_sum_x > 0.0)
			gradI_x = gradI_x / w_sum_x;
		if (w_sum_y > 0.0)
			gradI_y = gradI_y / w_sum_y;

		/* What is this?!
		double gradMag = sqrt(gradI_x*gradI_x + gradI_y*gradI_y);
		if (gradMag > 1.0e-5)
		{
			gradI_x = gradI_x / gradMag;
			gradI_y = gradI_y / gradMag;
		}
		//*/
		if (useGradients == false)
			I->set(i,j, I_sum / w_sum);
		else
			I->set(i, j, I_sum / w_sum + gradI_x + gradI_y);
	}
	//exit(1);

	return;
}

void inpaint2D::inpaint_other(int i, int j)
{
	// USES: gI_x, gI_y, fine, f, T, image
	// CALLS: -
	// f->data2D[i][j] == INSIDE

	// p: (i,j), value to inpaint
	// q: (k,l), neighbor with known value

	double I_sum = 0.0;
	double w_sum = 0.0;
	double w_sum_x = 0.0;
	double w_sum_y = 0.0;
	double gradI_x = 0.0;
	double gradI_y = 0.0;
	for (int k = i - EPSILON_PIXEL; k <= i + EPSILON_PIXEL; k++)
	{
		for (int l = j - EPSILON_PIXEL; l <= j + EPSILON_PIXEL; l++)
		{
			if (k < 0 || k >= I->numRows || l < 0 || l >= I->numCols)
				continue;

			if (k == i && l == j)
				continue;

			// i - k ==> i - [i-E:i+E] == [E:-E]


			//* r = p-q
			int r_x = j - l;
			int r_y = i - k;
			//*/
			/* r = q-p
			int r_x = l - j;
			int r_y = k - i;
			//*/
			double r_length = sqrt(double(r_x * r_x) + double(r_y * r_y));

			if (r_length > EPSILON_PIXEL)
				continue;

			if (f->get(k,l) != KNOWN)
				continue;

			//printf("inpainting: (%d, %d); known pixel: (%d, %d); gT = (%f, %f); gI = (%f, %f)\n", i, j, k, l, gT_x->data2D[k][l], gT_y->data2D[k][l], gI_x->data2D[k][l], gI_y->data2D[k][l]);
			// [k][l] is a KNOWN pixel
			double dir = (r_x * gT_x->get(k,l) + r_y * gT_y->get(k,l)) / r_length;
			dir = abs(dir);
			double dst = 1.0 / (r_x * r_x + r_y * r_y);
			double lev = 1.0 / (1.0 + abs(T->get(k,l) - T->get(i,j)));
			//double lev = 1.0 / (1.0 + (T->data2D[k][l] - T->data2D[i][j])*(T->data2D[k][l] - T->data2D[i][j]));
			double w = dir * dst * lev;
			//double w = dir;
			//w = 1.0;

			//printf("[%d, %d]: dir%f dst%f lev%f\n", k, l, dir, dst, lev);

			if (k > 0 && k < f->numRows - 1 && f->get(k - 1,l) != INSIDE && f->get(k + 1,l) != INSIDE
				&& l > 0 && l < f->numCols - 1 && f->get(k,l - 1) != INSIDE && f->get(k,l + 1) != INSIDE)
			{
				gradI_y += w * double(r_y) * strangeFactor * 0.5 * (I->get(k + 1,l) - I->get(k - 1,l));
				w_sum_y += w;

				gradI_x += w * double(r_x) * strangeFactor * 0.5 * (I->get(k,l + 1) - I->get(k,l - 1));
				w_sum_x += w;

				I_sum += w * I->get(k,l);
				w_sum += w;
			}
			/*
			double gradI_x = 0.0;
			double gradI_y = 0.0;
			if (k > 0 && k < f->numRows-1 && l > 0 && l < f->numCols-1 && f->data2D[k-1][l] != INSIDE && f->data2D[k][l-1] != INSIDE &&
				f->data2D[k][l+1] != INSIDE && f->data2D[k+1][l] != INSIDE)
			{
				gradI_x = gI_x->data2D[k][l];
				gradI_y = gI_y->data2D[k][l];
			}
			I_sum += w*(I->data2D[k][l] +
							(gradI_x*double(r_x) + gradI_y*double(r_y)));
			w_sum += w;
			//*/
		}
	}
	if (w_sum == 0.0 || w_sum_x == 0.0 || w_sum_y == 0.0)
	{
		// Cannot inpaint current pixel.  Likely because surrounding pixel gradients are not defined
		// In this case just take average of known neighboring pixels
		if (w_sum == 0.0)
		{
			I->set(i,j, 0.0);
			I_sum = 0.0;
			if (i - 1 >= 0 && f->get(i - 1,j) == KNOWN)
			{
				I_sum += I->get(i - 1,j);
				w_sum += 1.0;
			}
			if (j - 1 >= 0 && f->get(i,j - 1) == KNOWN)
			{
				I_sum += I->get(i,j - 1);
				w_sum += 1.0;
			}
			if (i + 1 < f->numRows && f->get(i + 1,j) == KNOWN)
			{
				I_sum += I->get(i + 1,j);
				w_sum += 1.0;
			}
			if (j + 1 < f->numCols && f->get(i,j + 1) == KNOWN)
			{
				I_sum += I->get(i,j + 1);
				w_sum += 1.0;
			}
		}
		if (w_sum > 0.0)
			I->set(i, j, I_sum / w_sum);
		else
		{
			/*
			printf("inpainting set with a zero denominator (%d, %d)!\n", i, j);
			//printf("flags: TOOFAR=%d, KNOWN=%d, BAND=%d, INSIDE=%d\n", TOOFAR, KNOWN, BAND, INSIDE);
			printf("neighboring flags: %.0f, %.0f, %.0f, %.0f\n", f->data2D[i-1][j], f->data2D[i+1][j], f->data2D[i][j-1], f->data2D[i][j+1]);
			printf("\n");
			//*/
			//exit(1);
		}
	}
	else
	{
		//printf("successful inpaint!\n");
		//I->data2D[i][j] = I_sum / w_sum;

		if (w_sum_x > 0.0)
			gradI_x = gradI_x / w_sum_x;
		if (w_sum_y > 0.0)
			gradI_y = gradI_y / w_sum_y;

		/* What is this?!
		double gradMag = sqrt(gradI_x*gradI_x + gradI_y*gradI_y);
		if (gradMag > 1.0e-5)
		{
			gradI_x = gradI_x / gradMag;
			gradI_y = gradI_y / gradMag;
		}
		//*/
		if (useGradients == false)
			I->set(i, j, I_sum / w_sum);
		else
			I->set(i, j, I_sum / w_sum + gradI_x + gradI_y);
	}
	//exit(1);

	return;

}

double inpaint2D::solve(int i1, int j1, int i2, int j2, int TYPE1, int TYPE2)
{
	// USES: f, T
	// CALLS: -
	// TYPE1 == BAND, TYPE2 == KNOWN
	double sol = INF_PIXEL;
	//*
	bool useful1 = false; bool useful2 = false;
	if ((i1 >= 0 && i1 < f->numRows && j1 >= 0 && j1 < f->numCols) &&
		(f->get(i1,j1) == TYPE1 || f->get(i1,j1) == TYPE2))
		useful1 = true;
	if ((i2 >= 0 && i2 < f->numRows && j2 >= 0 && j2 < f->numCols) &&
		(f->get(i2,j2) == TYPE1 || f->get(i2,j2) == TYPE2))
		useful2 = true;

	if (useful1 && useful2)
	{
		double r = (2.0 - (T->get(i1,j1) - T->get(i2,j2)) * (T->get(i1,j1) - T->get(i2,j2)));
		if (r < 0.0)
		{
			//*
			printf("inpaint2D: Cannot take sqrt of negative number!\n");
			printf("T->data2D[i1][j1] = %f\n", T->get(i1,j1));
			printf("T->data2D[i2][j2] = %f\n", T->get(i2,j2));
			exit(1);
			//*/
			//return sol;
		}
		r = sqrt(r);
		// r^2 should be positive
		double s = (T->get(i1,j1) + T->get(i2,j2) - r) / 2.0;
		//double s = (T->data2D[i1][j1] + T->data2D[i2][j2] + r) / 2.0;
		if (s >= T->get(i1,j1) && s >= T->get(i2,j2))
			sol = s;
		else
		{
			s += r;
			if (s >= T->get(i1,j1) && s >= T->get(i2,j2))
				sol = s;
		}
	}
	else if (useful1)
		sol = 1.0 + T->get(i1,j1);
	else if (useful2)
		sol = 1.0 + T->get(i2,j2);
	//*/

	/*
	if (i1 >= 0 && i1 < f->numRows && j1 >= 0 && j1 < f->numCols && (f->data2D[i1][j1] == KNOWN || f->data2D[i1][j1] == TOOFAR))
	{
		if (i2 >= 0 && i2 < f->numRows && j2 >= 0 && j2 < f->numCols && (f->data2D[i2][j2] == KNOWN || f->data2D[i2][j2] == TOOFAR))
		{
			double r = (2.0 - (T->data2D[i1][j1]-T->data2D[i2][j2])*(T->data2D[i1][j1]-T->data2D[i2][j2]));
			if (r < 0.0)
			{
				printf("Cannot take sqrt of negative number!\n");
				exit(1);
			}
			r = sqrt(r);
			double s = (T->data2D[i1][j1] + T->data2D[i2][j2] - r) / 2.0;
			//double s = (T->data2D[i1][j1] + T->data2D[i2][j2] + r) / 2.0;
			if (s >= T->data2D[i1][j1] && s >= T->data2D[i2][j2])
				sol = s;
			else
			{
				s += r;
				if (s >= T->data2D[i1][j1] && s >= T->data2D[i2][j2])
					sol = s;
			}
		}
		else
			sol = 1.0 + T->data2D[i1][j1];
	}
	else if (i2 >= 0 && i2 < f->numRows && j2 >= 0 && j2 < f->numCols && (f->data2D[i2][j2] == KNOWN || f->data2D[i2][j2] == TOOFAR))
		sol = 1.0 + T->data2D[i2][j2];
	//*/

	return sol;
}

void inpaint2D::insideFMM()
{
	//T->write("Tinpaint_0.tif"); exit(1);

	// USES: f, heap, T
	// CALLS: solve, inpaint

	for (int i = 0; i < I->numRows; i++)
	{
		for (int j = 0; j < I->numCols; j++)
		{
			if (f->get(i,j) == BAND)
				heap.push(xyCoord(i, j));
		}
	}

	while (!heap.isEmpty())
	{
		//printf("size = %d\n", int(heap.coords.size()));
		xyCoord p = heap.pop();
		int i = p.y; int j = p.x;
		//i = 107-1;
		//j = 1325;

		f->set(i, j, KNOWN);

		// f[i, j] must used to be BAND and its neighbour may be
		// INSIDE, BAND, KNOWN, (TOOFAR.) INSIDE are targets

		int nbs[4][2] = { {i - 1,j},{i,j - 1},{i,j + 1},{i + 1,j} };
		for (int n = 0; n < 4; n++)
		{
			//printf("n = %d\n", n);
			int k = nbs[n][0];
			int l = nbs[n][1];

			// check validity

			if (k == i - 1 && k < 0)
				continue;
			else if (k == i + 1 && k >= f->numRows)
				continue;
			else if (l == j - 1 && l < 0)
				continue;
			else if (l == j + 1 && l >= f->numCols)
				continue;

			if (f->get(k,l) != INSIDE)
				continue;

			//if (f->data2D[k][l] != KNOWN)
			//{
				//if (f->data2D[k][l] == INSIDE)
				//{
					//printf("inpaint...\n");
			inpaint(k, l);
			f->set(k, l, BAND);
			//printf("renew_gI_gT...\n");
			renew_gI_gT(k, l);
			//}

			//printf("solve x4...\n");
			T->set(k,l, min4(solve(k - 1, l, k, l - 1, BAND, KNOWN),
				solve(k + 1, l, k, l - 1, BAND, KNOWN),
				solve(k - 1, l, k, l + 1, BAND, KNOWN),
				solve(k + 1, l, k, l + 1, BAND, KNOWN)));

			heap.push(xyCoord(k, l));
			//}
		}
		//return;
	}

	return;
}

void inpaint2D::outsideFMM()
{
	// At this point values of f are set to TOOFAR, INSIDE, or BAND and T == 0
	// USES: f, heap, T
	// CALLS: solve

	image* f_save = f->copy();

	// 1. 2.
	// Push BAND pixels onto the heap
	// Set T to 10^6 for known pixels, otherwise should be zero
	for (int i = 0; i < I->numRows; i++)
	{
		for (int j = 0; j < I->numCols; j++)
		{
			if (f->get(i,j) == BAND)
				heap.push(xyCoord(i, j));
			else if (f->get(i,j) == TOOFAR)
			{
				T->set(i, j, INF_PIXEL);
				//f->data2D[i][j] = KNOWN; // I added this; why do we need TOOFAR label?; might mess up calculation of T
			}
		}
	}

	// Now T is either 10^6 (for TOOFAR pixels) or zero
	while (!heap.isEmpty())
	{
		//printf("size = %d\n", int(heap.coords.size()));

		xyCoord p = heap.pop();
		int i = p.y; int j = p.x;

		// f[i, j] may be BAND or KNOWN and its neighbour may be
		// INSIDE, BAND, KNOWN, TOOFAR, but only TOOFAR is target

		int nbs[4][2] = { {i - 1,j},{i,j - 1},{i,j + 1},{i + 1,j} };
		for (int n = 0; n < 4; n++)
		{
			int k = nbs[n][0];
			int l = nbs[n][1];

			// check validity
			if (k == i - 1 && k < 0)
				continue;
			else if (k == i + 1 && k >= f->numRows)
				continue;
			else if (l == j - 1 && l < 0)
				continue;
			else if (l == j + 1 && l >= f->numCols)
				continue;

			if (f->get(k,l) != TOOFAR)
				continue;

			// Calculate T for pixels labeled TOOFAR
			T->set(k, l, min4(solve(k - 1, l, k, l - 1, BAND, KNOWN),
				solve(k + 1, l, k, l - 1, BAND, KNOWN),
				solve(k - 1, l, k, l + 1, BAND, KNOWN),
				solve(k + 1, l, k, l + 1, BAND, KNOWN)));

			if (T->get(k,l) < T_MAX)
			{
				f->set(k, l, KNOWN);
				heap.push(xyCoord(k, l));
			}
		}
	}
	//T->write("outsideFMM.tif"); exit(1);

	// inverse
	for (int i = 0; i < I->numRows; i++)
	{
		for (int j = 0; j < I->numCols; j++)
		{
			if (f->get(i,j) == KNOWN || f->get(i,j) == TOOFAR)
				T->data[i*T->numCols+j] *= -1.0;
		}
	}
	f->equal(f_save);
	delete f_save;

	return;
}

bool inpaint2D::isBAND(image* J, int i, int j)
{
	if (J->get(i,j) == INSIDE)
		return false;
	else if (i > 0 && J->get(i - 1,j) == INSIDE)
		return true;
	else if (i < J->numRows - 1 && J->get(i + 1,j) == INSIDE)
		return true;
	else if (j > 0 && J->get(i,j - 1) == INSIDE)
		return true;
	else if (j < J->numCols - 1 && J->get(i,j + 1) == INSIDE)
		return true;
	else
		return false;
}

bool inpaint2D::init(image* theInput, image* theMask)
{
	// Sets f to TOOFAR, INSIDE, or BAND

	if (theInput == NULL || theMask == NULL || theInput->data == NULL || theMask->data == NULL || theInput->numRows != theMask->numRows || theInput->numCols != theMask->numCols)
		return false;

	I = theInput;
	f = theMask;

	// Set f to INSIDE if > 0
	// Set f to TOOFAR otherwise
	for (int i = 0; i < f->numRows; i++)
	{
		for (int j = 0; j < f->numCols; j++)
		{
			if (f->get(i,j) > 0.0)
				f->set(i, j, INSIDE);
			else
				f->set(i, j, TOOFAR);
		}
	}

	// If not set to INSIDE and one of the neighbors is set to INSIDE, switch from TOOFAR to BAND
	image* mask_save = f->copy();
	for (int i = 0; i < I->numRows; i++)
	{
		for (int j = 0; j < I->numCols; j++)
		{
			if (isBAND(mask_save, i, j))
				f->set(i, j, BAND);
		}
	}
	delete mask_save;

	// Initialize the image pixels labeled as INSIDE to zero
	for (int i = 0; i < I->numRows; i++)
	{
		for (int j = 0; j < I->numCols; j++)
		{
			if (f->get(i,j) == INSIDE)
				I->set(i, j, 0.0);
		}
	}
	T = I->copy(true);
	heap.init(T);

	return true;
}

void inpaint2D::get_gI(int i, int j)
{
	if (i < 0 || i >= f->numRows || j < 0 || j >= f->numCols)
		return;

	// gI_x

	int d = 0;
	double sum = 0.0;

	if (j - 1 >= 0 && f->get(i,j - 1) != INSIDE)
	{
		d += 1;
		sum += I->get(i,j) - I->get(i,j - 1);
	}
	if (j + 1 < I->numCols && f->get(i,j + 1) != INSIDE)
	{
		d += 1;
		sum += I->get(i,j + 1) - I->get(i,j);
	}
	if (d != 0)
	{
		sum = sum * strangeFactor / double(d);
		gI_x->set(i, j, sum);
	}

	// gI_y
	d = 0;
	sum = 0.0;
	if (i - 1 >= 0 && f->get(i - 1,j) != INSIDE)
	{
		d += 1;
		sum += I->get(i,j) - I->get(i - 1,j);
	}
	if (i + 1 < I->numRows && f->get(i + 1,j) != INSIDE)
	{
		d += 1;
		sum += I->get(i + 1,j) - I->get(i,j);
	}
	if (d != 0)
	{
		sum = sum * strangeFactor / double(d);
		gI_y->set(i, j, sum);
	}

	return;
}

void inpaint2D::init_gI()
{
	if (gI_x != NULL)
		delete gI_x;
	gI_x = I->copy(true);
	if (gI_y != NULL)
		delete gI_y;
	gI_y = I->copy(true);

	for (int i = 0; i < I->numRows; i++)
	{
		for (int j = 0; j < I->numCols; j++)
			get_gI(i, j);
	}
	//smoothI();

	return;
}

void inpaint2D::get_gT(int i, int j)
{
	if (i < 0 || i >= f->numRows || j < 0 || j >= f->numCols)
		return;

	// gT_x
	int d = 0;
	double sum = 0.0;
	if (j - 1 >= 0 && f->get(i,j - 1) != INSIDE)
	{
		d += 1;
		sum += T->get(i,j) - T->get(i,j - 1);
	}
	if (j + 1 < I->numCols && f->get(i,j + 1) != INSIDE)
	{
		d += 1;
		sum += T->get(i,j + 1) - T->get(i,j);
	}
	if (d != 0)
	{
		sum = sum / double(d);
		gT_x->set(i, j, sum);
	}

	// gT_y
	d = 0;
	sum = 0;
	if (i - 1 >= 0 && f->get(i - 1,j) != INSIDE)
	{
		d += 1;
		sum += T->get(i,j) - T->get(i - 1,j);
	}
	if (i + 1 < I->numRows && f->get(i + 1,j) != INSIDE)
	{
		d += 1;
		sum += T->get(i + 1,j) - T->get(i,j);
	}
	if (d != 0)
	{
		sum = sum / double(d);
		gT_y->set(i, j, sum);
	}

	return;
}

void inpaint2D::init_gT()
{
	if (gT_x != NULL)
		delete gT_x;
	gT_x = I->copy(true);
	if (gT_y != NULL)
		delete gT_y;
	gT_y = I->copy(true);

	for (int i = 0; i < I->numRows; i++)
	{
		for (int j = 0; j < I->numCols; j++)
			get_gT(i, j);
	}

	return;
}

void inpaint2D::renew_gI_gT(int i, int j)
{
	get_gI(i - 1, j);
	get_gI(i, j - 1);
	get_gI(i, j);
	get_gI(i, j + 1);
	get_gI(i + 1, j);

	get_gT(i - 1, j);
	get_gT(i, j - 1);
	get_gT(i, j);
	get_gT(i, j + 1);
	get_gT(i + 1, j);

	return;
}

double inpaint2D::min4(double a, double b, double c, double d)
{
	double r = a < b ? a : b;
	r = r < c ? r : c;
	r = r < d ? r : d;
	return r;
}

void inpaint2D::smoothT()
{
	image* T2 = T->copy();
	double h[3]; h[0] = 0.25; h[1] = 0.5; h[2] = 0.25;
	for (int i = 1; i < T->numRows - 1; i++)
	{
		for (int j = 0; j < T->numCols; j++)
			T->data[i*T->numCols+j] = h[0] * T2->get(i - 1,j) + h[1] * T2->get(i,j) + h[2] * T2->get(i + 1,j);
	}

	T2->equal(T);
	for (int i = 0; i < T->numRows; i++)
	{
		for (int j = 1; j < T->numCols - 1; j++)
			T->data[i*T->numCols+j] = h[0] * T2->get(i,j - 1) + h[1] * T2->get(i,j) + h[2] * T2->get(i,j + 1);
	}
	delete T2;
}

void inpaint2D::smoothI()
{
	image* I2 = I->copy();
	double h[3]; h[0] = 0.25; h[1] = 0.5; h[2] = 0.25;
	for (int i = 1; i < I->numRows - 1; i++)
	{
		for (int j = 0; j < I->numCols; j++)
			I->data[i*I->numCols+j] = h[0] * I2->get(i - 1,j) + h[1] * I2->get(i,j) + h[2] * I2->get(i + 1,j);
	}

	I2->equal(I);
	for (int i = 0; i < I->numRows; i++)
	{
		for (int j = 1; j < I->numCols - 1; j++)
			I->data[i*I->numCols+j] = h[0] * I2->get(i,j - 1) + h[1] * I2->get(i,j) + h[2] * I2->get(i,j + 1);
	}
	delete I2;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Helper classes
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
xyCoord::xyCoord()
{
	y = 0;
	x = 0;
}

xyCoord::xyCoord(int y_in, int x_in)
{
	y = y_in;
	x = x_in;
}

xyCoord::~xyCoord()
{
	y = 0;
	x = 0;
}

imageHeap::imageHeap()
{
	coords.clear();
	I = NULL;
}

imageHeap::~imageHeap()
{
	coords.clear();
	I = NULL;
}

bool imageHeap::init(image* input)
{
	coords.clear();
	I = input;
	return true;
}

bool imageHeap::isEmpty()
{
	return coords.empty();
}

xyCoord imageHeap::pop()
{
	// grab smallest element which is at the end
	xyCoord retVal;
	if (!isEmpty())
	{
		retVal = coords.back();
		coords.pop_back();
	}
	return retVal;
}

bool imageHeap::push(xyCoord newCoord)
{
	// biggest element is at the beginning
	/*
	if (coords.size() > 1 && compare_larger(newCoord, coords[coords.size()-1]) == false)
	{
		coords.insert(coords.begin()+coords.size(), newCoord);
		return true;
	}
	//*/
	int i;
	for (i = 0; i < int(coords.size()); i++)
	{
		if (compare_larger(newCoord, coords[i]) == true)
			break;
	}
	coords.insert(coords.begin() + i, newCoord);

	return true;
}

bool imageHeap::compare_larger(const xyCoord& a, const xyCoord& b)
{
	return (I->get(a.y, a.x) > I->get(b.y, b.x));
}

bool imageHeap::printAll()
{
	for (int i = 0; i < int(coords.size()); i++)
		printf("(%d, %d): %f\n", coords[i].y, coords[i].x, I->get(coords[i].y, coords[i].x));
	return true;
}

image::image()
{
	data = NULL;
	numRows = 0;
	numCols = 0;
	ownsData = true;
}

image::image(int M, int N)
{
	data = NULL;
	numRows = 0;
	numCols = 0;
	ownsData = true;
	if (M > 0 && N > 0)
	{
		numRows = M;
		numCols = N;
		malloc();
	}
}

image::~image()
{
	clearAll();
}

void image::free()
{
	if (data != NULL && ownsData)
		delete[] data;
	data = NULL;
	ownsData = true;
}

void image::clearAll()
{
	free();
	numRows = 0;
	numCols = 0;
}

bool image::malloc()
{
	if (numRows > 0 && numCols > 0)
	{
		free();
		data = new float[numRows * numCols];
		return true;
	}
	else
		return false;
}

image* image::copy(bool setToZero)
{
	image* aCopy = new image;
	aCopy->numRows = numRows;
	aCopy->numCols = numCols;
	if (aCopy->malloc())
	{
		if (setToZero)
			memset(aCopy->data, 0, sizeof(float) * numRows * numCols);
		else
			memcpy(aCopy->data, data, sizeof(float) * numRows * numCols);
	}
	return aCopy;
}

image* image::equal(image* lhs)
{
	if (lhs != NULL && data != NULL && lhs->data != NULL && numRows == lhs->numRows && numCols == lhs->numCols)
	{
		memcpy(data, lhs->data, sizeof(float) * numRows * numCols);
		return this;
	}
	else
		return NULL;
}

float image::get(int i, int j)
{
	if (i < 0 || i >= numRows || j < 0 || j >= numCols || data == NULL)
		return 0.0;
	else
		return data[i * numCols + j];
}

bool image::set(int i, int j, float val)
{
	if (i < 0 || i >= numRows || j < 0 || j >= numCols || data == NULL)
		return false;
	else
	{
		data[i*numCols + j] = val;
		return true;
	}
}

bool image::set(float* data_in, int M, int N)
{
	if (data_in == NULL || M <= 0 || N <= 0)
		return false;
	else
	{
		clearAll();
		data = data_in;
		ownsData = false;
		numRows = M;
		numCols = N;
		return true;
	}
}
