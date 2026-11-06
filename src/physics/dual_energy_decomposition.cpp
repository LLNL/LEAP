#include "dual_energy_decomposition.h"
#include <math.h>

#ifdef USE_OPENMP
#include <omp.h>
#endif

dualEnergyDecomposition::dualEnergyDecomposition()
{
	d_L = NULL;
	d_H = NULL;

	b_L = NULL;
	b_H = NULL;

	gammas = NULL;
}

dualEnergyDecomposition::~dualEnergyDecomposition()
{
	clearAll();
}

bool dualEnergyDecomposition::clearAll()
{
	if (d_L != NULL)
		delete[] d_L;
	d_L = NULL;
	if (d_H != NULL)
		delete[] d_H;
	d_H = NULL;

	if (b_L != NULL)
		delete[] b_L;
	b_L = NULL;
	if (b_H != NULL)
		delete[] b_H;
	b_H = NULL;

	if (gammas != NULL)
		delete[] gammas;
	gammas = NULL;

	return true;
}

double dualEnergyDecomposition::gamma(int i)
{
	if (0 <= i && i <= N_gamma - 1)
		return gammas[i];
	else
		return 0.0;
}

double dualEnergyDecomposition::gamma_inv(double val)
{
	if (val <= gammas[0])
		return 0.0;
	// energies[0] < val
	for (int i = 1; i < N_gamma; i++)
	{
		if (val <= gammas[i])
		{
			// energies[i-1] <= val <= energies[i]

			float d = (val - gammas[i - 1]) / (gammas[i] - gammas[i - 1]);

			return float(i - 1) + d;
		}
	}

	return float(N_gamma - 1);
}

double dualEnergyDecomposition::T_gamma(int i)
{
	if (i == 0)
		return gammas[i + 1] - gammas[i];
	else if (i == N_gamma - 1)
		return gammas[i] - gammas[i - 1];
	else
		return 0.5 * (gammas[i + 1] - gammas[i - 1]);
}

bool dualEnergyDecomposition::generateDEDlookUpTables(float* spectralResponses_in, float* gammas_in, int N_gamma_in, float* referenceEnergies, float* basisFunctions_in, float* LUT, float T_lac, int N_lac)
{
	// FIXME: need error checking
	clearAll();

	// Set energy array
	N_gamma = N_gamma_in;
	gammas = new double[N_gamma];
	for (int i = 0; i < N_gamma; i++)
		gammas[i] = gammas_in[i];

	// Set normalized spectra arrays
	d_L = new double[N_gamma];
	d_H = new double[N_gamma];
	double accum_L = 0.0;
	double accum_H = 0.0;
	for (int i = 0; i < N_gamma; i++)
	{
		d_L[i] = spectralResponses_in[i];
		accum_L += d_L[i] * T_gamma(i);

		d_H[i] = spectralResponses_in[N_gamma+i];
		accum_H += d_H[i] * T_gamma(i);
	}
	for (int i = 0; i < N_gamma; i++)
	{
		d_L[i] = d_L[i] / accum_L;
		d_H[i] = d_H[i] / accum_H;
		//printf("%f: %f, %f\n", gammas[i], d_L[i], d_H[i]);
	}

	// Set reference energies
	ref_L = referenceEnergies[0];
	ref_H = referenceEnergies[1];

	// Set energy basis functions
	b_L = new double[N_gamma];
	b_H = new double[N_gamma];
	float* b_1 = &basisFunctions_in[0];
	float* b_2 = &basisFunctions_in[N_gamma];
	double ind_L = gamma_inv(ref_L);
	int ind_Li = int(floor(ind_L));
	double h_L = ind_L - double(ind_Li);
	double ind_H = gamma_inv(ref_H);
	int ind_Hi = int(floor(ind_H));
	double h_H = ind_H - double(ind_Hi);

	double A[4];
	A[0] = (1.0 - h_L) * b_1[ind_Li] + h_L * b_1[int(ceil(ind_L))];
	A[1] = (1.0 - h_L) * b_2[ind_Li] + h_L * b_2[int(ceil(ind_L))];
	A[2] = (1.0 - h_H) * b_1[ind_Hi] + h_H * b_1[int(ceil(ind_H))];
	A[3] = (1.0 - h_H) * b_2[ind_Hi] + h_H * b_2[int(ceil(ind_H))];

	inverse2x2(A);
	for (int i = 0; i < N_gamma; i++)
	{
		b_L[i] = b_1[i] * A[0] + b_2[i] * A[2];
		b_H[i] = b_1[i] * A[1] + b_2[i] * A[3];
		//printf("%f: %f, %f\n", gammas[i], b_L[i], b_H[i]);
	}

	// Build DED LUT
	#ifdef USE_OPENMP
	omp_set_num_threads(omp_get_num_procs());
	#pragma omp parallel for schedule(dynamic)
	#endif
	for (int i = 0; i < N_lac; i++)
	{
		float* LUT_L = &LUT[i * N_lac];
		float* LUT_H = &LUT[N_lac* N_lac + i * N_lac];
		float* LUT_error = &LUT[2*N_lac * N_lac + i * N_lac];
		double g_L = double(i) * T_lac;

		if (g_L > 0.0)
		{
			LUT_L[0] = 0.0;
			LUT_H[0] = 0.0;
			LUT_error[0] = 0.0;
			for (int j = 1; j < N_lac; j++)
			{
				double g_H = double(j) * T_lac;

				double g_init_L = LUT_L[j - 1];
				double g_init_H = LUT_H[j - 1];
				if (g_init_L == 0.0)
					g_init_L = g_L;
				if (g_init_H == 0.0)
					g_init_H = g_H;

				double g_mono_L, g_mono_H, theError;
				decompose(g_L, g_H, g_init_L, g_init_H, g_mono_L, g_mono_H, theError);

				//printf("(%f, %f) ==> (%f, %f)\n", g_L, g_H, g_mono_L, g_mono_H);
				LUT_L[j] = float(g_mono_L);
				LUT_H[j] = float(g_mono_H);
				LUT_error[j] = float(sqrt(2.0*theError));
				/*
				if (g_L < g_H)
				{
					LUT_L[j] = g_L;
					LUT_H[j] = g_H;
				}
				//*/

				//LUT_L[j] = float(g_L);
				//LUT_H[j] = float(g_H);
			}
		}
		else
		{
			for (int j = 0; j < N_lac; j++)
			{
				LUT_L[j] = 0.0;
				LUT_H[j] = 0.0;
				LUT_error[j] = 0.0;
			}
		}
	}
	return true;
}

bool dualEnergyDecomposition::inverse2x2(double* A)
{
	double det = A[0] * A[3] - A[1] * A[2];
	if (det == 0.0)
		return false;
	else
	{
		double temp = A[0];
		A[0] = A[3] / det;
		A[3] = temp / det;
		A[1] *= -1.0 / det;
		A[2] *= -1.0 / det;
		return true;
	}
}

bool dualEnergyDecomposition::positiveDefinite2x2(double* A)
{
	double det = A[0] * A[3] - A[1] * A[2];
	if (det > 0.0 && A[0] > 0.0)
		return true;
	else
		return false;
}

bool dualEnergyDecomposition::decompose(double g_L, double g_H, double g_init_L, double g_init_H, double& g_mono_L, double& g_mono_H, double& theError)
{
	g_mono_L = g_init_L;
	g_mono_H = g_init_H;

	if (g_L <= 0.0 || g_H <= 0.0 /*|| (ref_L < ref_H && g_L <= g_H) || (ref_L > ref_H && g_H <= g_L)*/)
	{
		theError = 0.0;
		return true;
	}

	double g_meas[2];
	g_meas[0] = g_L;
	g_meas[1] = g_H;

	double g[2];
	g[0] = g_init_L;
	g[1] = g_init_H;
	double g_1_init = g[0];
	double g_2_init = g[1];

	double grad[2];
	int maxIter = 30;
	//maxIter = 10;
	int maxSubIter = 30;
	double epsilon = 1.0e-6; // better than 19 bit resolution
	double tol = 0.5 * epsilon * epsilon;
	int n;
	int numUpdates;

	double* polyTrans_0 = calcPolyTrans(g_meas);
	double startError_0 = dualEnergyDecomposition_cost(g_meas, g_meas, polyTrans_0);

	double* polyTrans = calcPolyTrans(g);
	double startError = dualEnergyDecomposition_cost(g_meas, g, polyTrans);

	if (startError_0 < startError)
	{
		startError = startError_0;
		free(polyTrans);
		polyTrans = polyTrans_0;
		g[0] = g_meas[0];
		g[1] = g_meas[1];
		g_1_init = g[0];
		g_2_init = g[1];
	}
	else
	{
		free(polyTrans_0);
	}

	double curError = startError;
	theError = curError;

	// Solve with Newton's Method/ Steepest Descent
	double dampingParameter = 0.7;
	double H[4];
	numUpdates = 0;
	for (n = 0; n < maxIter; n++)
	{
		dampingParameter = 0.7;
		// Compute Gradient
		dualEnergyDecomposition_gradientAndHessian(g_meas, g, polyTrans, grad, H);

		// Compute y
		double y[2];
		y[0] = -grad[0];
		y[1] = -grad[1];

		double lambda = 1.0;
		double descentDirection[2];
		if (inverse2x2(H) == true && positiveDefinite2x2(H) == true)
		{
			descentDirection[0] = H[0] * y[0] + H[1] * y[1];
			descentDirection[1] = H[2] * y[0] + H[3] * y[1];

			lambda = dampingParameter;

			if (g[0] + lambda * descentDirection[0] < 0.0 || g[1] + lambda * descentDirection[1] < 0.0)
			{
				descentDirection[0] = y[0];
				descentDirection[1] = y[1];

				lambda = 1.0 / max(fabs(descentDirection[0]), fabs(descentDirection[1]));
			}
		}
		else
		{
			//printf("Hessian is singular or not positive definite!\n");
			descentDirection[0] = y[0];
			descentDirection[1] = y[1];

			lambda = 1.0 / max(fabs(descentDirection[0]), fabs(descentDirection[1]));
		}

		double newError = curError + 1.0;
		for (int m = 0; m < maxSubIter; m++)
		{
			// Update solution
			double g_new[2];
			g_new[0] = g[0] + lambda * descentDirection[0];
			g_new[1] = g[1] + lambda * descentDirection[1];
			g_new[0] = max(g_new[0], 0.0);
			g_new[1] = max(g_new[1], 0.0);

			// Accept or reject new solution?
			double* polyTrans_new = calcPolyTrans(g_new);
			newError = dualEnergyDecomposition_cost(g_meas, g_new, polyTrans_new);
			if (newError < curError)
			{
				g[0] = g_new[0];
				g[1] = g_new[1];
				for (int i = 0; i < N_gamma; i++)
					polyTrans[i] = polyTrans_new[i];
				numUpdates += 1;
				free(polyTrans_new);
				break;
			}
			else
				lambda *= dampingParameter;
			free(polyTrans_new);
		}
		if (newError < tol || newError >= curError)
		{
			if (newError < curError)
				curError = newError;
			break;
		}
		else
			curError = newError;
	}
	free(polyTrans);

	if (curError > startError)
	{
		//printf("doing nothing (%f to %f after %d iterations)!\n", startError, curError, n);
		g[0] = g_1_init;
		g[1] = g_2_init;
		curError = startError;
	}

	g_mono_L = g[0];
	g_mono_H = g[1];
	theError = curError;

	return true;
}

double* dualEnergyDecomposition::calcPolyTrans(double* g)
{
	// g are estimates of basis coefficients
	double* polyTrans = (double*)calloc(size_t(N_gamma), sizeof(double));
	double maxExponent = 10.0;
	int gamma_ind_min = max(0, int(gamma_inv(1.0)));
	for (int i = gamma_ind_min; i < N_gamma; i++)
	{
		double curExponent = -(b_L[i] * g[0] + b_H[i] * g[1]);
		if (curExponent > maxExponent)
			polyTrans[i] = exp(maxExponent);
		else
			polyTrans[i] = exp(curExponent);
	}
	return polyTrans;
}

double dualEnergyDecomposition::dualEnergyDecomposition_cost(double* g, double* g_est, double* polyTrans)
{
	// g are the measured spectra
	double model_minus_meas[2]; model_minus_meas[0] = 0.0; model_minus_meas[1] = 0.0;
	int gamma_ind_min = max(0, int(gamma_inv(1.0)));
	for (int i = gamma_ind_min; i < N_gamma; i++)
	{
		model_minus_meas[0] += d_L[i] * polyTrans[i];
		model_minus_meas[1] += d_H[i] * polyTrans[i];
	}
	for (int i = 0; i < 2; i++)
		model_minus_meas[i] = -log(model_minus_meas[i]) - g[i];
	return 0.5 * (model_minus_meas[0] * model_minus_meas[0] + model_minus_meas[1] * model_minus_meas[1]);// +penaltyCost(g_est);
}

bool dualEnergyDecomposition::dualEnergyDecomposition_gradientAndHessian(double* g, double* g_est, double* polyTrans, double* grad, double* H)
{
	// cost function: 0.5*{ [g[0] + log(sum(d_L*polyTrans))]^2+ [g[1] + log(sum(d_H*polyTrans))]^2 }
	// so g is the set of measured spectra
	int gamma_ind_min = max(0, int(gamma_inv(1.0)));
	double monoTrans[2];
	double monoTransDiff1[2];
	double monoTransDiff2[2];

	double monoTransDiff11[2];
	double monoTransDiff22[2];
	double monoTransDiff12[2];

	for (int i = 0; i < 2; i++)
	{
		grad[i] = 0.0;

		monoTrans[i] = 0.0;
		monoTransDiff1[i] = 0.0;
		monoTransDiff2[i] = 0.0;

		monoTransDiff11[i] = 0.0;
		monoTransDiff22[i] = 0.0;
		monoTransDiff12[i] = 0.0;
	}
	H[0] = 0.0; H[1] = 0.0; H[2] = 0.0; H[3] = 0.0;

	for (int i = gamma_ind_min; i < N_gamma; i++)
	{
		monoTrans[0] += d_L[i] * polyTrans[i];
		monoTrans[1] += d_H[i] * polyTrans[i];

		// First Order Derivatives
		monoTransDiff1[0] += d_L[i] * polyTrans[i] * b_L[i];
		monoTransDiff1[1] += d_H[i] * polyTrans[i] * b_L[i];

		monoTransDiff2[0] += d_L[i] * polyTrans[i] * b_H[i];
		monoTransDiff2[1] += d_H[i] * polyTrans[i] * b_H[i];

		// Second Order Derivatives
		monoTransDiff11[0] += d_L[i] * polyTrans[i] * b_L[i] * b_L[i];
		monoTransDiff11[1] += d_H[i] * polyTrans[i] * b_L[i] * b_L[i];

		monoTransDiff22[0] += d_L[i] * polyTrans[i] * b_H[i] * b_H[i];
		monoTransDiff22[1] += d_H[i] * polyTrans[i] * b_H[i] * b_H[i];

		// Second Order Mixed Derivatives
		monoTransDiff12[0] += d_L[i] * polyTrans[i] * b_L[i] * b_H[i];
		monoTransDiff12[1] += d_H[i] * polyTrans[i] * b_L[i] * b_H[i];
	}
	for (int i = 0; i < 2; i++)
	{
		monoTransDiff1[i] *= -1.0;
		monoTransDiff2[i] *= -1.0;
	}

	for (int i = 0; i < 2; i++)
	{
		double model_minus_meas = -(log(monoTrans[i]) + g[i]);

		// First order derivatives of model
		double modelDiff1 = -monoTransDiff1[i] / monoTrans[i];
		double modelDiff2 = -monoTransDiff2[i] / monoTrans[i];

		// Second order derivatives of model
		double modelDiff11 = -(monoTransDiff11[i] * monoTrans[i] - monoTransDiff1[i] * monoTransDiff1[i]) / (monoTrans[i] * monoTrans[i]);
		double modelDiff22 = -(monoTransDiff22[i] * monoTrans[i] - monoTransDiff2[i] * monoTransDiff2[i]) / (monoTrans[i] * monoTrans[i]);

		double modelDiff12 = -(monoTransDiff12[i] * monoTrans[i] - monoTransDiff1[i] * monoTransDiff2[i]) / (monoTrans[i] * monoTrans[i]);

		grad[0] += model_minus_meas * modelDiff1;
		grad[1] += model_minus_meas * modelDiff2;

		H[0] += modelDiff1 * modelDiff1 + model_minus_meas * modelDiff11;
		H[3] += modelDiff2 * modelDiff2 + model_minus_meas * modelDiff22;

		H[1] += modelDiff1 * modelDiff2 + model_minus_meas * modelDiff12;
	}
	H[2] = H[1];

	return true;
}
