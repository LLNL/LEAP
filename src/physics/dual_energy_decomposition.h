#ifndef DUAL_ENERGY_DECOMPOSITION_H
#define DUAL_ENERGY_DECOMPOSITION_H

#ifdef WIN32
#pragma once
#endif

#include "xsec.h"
#include "xsource.h"

class dualEnergyDecomposition
{
public:
    dualEnergyDecomposition();
    ~dualEnergyDecomposition();
    bool clearAll();

    //bool setBHlookupTable(float Ze, float* spectralResponse, float* gammas, int N_gamma, float* LUT, float T_lac, int N_lac, float referenceEnergy);
    //bool generateDEDlookUpTables(ParameterSet*, sct*, sct*, char* args = NULL);
    bool generateDEDlookUpTables(float* spectralResponses, float* gammas, int N_gamma, float* referenceEnergies, float* basisFunctions, float* LUT, float T_lac, int N_lac);
    //float* spectralResponses, float* gammas, int N_gamma
    //float* referenceEnergies
    //float* basisFunctions
    //float* LUT, float T_lac, int N_lac
private:

    // normalized low and high energy spectra
    double* d_L;
    double* d_H;

    // low and high energy basis functions
    double* b_L;
    double* b_H;

    // low and high reference energies
    double ref_L;
    double ref_H;

    double gamma(int i);
    double gamma_inv(double);

    int N_gamma;
    double* gammas;

    double T_gamma(int);

    bool decompose(double g_L, double g_H, double g_init_L, double g_init_H, double& g_mono_L, double& g_mono_H, double& theError);

    bool inverse2x2(double* A);
    bool positiveDefinite2x2(double* A);

    double* calcPolyTrans(double* g);
    double dualEnergyDecomposition_cost(double* g, double* g_est, double* polyTrans);
    bool dualEnergyDecomposition_gradientAndHessian(double* g, double* g_est, double* polyTrans, double* grad, double* H);
};

#endif
