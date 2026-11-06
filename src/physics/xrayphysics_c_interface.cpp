
#include "xrayphysics_c_interface.h"
//#include "xsec.h"
//#include "xsource.h"
#include "xrayphysics.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

XrayPhysics physics;
//xsecTables.init();

/*
void about()
{
    physics.about();
}
//*/

bool initialize()
{
    float x = sigma(1.0, 1.0);
	float y = incoherentScatterDistribution(1.0, 1.0, 1.0);
    if (isnan(x) || isnan(y) || x <= 0.0 || y <= 0.0)
        return false;
    else
        return true;
}

bool initializeXrayPhysicsTables()
{
    return physics.initializeTables();
}

float atomicMass(int Z)
{
    return physics.atomicMass(Z);
}

float massDensity(int Z)
{
    return physics.xsecTables.getMassDensity(Z);
}

int elementSymbolToAtomicNumber(const char* chemForm)
{
    string chemForm_str = chemForm;
    return physics.xsecTables.elementStringToAtomicNumber(chemForm_str);
}

bool simulateSpectra(float kVp, float takeOffAngle, int Z, float* gammas, int N, float* output)
{
    return physics.simulateSpectra(kVp, takeOffAngle, Z, gammas, N, output);
}

bool changeTakeOffAngle(float kVp, float takeOffAngle_cur, float takeOffAngle_new, int Z, float* gammas, int N, float* s)
{
    return physics.changeTakeOffAngle(kVp, takeOffAngle_cur, takeOffAngle_new, Z, gammas, N, s);
}

float sigma(float Z, float gamma)
{
    return physics.xsecTables.sigma(Z, gamma);
}

float sigmaCompound(const char* chemForm, float gamma)
{
    return physics.xsecTables.sigma(chemForm, gamma);
}

float sigmae(float Z, float gamma)
{
    return physics.xsecTables.sigma_e(Z, gamma);
}

float sigmaeCompound(const char* chemForm, float gamma)
{
    return physics.xsecTables.sigma_e(chemForm, gamma);
}

float sigmaPE(float Z, float gamma)
{
    return physics.xsecTables.sigmaPE(Z, gamma);
}

float sigmaCompoundPE(const char* chemForm, float gamma)
{
    return physics.xsecTables.sigmaPE(chemForm, gamma);
}

float sigmaCS(float Z, float gamma)
{
    return physics.xsecTables.sigmaCS(Z, gamma);
}

float sigmaCompoundCS(const char* chemForm, float gamma)
{
    return physics.xsecTables.sigmaCS(chemForm, gamma);
}

float sigmaRS(float Z, float gamma)
{
    return physics.xsecTables.sigmaRS(Z, gamma);
}

float sigmaCompoundRS(const char* chemForm, float gamma)
{
    return physics.xsecTables.sigmaRS(chemForm, gamma);
}

float sigmaPP(float Z, float gamma)
{
    return physics.xsecTables.sigmaPP(Z, gamma);
}

float sigmaCompoundPP(const char* chemForm, float gamma)
{
    return physics.xsecTables.sigmaPP(chemForm, gamma);
}

float sigmaTP(float Z, float gamma)
{
    return physics.xsecTables.sigmaTP(Z, gamma);
}

float sigmaCompoundTP(const char* chemForm, float gamma)
{
    return physics.xsecTables.sigmaTP(chemForm, gamma);
}

float incoherentScatterDistribution(float Z, float gamma, float theta)
{
    return physics.incoherentScatterDistribution(Z, gamma, theta);
}

float coherentScatterDistribution(float Z, float gamma, float theta)
{
    return physics.coherentScatterDistribution(Z, gamma, theta);
}

float incoherentScatterDistributionCompound(const char* chemForm, float gamma, float theta)
{
    return physics.incoherentScatterDistribution(chemForm, gamma, theta);
}

float coherentScatterDistributionCompound(const char* chemForm, float gamma, float theta)
{
    return physics.coherentScatterDistribution(chemForm, gamma, theta);
}

float incoherentScatterDistribution_normalizationFactor(float Z, float gamma)
{
    return physics.incoherentScatterDistribution_normalizationFactor(Z, gamma);
}

float coherentScatterDistribution_normalizationFactor(float Z, float gamma)
{
    return physics.coherentScatterDistribution_normalizationFactor(Z, gamma);
}

float incoherentScatterDistributionCompound_normalizationFactor(const char* chemForm, float gamma)
{
    return physics.incoherentScatterDistribution_normalizationFactor(chemForm, gamma);
}

float coherentScatterDistributionCompound_normalizationFactor(const char* chemForm, float gamma)
{
    return physics.coherentScatterDistribution_normalizationFactor(chemForm, gamma);
}

float meanEnergy(float* spectralResponse, float* gammas, int N)
{
    return physics.meanEnergy(spectralResponse, gammas, N);
}

bool normalizeSpectrum(float* spectralResponse, float* gammas, int N)
{
    return physics.normalizeSpectrum(spectralResponse, gammas, N);
}

float effectiveZ(const char* chemForm, float min_energy, float max_energy, float arealDensity)
{
    return physics.effectiveZ(chemForm, min_energy, max_energy, arealDensity);
}

float effectiveAttenuation(float Ze, float density, float thickness, float* spectralResponse, float* gammas, int N)
{
    return physics.effectiveAttenuation(Ze, density, thickness, spectralResponse, gammas, N);
}

float effectiveEnergy(float Ze, float density, float thickness, float* spectralResponse, float* gammas, int N)
{
    return physics.effectiveEnergy(Ze, density, thickness, spectralResponse, gammas, N);
}

float effectiveAttenuation_compound(const char* chemForm, float density, float thickness, float* spectralResponse, float* gammas, int N)
{
    return physics.effectiveAttenuation(chemForm, density, thickness, spectralResponse, gammas, N);
}

float effectiveEnergy_compound(const char* chemForm, float density, float thickness, float* spectralResponse, float* gammas, int N)
{
    return physics.effectiveEnergy(chemForm, density, thickness, spectralResponse, gammas, N);
}

float transmission(float Z, float density, float thickness, float* spectralResponse, float* gammas, int N)
{
    return physics.transmission(Z, density, thickness, spectralResponse, gammas, N);
}

float transmission_compound(const char* chemForm, float density, float thickness, float* spectralResponse, float* gammas, int N)
{
    return physics.transmission(chemForm, density, thickness, spectralResponse, gammas, N);
}

bool setBHlookupTable(float Ze, float* spectralResponse, float* gammas, int N_gamma, float* LUT, float T_lac, int N_lac, float referenceEnergy)
{
    return physics.setBHlookupTable(Ze, spectralResponse, gammas, N_gamma, LUT, T_lac, N_lac, referenceEnergy);
}

bool setBHlookupTable_compound(const char* chemForm, float* spectralResponse, float* gammas, int N_gamma, float* LUT, float T_lac, int N_lac, float referenceEnergy)
{
    return physics.setBHlookupTable(chemForm, spectralResponse, gammas, N_gamma, LUT, T_lac, N_lac, referenceEnergy);
}

bool setBHClookupTable(float Ze, float* spectralResponse, float* gammas, int N_gamma, float* LUT, float T_lac, int N_lac, float referenceEnergy)
{
    return physics.setBHClookupTable(Ze, spectralResponse, gammas, N_gamma, LUT, T_lac, N_lac, referenceEnergy);
}

bool setBHClookupTable_compound(const char* chemForm, float* spectralResponse, float* gammas, int N_gamma, float* LUT, float T_lac, int N_lac, float referenceEnergy)
{
    return physics.setBHClookupTable(chemForm, spectralResponse, gammas, N_gamma, LUT, T_lac, N_lac, referenceEnergy);
}

bool generateDEDlookUpTables(float* spectralResponses, float* gammas, int N_gamma, float* referenceEnergies, float* basisFunctions, float* LUT, float T_lac, int N_lac)
{
    return physics.generateDEDlookUpTables(spectralResponses, gammas, N_gamma, referenceEnergies, basisFunctions, LUT, T_lac, N_lac);
}

bool setTwoMaterialBHClookupTable(float* spectralResponse, float* gammas, int N_gamma, float referenceEnergy, float* sigmas, float* LUT, float T_lac, int N_lac, bool usingBasis)
{
    return physics.setTwoMaterialBHClookupTable(spectralResponse, gammas, N_gamma, referenceEnergy, sigmas, LUT, T_lac, N_lac, usingBasis);
}

bool setThreeMaterialBHClookupTable(float* spectralResponse, float* gammas, int N_gamma, float referenceEnergy, float* sigmas, float* LUT, float T_lac, int N_lac)
{
    return physics.setThreeMaterialBHClookupTable(spectralResponse, gammas, N_gamma, referenceEnergy, sigmas, LUT, T_lac, N_lac);
}

bool mono_to_poly_transmission(float* LUT, float* spectralResponse, float* b_L, float* b_H, int N_gamma, float* t_mono, int N_mono, float* frac, int N_frac)
{
    if (LUT == nullptr || b_L == nullptr || b_H == nullptr || N_gamma <= 0 || t_mono == nullptr || N_mono <= 0 || frac == nullptr || N_frac <= 0)
        return false;
 
    omp_set_num_threads(omp_get_num_procs());
    #pragma omp parallel for
    for (int ifrac = 0; ifrac < N_frac; ifrac++)
    {
        float* sigma = new float[N_gamma];
        for (int l = 0; l < N_gamma; l++)
            sigma[l] = b_L[l] + frac[ifrac] * b_H[l];
        for (int i = 0; i < N_mono; i++)
        {
            float a_mono = -log(t_mono[i]);
            float t_poly = 0.0;
            for (int l = 0; l < N_gamma; l++)
                t_poly += spectralResponse[l] * exp(-sigma[l] * a_mono);
            LUT[ifrac * N_mono + i] = t_poly;
        }
        delete [] sigma;
    }
    return true;
}

bool poly_to_mono_transmission(float* LUT, float* spectralResponse, float* b_L, float* b_H, int N_gamma, float* t_poly, int N_poly, float* frac, int N_frac)
{
    if (LUT == nullptr || b_L == nullptr || b_H == nullptr || N_gamma <= 0 || t_poly == nullptr || N_poly <= 0 || frac == nullptr || N_frac <= 0)
        return false;

    int N_iter = 10;
    double tol = 1.0e-7;

    double* d = new double[N_gamma];
    for (int l = 0; l < N_gamma; l++)
        d[l] = spectralResponse[l];

    omp_set_num_threads(omp_get_num_procs());
    #pragma omp parallel for schedule(dynamic)
    for (int ifrac = 0; ifrac < N_frac; ifrac++)
    {
        double* sigma = new double[N_gamma];
        for (int l = 0; l < N_gamma; l++)
            sigma[l] = b_L[l] + frac[ifrac] * b_H[l];
        for (int i = 0; i < N_poly; i++)
        {
            double polyAtten = -log(t_poly[i]);
            double monoAtten = polyAtten;
            if (i > 0)
                monoAtten = -log(LUT[ifrac * N_poly + i-1]);
            double monoAtten_save = monoAtten;
            if (physics.BHCkernel(monoAtten, polyAtten, sigma, d, N_gamma) == false)
                monoAtten = monoAtten_save;
            LUT[ifrac * N_poly + i] = exp(-monoAtten);
        }
        delete [] sigma;
    }
    delete [] d;

    /*
    for (int j = 0; j < N_atten; j++)
    {
        double frac = double(j) * T_frac;
        double* normalizedCrossSection = new double[N_gamma];
        for (int l = 0; l < N_gamma; l++)
        {
            if (usingBasis)
                normalizedCrossSection[l] = sigma_hat_1[l] + frac * sigma_hat_2[l];
            else
                normalizedCrossSection[l] = (1.0 - frac) * sigma_hat_1[l] + frac * sigma_hat_2[l];
        }
        for (int i = 0; i < N_atten; i++)
        {
            double polyAtten = double(i) * T_atten;
            double monoAtten = polyAtten;
            if (i > 0)
                monoAtten = LUT[j * N_atten + i-1];
            double monoAtten_save = monoAtten;
            if (BHCkernel(monoAtten, polyAtten, normalizedCrossSection, d, N_gamma) == false)
                monoAtten = monoAtten_save;
            if (monoAtten < 0.0 || monoAtten < monoAtten_save || (monoAtten-monoAtten_save)/T_atten > 10000.0)
                monoAtten = monoAtten_save;
            LUT[j * N_atten + i] = monoAtten;
        }
        delete[] normalizedCrossSection;
    }
    //*/
    return true;
}
