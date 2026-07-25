#ifndef XRAYPHYSICS_H
#define XRAYPHYSICS_H

#ifdef WIN32
#pragma once
#endif

#define XRAY_PHYSICS_VERSION "1.2"

#include "xsec.h"
#include "xscatter.h"
#include "xsource.h"

class XrayPhysics
{
public:
    XrayPhysics();
    ~XrayPhysics();

    //const char* about();

    // Force-loads the cross-section and scatter-distribution tables. These tables otherwise use a
    // lazy, non-thread-safe initialization on first use; call this once (single-threaded) before
    // any multi-threaded / multi-GPU work to avoid initialization races.
    bool initializeTables();

    float atomicMass(int Z);

    bool simulateSpectra(float kVp, float takeOffAngle, int Z, float* gammas, int N, float* output);
    bool changeTakeOffAngle(float kVp, float takeOffAngle_cur, float takeOffAngle_new, int Z, float* gammas, int N, float* s);

    float meanEnergy(float* spectralResponse, float* gammas, int N);
    bool normalizeSpectrum(float* spectralResponse, float* gammas, int N);

    float effectiveAttenuation(float Z, float density, float thickness, float* spectralResponse, float* gammas, int N);
    float effectiveEnergy(float Z, float density, float thickness, float* spectralResponse, float* gammas, int N);

    float effectiveAttenuation(const char* chemForm, float density, float thickness, float* spectralResponse, float* gammas, int N);
    float effectiveEnergy(const char* chemForm, float density, float thickness, float* spectralResponse, float* gammas, int N);

    float transmission(float Z, float density, float thickness, float* spectralResponse, float* gammas, int N);
    float transmission(const char* chemForm, float density, float thickness, float* spectralResponse, float* gammas, int N);

    float effectiveZ(const char* chemForm, float min_energy, float max_energy, float arealDensity = 0.0);

    float incoherentScatterDistribution(float Z, float gamma, float theta);
    float coherentScatterDistribution(float Z, float gamma, float theta);

    float incoherentScatterDistribution(const char* chemForm, float gamma, float theta);
    float coherentScatterDistribution(const char* chemForm, float gamma, float theta);

    float incoherentScatterDistribution_normalizationFactor(float Z, float gamma);
    float coherentScatterDistribution_normalizationFactor(float Z, float gamma);

    float incoherentScatterDistribution_normalizationFactor(const char* chemForm, float gamma);
    float coherentScatterDistribution_normalizationFactor(const char* chemForm, float gamma);

    bool setBHlookupTable(float Ze, float* spectralResponse, float* gammas, int N_gamma, float* LUT, float T_lac, int N_lac, float referenceEnergy);
    bool setBHlookupTable(const char* chemForm, float* spectralResponse, float* gammas, int N_gamma, float* LUT, float T_lac, int N_lac, float referenceEnergy);

    bool setBHClookupTable(float Ze, float* spectralResponse, float* gammas, int N_gamma, float* LUT, float T_lac, int N_lac, float referenceEnergy);
    bool setBHClookupTable(const char* chemForm, float* spectralResponse, float* gammas, int N_gamma, float* LUT, float T_lac, int N_lac, float referenceEnergy);

    bool generateDEDlookUpTables(float* spectralResponses, float* gammas, int N_gamma, float* referenceEnergies, float* basisFunctions, float* LUT, float T_lac, int N_lac);

    bool setTwoMaterialBHClookupTable(float* spectralResponse, float* gammas, int N_gamma, float referenceEnergy, float* sigmas, float* LUT, float T_atten, int N_atten, bool usingBasis);
    bool setThreeMaterialBHClookupTable(float* spectralResponse, float* gammas, int N_gamma, float referenceEnergy, float* sigmas, float* LUT, float T_atten, int N_atten);

    /**
 	 * \fn          polychromatic_attenuation
	 * \brief       calculates the polychromatic attenuation from 1, 2 or 3 monochromatic attenuations
	 * \param[in]   spectralResponse: pointer to the total system spectral response
	 * \param[in]   gammas: the energy samples (keV)
	 * \param[in]   N_gamma: number of energy samples
	 * \param[in]   g_1: monochromatic attenuation of 1st material
	 * \param[in]	sigma_1: mass cross section of 1st material
	 * \param[in]   sigma_1_ref: mass cross section at reference energy of 1st material
	 * \param[in]   g_2: monochromatic attenuation of 2nd material
	 * \param[in]	sigma_2: mass cross section of 2nd material
	 * \param[in]   sigma_2_ref: mass cross section at reference energy of 2nd material
	 * \param[in]   g_3: monochromatic attenuation of 3rd material
	 * \param[in]	sigma_3: mass cross section of 3rd material
	 * \param[in]   sigma_3_ref: mass cross section at reference energy of 3rd material
	 * \return      the polychromatic attenuation
	 */
	float polychromatic_attenuation(float* spectralResponse, float* gammas, int N_gamma, float g_1, float* sigma_1, float sigma_1_ref, float g_2 = 0.0, float* sigma_2 = nullptr, float sigma_2_ref = 1.0, float g_3 = 0.0, float* sigma_3 = nullptr, float sigma_3_ref = 1.0);

    xsec xsecTables;
    xscatter xscatterTables;
    xraySource XraySourceModel;

    bool BHCkernel(double& monoAtten, double polyAtten, double* normalizedCrossSection, double* d, int N_gamma);
private:

    bool setBHlookupTable_helper(double* sigma_hat, float* spectralResponse, float* gammas, int N_gamma, float* LUT, float T_lac, int N_lac, float referenceEnergy);
    bool setBHClookupTable_helper(double* sigma_hat, float* spectralResponse, float* gammas, int N_gamma, float* LUT, float T_lac, int N_lac, float referenceEnergy);

    float gamma_inv(float val, float* gammas, int N_gamma);
};

//BHC and BH
//pBHC

#endif
