#ifdef WIN32
    #pragma once

    #ifdef PROJECTOR_EXPORTS
        #define XRAYPHYSICS_API __declspec(dllexport)
    #else
        #define XRAYPHYSICS_API __declspec(dllimport)
    #endif
#else
    #define XRAYPHYSICS_API
#endif

//extern "C" XRAYPHYSICS_API void about();

extern "C" XRAYPHYSICS_API bool initialize();

// Force-loads all cross-section and scatter-distribution tables in a single (single-threaded) call.
// Call this once before any multi-threaded / multi-GPU work to avoid lazy-initialization races.
extern "C" XRAYPHYSICS_API bool initializeXrayPhysicsTables();

extern "C" XRAYPHYSICS_API float atomicMass(int Z);
extern "C" XRAYPHYSICS_API float massDensity(int Z);
extern "C" XRAYPHYSICS_API int elementSymbolToAtomicNumber(const char* chemForm);

extern "C" XRAYPHYSICS_API bool simulateSpectra(float kVp, float takeOffAngle, int Z, float* gammas, int N, float* output);
extern "C" XRAYPHYSICS_API bool changeTakeOffAngle(float kVp, float takeOffAngle_cur, float takeOffAngle_new, int Z, float* gammas, int N, float* s);

extern "C" XRAYPHYSICS_API float sigma(float Z, float gamma);
extern "C" XRAYPHYSICS_API float sigmaCompound(const char* chemForm, float gamma);

extern "C" XRAYPHYSICS_API float sigmae(float Z, float gamma);
extern "C" XRAYPHYSICS_API float sigmaeCompound(const char* chemForm, float gamma);

extern "C" XRAYPHYSICS_API float sigmaPE(float Z, float gamma);
extern "C" XRAYPHYSICS_API float sigmaCompoundPE(const char* chemForm, float gamma);

extern "C" XRAYPHYSICS_API float sigmaCS(float Z, float gamma);
extern "C" XRAYPHYSICS_API float sigmaCompoundCS(const char* chemForm, float gamma);

extern "C" XRAYPHYSICS_API float sigmaRS(float Z, float gamma);
extern "C" XRAYPHYSICS_API float sigmaCompoundRS(const char* chemForm, float gamma);

extern "C" XRAYPHYSICS_API float sigmaPP(float Z, float gamma);
extern "C" XRAYPHYSICS_API float sigmaCompoundPP(const char* chemForm, float gamma);

extern "C" XRAYPHYSICS_API float sigmaTP(float Z, float gamma);
extern "C" XRAYPHYSICS_API float sigmaCompoundTP(const char* chemForm, float gamma);

extern "C" XRAYPHYSICS_API float incoherentScatterDistribution(float Z, float gamma, float theta);
extern "C" XRAYPHYSICS_API float coherentScatterDistribution(float Z, float gamma, float theta);

extern "C" XRAYPHYSICS_API float incoherentScatterDistributionCompound(const char* chemForm, float gamma, float theta);
extern "C" XRAYPHYSICS_API float coherentScatterDistributionCompound(const char* chemForm, float gamma, float theta);

extern "C" XRAYPHYSICS_API float incoherentScatterDistribution_normalizationFactor(float Z, float gamma);
extern "C" XRAYPHYSICS_API float coherentScatterDistribution_normalizationFactor(float Z, float gamma);

extern "C" XRAYPHYSICS_API float incoherentScatterDistributionCompound_normalizationFactor(const char* chemForm, float gamma);
extern "C" XRAYPHYSICS_API float coherentScatterDistributionCompound_normalizationFactor(const char* chemForm, float gamma);

extern "C" XRAYPHYSICS_API float meanEnergy(float* spectralResponse, float* gammas, int N);
extern "C" XRAYPHYSICS_API bool normalizeSpectrum(float* spectralResponse, float* gammas, int N);

extern "C" XRAYPHYSICS_API float effectiveZ(const char* chemForm, float min_energy, float max_energy, float arealDensity);

extern "C" XRAYPHYSICS_API float effectiveAttenuation(float Ze, float density, float thickness, float* spectralResponse, float* gammas, int N);
extern "C" XRAYPHYSICS_API float effectiveEnergy(float Ze, float density, float thickness, float* spectralResponse, float* gammas, int N);

extern "C" XRAYPHYSICS_API float effectiveAttenuation_compound(const char* chemForm, float density, float thickness, float* spectralResponse, float* gammas, int N);
extern "C" XRAYPHYSICS_API float effectiveEnergy_compound(const char* chemForm, float density, float thickness, float* spectralResponse, float* gammas, int N);

extern "C" XRAYPHYSICS_API float transmission(float Z, float density, float thickness, float* spectralResponse, float* gammas, int N);
extern "C" XRAYPHYSICS_API float transmission_compound(const char* chemForm, float density, float thickness, float* spectralResponse, float* gammas, int N);

extern "C" XRAYPHYSICS_API bool setBHlookupTable(float Ze, float* spectralResponse, float* gammas, int N_gamma, float* LUT, float T_lac, int N_lac, float referenceEnergy);
extern "C" XRAYPHYSICS_API bool setBHlookupTable_compound(const char* chemForm, float* spectralResponse, float* gammas, int N_gamma, float* LUT, float T_lac, int N_lac, float referenceEnergy);

extern "C" XRAYPHYSICS_API bool setBHClookupTable(float Ze, float* spectralResponse, float* gammas, int N_gamma, float* LUT, float T_lac, int N_lac, float referenceEnergy);
extern "C" XRAYPHYSICS_API bool setBHClookupTable_compound(const char* chemForm, float* spectralResponse, float* gammas, int N_gamma, float* LUT, float T_lac, int N_lac, float referenceEnergy);

extern "C" XRAYPHYSICS_API bool generateDEDlookUpTables(float* spectralResponses, float* gammas, int N_gamma, float* referenceEnergies, float* basisFunctions, float* LUT, float T_lac, int N_lac);

extern "C" XRAYPHYSICS_API bool setTwoMaterialBHClookupTable(float* spectralResponse, float* gammas, int N_gamma, float referenceEnergy, float* sigmas, float* LUT, float T_lac, int N_lac, bool usingBasis);
extern "C" XRAYPHYSICS_API bool setThreeMaterialBHClookupTable(float* spectralResponse, float* gammas, int N_gamma, float referenceEnergy, float* sigmas, float* LUT, float T_lac, int N_lac);

extern "C" XRAYPHYSICS_API bool mono_to_poly_transmission(float* LUT, float* spectralResponse, float* b_L, float* b_H, int N_gamma, float* t_mono, int N_mono, float* frac, int N_frac);
extern "C" XRAYPHYSICS_API bool poly_to_mono_transmission(float* LUT, float* spectralResponse, float* b_L, float* b_H, int N_gamma, float* t_poly, int N_poly, float* frac, int N_frac);
