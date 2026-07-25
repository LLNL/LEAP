#include "xrayphysics.h"
#include "dual_energy_decomposition.h"
#include <string>
#include <math.h>

#ifdef USE_OPENMP
#include <omp.h>
#endif

#ifndef PI
#define PI 3.1415926535897932385
#endif

XrayPhysics::XrayPhysics()
{
    XraySourceModel.init(&xsecTables);
}

XrayPhysics::~XrayPhysics()
{
}

/*
const char* XrayPhysics::about()
{
    printf("************************************\n");
    printf("        XrayPhysics\n");
    printf("        version %s\n", XRAY_PHYSICS_VERSION);
    printf("      Based on EPDL97\n");
    //printf("\n");
    printf("compiled: %s %s\n", __DATE__, __TIME__);
    printf("  written by: Kyle Champley\n");
    printf("************************************\n");

    return XRAY_PHYSICS_VERSION;
}
//*/

bool XrayPhysics::initializeTables()
{
    // xsecTables.init() is internally guarded (no-op if already loaded); xscatterTables.init()
    // rebuilds unconditionally, so guard it with isInitialized() to keep this idempotent.
    bool ok = xsecTables.init();
    if (xscatterTables.isInitialized() == false)
        ok = xscatterTables.init() && ok;
    return ok;
}

float XrayPhysics::atomicMass(int Z)
{
    return xsecTables.getAtomicMass(Z);
}

bool XrayPhysics::simulateSpectra(float kVp, float takeOffAngle, int Z, float* gammas, int N, float* output)
{
    float* s = XraySourceModel.simulateSpectra(kVp, takeOffAngle, Z, gammas, N);
    if (s != NULL)
    {
        for (int i = 0; i < N; i++)
            output[i] = s[i];
        free(s);
        return true;
    }
    else
        return false;
}

bool XrayPhysics::changeTakeOffAngle(float kVp, float takeOffAngle_cur, float takeOffAngle_new, int Z, float* gammas, int N, float* s)
{
    float* factors = XraySourceModel.takeOffAngleConversionFactor(kVp, takeOffAngle_cur, takeOffAngle_new, Z, gammas, N);
    if (factors != NULL)
    {
        for (int i = 0; i < N; i++)
            s[i] *= factors[i];
        free(factors);
        return true;
    }
    else
        return false;

}

float XrayPhysics::gamma_inv(float val, float* gammas, int N_gamma)
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

float XrayPhysics::meanEnergy(float* spectralResponse, float* gammas, int N)
{
    if (spectralResponse == NULL || gammas == NULL || N <= 0)
        return 0.0;
    if (N == 1)
        return gammas[0];
    double retVal = 0.0;
    double accum = 0.0;
    for (int i = 0; i < N; i++)
    {
        double T_phi;
        if (i == 0)
            T_phi = gammas[i + 1] - gammas[i];
        else if (i == N - 1)
            T_phi = gammas[i] - gammas[i - 1];
        else
            T_phi = 0.5 * (gammas[i + 1] - gammas[i - 1]);

        retVal += gammas[i] * T_phi * spectralResponse[i];
        accum += T_phi * spectralResponse[i];
    }
    return float(retVal / accum);
}

bool XrayPhysics::normalizeSpectrum(float* spectralResponse, float* gammas, int N)
{
    if (spectralResponse == NULL || gammas == NULL || N <= 0)
        return false;
    if (N == 1)
    {
        spectralResponse[0] = 1.0;
    }
    else
    {
        double accum = 0.0;
        for (int i = 0; i < N; i++)
        {
            float T_phi;
            if (i == 0)
                T_phi = gammas[i + 1] - gammas[i];
            else if (i == N - 1)
                T_phi = gammas[i] - gammas[i - 1];
            else
                T_phi = 0.5 * (gammas[i + 1] - gammas[i - 1]);

            accum += T_phi * spectralResponse[i];
        }
        for (int i = 0; i < N; i++)
            spectralResponse[i] = spectralResponse[i] / accum;
    }
    return true;
}

float XrayPhysics::effectiveAttenuation(float Z, float density, float thickness, float* spectralResponse, float* gammas, int N)
{
    if (spectralResponse == NULL || gammas == NULL || N <= 0)
        return 0.0;
    if (N == 1)
        return density*xsecTables.sigma(Z, gammas[0]);

    double retVal = 0.0;
    double accum = 0.0;
    for (int i = 0; i < N; i++)
    {
        float T_phi;
        if (i == 0)
            T_phi = gammas[i + 1] - gammas[i];
        else if (i == N - 1)
            T_phi = gammas[i] - gammas[i - 1];
        else
            T_phi = 0.5 * (gammas[i + 1] - gammas[i - 1]);

        if (thickness == 0.0)
        {
            retVal += T_phi * spectralResponse[i] * density * xsecTables.sigma(Z, gammas[i]);
        }
        else
        {
            retVal += T_phi * spectralResponse[i] * exp(-density * thickness * xsecTables.sigma(Z, gammas[i]));
        }
        accum += T_phi * spectralResponse[i];
    }
    retVal = retVal / accum;

    if (thickness != 0.0)
        retVal = -log(retVal) / thickness;
    return float(retVal);
}

float XrayPhysics::effectiveEnergy(float Z, float density, float thickness, float* spectralResponse, float* gammas, int N)
{
    float mu_eff = effectiveAttenuation(Z, density, thickness, spectralResponse, gammas, N);
    return xsecTables.sigma_inv(Z, mu_eff /density);
}

float XrayPhysics::effectiveAttenuation(const char* chemForm, float density, float thickness, float* spectralResponse, float* gammas, int N)
{
    if (spectralResponse == NULL || gammas == NULL || N <= 0)
        return 0.0;
    if (N == 1)
        return density * xsecTables.sigma(chemForm, gammas[0]);

    double retVal = 0.0;
    double accum = 0.0;
    for (int i = 0; i < N; i++)
    {
        float T_phi;
        if (i == 0)
            T_phi = gammas[i + 1] - gammas[i];
        else if (i == N - 1)
            T_phi = gammas[i] - gammas[i - 1];
        else
            T_phi = 0.5 * (gammas[i + 1] - gammas[i - 1]);

        if (thickness == 0.0)
        {
            retVal += T_phi * spectralResponse[i] * density * xsecTables.sigma(chemForm, gammas[i]);
        }
        else
        {
            retVal += T_phi * spectralResponse[i] * exp(-density * thickness * xsecTables.sigma(chemForm, gammas[i]));
        }
        accum += T_phi * spectralResponse[i];
    }
    retVal = retVal / accum;

    if (thickness != 0.0)
        retVal = -log(retVal) / thickness;
    return float(retVal);
}

float XrayPhysics::effectiveEnergy(const char* chemForm, float density, float thickness, float* spectralResponse, float* gammas, int N)
{
    float mu_eff = effectiveAttenuation(chemForm, density, thickness, spectralResponse, gammas, N);
    return xsecTables.sigma_inv(chemForm, mu_eff / density);
}

float XrayPhysics::effectiveZ(const char* chemForm, float min_energy, float max_energy, float arealDensity)
{
    if (chemForm == NULL)
        return 0.0;
    if (arealDensity <= 0.0)
        arealDensity = 2.5;
    float c = arealDensity * xsecTables.sigma(chemForm, max_energy) / xsecTables.sigma_e(chemForm, max_energy);

    //printf("arealElectronDensity = %f\n", c);

    float gamma_0 = floor(min_energy);
    int N_gamma = int(ceil(max_energy) - floor(min_energy)) + 1;
    float* gammas = new float[N_gamma];
    float* theWeights = new float[N_gamma];
    float* crossSection = new float[N_gamma];
    for (int i = 0; i < N_gamma; i++)
    {
        gammas[i] = float(i) + gamma_0;
        crossSection[i] = xsecTables.sigma_e(chemForm, gammas[i]);
        theWeights[i] = exp(-2.0 * c * crossSection[i]);
    }

    double target_dot_target = 0.0;
    for (int j = 0; j < N_gamma; j++)
        target_dot_target += crossSection[j] * crossSection[j] * theWeights[j];

    int currentZeRange[2];
    currentZeRange[0] = 1; currentZeRange[1] = 100;

    //printf("arealDensity = %f\n", arealDensity);
    //printf("energy range: %f to %f\n", gammas[0], gammas[N_gamma-1]);

    //##############################################################################
    float retVal = 1.0;
    double minError = 1e12;
    int minError_arg = currentZeRange[0];
    double target_dot_ref = 0.0;
    double ref_dot_ref = 0.0;
    for (int i = currentZeRange[0]; i <= currentZeRange[1]; i++)
    {
        target_dot_ref = 0.0;
        ref_dot_ref = 0.0;
        for (int j = 0; j < N_gamma; j++)
        {
            double temp = xsecTables.sigma_e(i, gammas[j]);
            target_dot_ref += crossSection[j] * temp * theWeights[j];
            ref_dot_ref += temp * temp * theWeights[j];
        }
        double curError = ref_dot_ref - 2.0 * target_dot_ref + target_dot_target;
        if (curError < minError)
        {
            minError = curError;
            minError_arg = i;
        }
    }

    if (minError_arg <= currentZeRange[0] || minError_arg >= currentZeRange[1])
    {
        currentZeRange[0] = 1; currentZeRange[1] = 100;
        retVal = float(minError_arg);
    }
    else
    {
        currentZeRange[0] = 1; currentZeRange[1] = 100;

        double sig1_minus_sig2_mag_sq = 0.0;
        double target_minus_sig2_dot_sig1_minus_sig2 = 0.0;
        for (int j = 0; j < N_gamma; j++)
        {
            double sig2 = xsecTables.sigma_e(minError_arg - 1, gammas[j]);
            double sig1 = xsecTables.sigma_e(minError_arg, gammas[j]);

            sig1_minus_sig2_mag_sq += (sig1 - sig2) * (sig1 - sig2) * theWeights[j];
            target_minus_sig2_dot_sig1_minus_sig2 += theWeights[j] * (crossSection[j] - sig2) * (sig1 - sig2);
        }
        double denom_low = sig1_minus_sig2_mag_sq;
        double x_low = target_minus_sig2_dot_sig1_minus_sig2 / denom_low;

        sig1_minus_sig2_mag_sq = 0.0;
        target_minus_sig2_dot_sig1_minus_sig2 = 0.0;
        for (int j = 0; j < N_gamma; j++)
        {
            double sig2 = xsecTables.sigma_e(minError_arg, gammas[j]);
            double sig1 = xsecTables.sigma_e(minError_arg + 1, gammas[j]);

            sig1_minus_sig2_mag_sq += (sig1 - sig2) * (sig1 - sig2) * theWeights[j];
            target_minus_sig2_dot_sig1_minus_sig2 += theWeights[j] * (crossSection[j] - sig2) * (sig1 - sig2);
        }
        double denom_high = sig1_minus_sig2_mag_sq;
        double x_high = target_minus_sig2_dot_sig1_minus_sig2 / denom_high;

        //printf("x_low = %f, x_high = %f\n", x_low, x_high);

        if (x_low >= 0.0 && x_low <= 1.0)
        {
            if (x_high >= 0.0 && x_high <= 1.0)
            {
                // both are valid, choose the best one
                float val_low = float(minError_arg - 1) + x_low;
                float val_high = float(minError_arg) + x_high;

                target_dot_ref = 0.0;
                ref_dot_ref = 0.0;
                for (int j = 0; j < N_gamma; j++)
                {
                    double temp = xsecTables.sigma_e(val_low, gammas[j]);
                    target_dot_ref += crossSection[j] * temp * theWeights[j];
                    ref_dot_ref += temp * temp * theWeights[j];
                }
                double curError_below = ref_dot_ref - 2.0 * target_dot_ref + target_dot_target;

                target_dot_ref = 0.0;
                ref_dot_ref = 0.0;
                for (int j = 0; j < N_gamma; j++)
                {
                    double temp = xsecTables.sigma_e(val_high, gammas[j]);
                    target_dot_ref += crossSection[j] * temp * theWeights[j];
                    ref_dot_ref += temp * temp * theWeights[j];
                }
                double curError_above = ref_dot_ref - 2.0 * target_dot_ref + target_dot_target;

                if (curError_below < curError_above)
                    retVal = val_low;
                else
                    retVal = val_high;
            }
            else
            {
                retVal = float(minError_arg - 1) + x_low;
            }
        }
        else if (x_high >= 0.0 && x_high <= 1.0)
        {
            retVal = float(minError_arg) + x_high;
        }
        else
        {
            retVal = float(minError_arg);
        }
    }
    //##############################################################################

    delete[] gammas;
    delete[] crossSection;
    delete[] theWeights;

    return retVal;
}

float XrayPhysics::incoherentScatterDistribution(float Z, float gamma, float theta)
{
    if (xscatterTables.isInitialized() == false)
        xscatterTables.init();
    return xscatterTables.incoherentScatterDistribution(Z, gamma, theta*PI/180.0);
}

float XrayPhysics::coherentScatterDistribution(float Z, float gamma, float theta)
{
    if (xscatterTables.isInitialized() == false)
        xscatterTables.init();
    return xscatterTables.coherentScatterDistribution(Z, gamma, theta * PI / 180.0);
}

float XrayPhysics::incoherentScatterDistribution(const char* chemForm, float gamma, float theta)
{
    if (xscatterTables.isInitialized() == false)
        xscatterTables.init();
    return xscatterTables.incoherentScatterDistribution(chemForm, gamma, theta * PI / 180.0);
}

float XrayPhysics::coherentScatterDistribution(const char* chemForm, float gamma, float theta)
{
    if (xscatterTables.isInitialized() == false)
        xscatterTables.init();
    return xscatterTables.coherentScatterDistribution(chemForm, gamma, theta * PI / 180.0);
}

float XrayPhysics::incoherentScatterDistribution_normalizationFactor(float Z, float gamma)
{
    if (xscatterTables.isInitialized() == false)
        xscatterTables.init();
    return xscatterTables.incoherentScatterDistribution_normalizationFactor(Z, gamma);
}

float XrayPhysics::coherentScatterDistribution_normalizationFactor(float Z, float gamma)
{
    if (xscatterTables.isInitialized() == false)
        xscatterTables.init();
    return xscatterTables.coherentScatterDistribution_normalizationFactor(Z, gamma);
}

float XrayPhysics::incoherentScatterDistribution_normalizationFactor(const char* chemForm, float gamma)
{
    if (xscatterTables.isInitialized() == false)
        xscatterTables.init();
    return xscatterTables.incoherentScatterDistribution_normalizationFactor(chemForm, gamma);
}

float XrayPhysics::coherentScatterDistribution_normalizationFactor(const char* chemForm, float gamma)
{
    if (xscatterTables.isInitialized() == false)
        xscatterTables.init();
    return xscatterTables.coherentScatterDistribution_normalizationFactor(chemForm, gamma);
}

float XrayPhysics::transmission(float Z, float density, float thickness, float* spectralResponse, float* gammas, int N)
{
    if (spectralResponse == NULL || gammas == NULL || N <= 0)
        return 1.0;
    if (N == 1)
        return exp(-thickness * density * xsecTables.sigma(Z, gammas[0]));

    double retVal = 0.0;
    double accum = 0.0;
    for (int i = 0; i < N; i++)
    {
        float T_phi;
        if (i == 0)
            T_phi = gammas[i + 1] - gammas[i];
        else if (i == N - 1)
            T_phi = gammas[i] - gammas[i - 1];
        else
            T_phi = 0.5 * (gammas[i + 1] - gammas[i - 1]);

        if (thickness == 0.0)
            retVal += T_phi * spectralResponse[i] * density * xsecTables.sigma(Z, gammas[i]);
        else
            retVal += T_phi * spectralResponse[i] * exp(-density * thickness * xsecTables.sigma(Z, gammas[i]));
        accum += T_phi * spectralResponse[i];
    }
    retVal = retVal / accum;

    return float(retVal);
}

float XrayPhysics::transmission(const char* chemForm, float density, float thickness, float* spectralResponse, float* gammas, int N)
{
    if (spectralResponse == NULL || gammas == NULL || N <= 0)
        return 1.0;
    if (N == 1)
        return exp(-thickness * density * xsecTables.sigma(chemForm, gammas[0]));

    double retVal = 0.0;
    double accum = 0.0;
    for (int i = 0; i < N; i++)
    {
        float T_phi;
        if (i == 0)
            T_phi = gammas[i + 1] - gammas[i];
        else if (i == N - 1)
            T_phi = gammas[i] - gammas[i - 1];
        else
            T_phi = 0.5 * (gammas[i + 1] - gammas[i - 1]);

        if (thickness == 0.0)
            retVal += T_phi * spectralResponse[i] * density * xsecTables.sigma(chemForm, gammas[i]);
        else
            retVal += T_phi * spectralResponse[i] * exp(-density * thickness * xsecTables.sigma(chemForm, gammas[i]));
        accum += T_phi * spectralResponse[i];
    }
    retVal = retVal / accum;

    return float(retVal);
}

bool XrayPhysics::setBHClookupTable_helper(double* sigma_hat, float* spectralResponse, float* gammas, int N_gamma, float* LUT, float T_lac, int N_lac, float referenceEnergy)
{
    if (spectralResponse == NULL || gammas == NULL || N_gamma <= 0)
        return false;
    if (LUT == NULL || T_lac <= 0.0 || N_lac <= 0)
        return false;

    double* d = new double[N_gamma];
    double accum = 0.0;
    for (int i = 0; i < N_gamma; i++)
    {
        double T_phi;
        if (i == 0)
            T_phi = gammas[i + 1] - gammas[i];
        else if (i == N_gamma - 1)
            T_phi = gammas[i] - gammas[i - 1];
        else
            T_phi = 0.5 * (gammas[i + 1] - gammas[i - 1]);

        d[i] = T_phi * spectralResponse[i];
        accum += T_phi * spectralResponse[i];
    }
    for (int i = 0; i < N_gamma; i++)
        d[i] = d[i] / accum;

    bool retVal = true;

    int N_iter = 10;
    double tol = 1.0e-7;

    LUT[0] = 0.0;
    for (int i = 1; i < N_lac; i++)
    {
        double curLAC = double(i) * T_lac;
        double p_cur = LUT[i - 1];
        if (i == 1)
            p_cur = curLAC;
        /*
        double modelError, denom, theExponent, expFactor;
        for (int n = 0; n < N_iter; n++)
        {
            modelError = 0.0;
            denom = 0.0;
            for (int l = 0; l < N_gamma; l++)
            {
                theExponent = sigma_hat[l];
                expFactor = exp(-theExponent * p_cur);
                modelError += d[l] * expFactor;
                denom += d[l] * theExponent * expFactor;
            }
            //modelError *= T_gamma;
            //denom *= T_gamma;

            denom = denom / modelError; // new method
            modelError = (curLAC + log(modelError)); // new method
            if (fabs(modelError) < tol || fabs(denom) < 1.0e-15)
                break;
            p_cur = p_cur + modelError / denom;
        }
        //*/
        BHCkernel(p_cur, curLAC, sigma_hat, d, N_gamma);
        if (isnan(p_cur))
        {
            printf("XrayPhysics::setBHClookupTable: nan value encountered, quitting!\n");
            retVal = false;
            break;
        }
        if (p_cur >= 0.0)
            LUT[i] = p_cur;
        else
            LUT[i] = LUT[i - 1];
        //printf("%f => %f\n", curLAC, LUT[i]);
    }
    delete[] d;
    return retVal;
}

bool XrayPhysics::BHCkernel(double& monoAtten, double polyAtten, double* normalizedCrossSection, double* d, int N_gamma)
{
    int N_iter = 10;
    double tol = 1.0e-7;

    double modelError, denom, expFactor;
    double theExponent;
    for (int n = 0; n < N_iter; n++)
    {
        modelError = 0.0;
        denom = 0.0;
        for (int l = 0; l < N_gamma; l++)
        {
            theExponent = normalizedCrossSection[l];
            expFactor = exp(-theExponent * monoAtten);
            modelError += d[l] * expFactor;
            denom += d[l] * theExponent * expFactor;
        }
        //modelError *= T_gamma;
        //denom *= T_gamma;

        denom = denom / modelError; // new method
        modelError = (polyAtten + log(modelError)); // new method
        if (fabs(modelError) < tol || fabs(denom) < 1.0e-15)
            break;
        monoAtten += modelError / denom;
    }

    if (isnan(monoAtten))
    {
        //printf("XrayPhysics::BHCkernel: nan value encountered, quitting!\n");
        return false;
    }
    else
        return true;
}

bool XrayPhysics::setBHClookupTable(float Ze, float* spectralResponse, float* gammas, int N_gamma, float* LUT, float T_lac, int N_lac, float referenceEnergy)
{
    if (spectralResponse == NULL || gammas == NULL || N_gamma <= 0)
        return false;
    if (LUT == NULL || T_lac <= 0.0 || N_lac <= 0)
        return false;

    double* sigma_hat = new double[N_gamma];
    double sigma_refE = xsecTables.sigma(Ze, referenceEnergy);
    for (int i = 0; i < N_gamma; i++)
        sigma_hat[i] = xsecTables.sigma(Ze, gammas[i]) / sigma_refE;

    bool retVal = setBHClookupTable_helper(sigma_hat, spectralResponse, gammas, N_gamma, LUT, T_lac, N_lac, referenceEnergy);

    delete[] sigma_hat;
    return retVal;
}

bool XrayPhysics::setBHClookupTable(const char* chemForm, float* spectralResponse, float* gammas, int N_gamma, float* LUT, float T_lac, int N_lac, float referenceEnergy)
{
    if (spectralResponse == NULL || gammas == NULL || N_gamma <= 0)
        return false;
    if (LUT == NULL || T_lac <= 0.0 || N_lac <= 0)
        return false;

    double* sigma_hat = new double[N_gamma];
    double sigma_refE = xsecTables.sigma(chemForm, referenceEnergy);
    for (int i = 0; i < N_gamma; i++)
        sigma_hat[i] = xsecTables.sigma(chemForm, gammas[i]) / sigma_refE;

    //printf("referenceEnergy = %f, sigma_refE = %f\n", referenceEnergy, sigma_refE);

    bool retVal = setBHClookupTable_helper(sigma_hat, spectralResponse, gammas, N_gamma, LUT, T_lac, N_lac, referenceEnergy);

    delete[] sigma_hat;
    return retVal;
}

bool XrayPhysics::setBHlookupTable(float Ze, float* spectralResponse, float* gammas, int N_gamma, float* LUT, float T_lac, int N_lac, float referenceEnergy)
{
    if (spectralResponse == NULL || gammas == NULL || N_gamma <= 0)
        return false;
    if (LUT == NULL || T_lac <= 0.0 || N_lac <= 0)
        return false;

    double* sigma_hat = new double[N_gamma];
    double sigma_refE = xsecTables.sigma(Ze, referenceEnergy);
    for (int i = 0; i < N_gamma; i++)
        sigma_hat[i] = xsecTables.sigma(Ze, gammas[i]) / sigma_refE;

    bool retVal = setBHlookupTable_helper(sigma_hat, spectralResponse, gammas, N_gamma, LUT, T_lac, N_lac, referenceEnergy);

    delete[] sigma_hat;
    return retVal;
}

bool XrayPhysics::setBHlookupTable(const char* chemForm, float* spectralResponse, float* gammas, int N_gamma, float* LUT, float T_lac, int N_lac, float referenceEnergy)
{
    if (spectralResponse == NULL || gammas == NULL || N_gamma <= 0)
        return false;
    if (LUT == NULL || T_lac <= 0.0 || N_lac <= 0)
        return false;

    double* sigma_hat = new double[N_gamma];
    double sigma_refE = xsecTables.sigma(chemForm, referenceEnergy);
    for (int i = 0; i < N_gamma; i++)
        sigma_hat[i] = xsecTables.sigma(chemForm, gammas[i]) / sigma_refE;

    //printf("referenceEnergy = %f, sigma_refE = %f\n", referenceEnergy, sigma_refE);

    bool retVal = setBHlookupTable_helper(sigma_hat, spectralResponse, gammas, N_gamma, LUT, T_lac, N_lac, referenceEnergy);

    delete[] sigma_hat;
    return retVal;
}

bool XrayPhysics::setBHlookupTable_helper(double* sigma_hat, float* spectralResponse, float* gammas, int N_gamma, float* LUT, float T_lac, int N_lac, float referenceEnergy)
{
    if (spectralResponse == NULL || gammas == NULL || N_gamma <= 0)
        return false;
    if (LUT == NULL || T_lac <= 0.0 || N_lac <= 0)
        return false;

    double* d = new double[N_gamma];
    double accum = 0.0;
    for (int i = 0; i < N_gamma; i++)
    {
        double T_phi;
        if (i == 0)
            T_phi = gammas[i + 1] - gammas[i];
        else if (i == N_gamma - 1)
            T_phi = gammas[i] - gammas[i - 1];
        else
            T_phi = 0.5 * (gammas[i + 1] - gammas[i - 1]);

        d[i] = T_phi * spectralResponse[i];
        accum += T_phi * spectralResponse[i];
    }
    for (int i = 0; i < N_gamma; i++)
        d[i] = d[i] / accum;

    bool retVal = true;

    int N_iter = 10;
    double tol = 1.0e-7;

    LUT[0] = 0.0;
    for (int i = 1; i < N_lac; i++)
    {
        double curLAC = double(i) * T_lac;
        double accum = 0.0;
        for (int l = 0; l < N_gamma; l++)
        {
            accum += d[l]*exp(-sigma_hat[l] * curLAC);
        }
        LUT[i] = -log(accum);
        //printf("%f => %f\n", curLAC, LUT[i]);
    }
    delete[] d;
    return retVal;
}

bool XrayPhysics::generateDEDlookUpTables(float* spectralResponses, float* gammas, int N_gamma, float* referenceEnergies, float* basisFunctions, float* LUT, float T_lac, int N_lac)
{
    dualEnergyDecomposition DED;
    return DED.generateDEDlookUpTables(spectralResponses, gammas, N_gamma, referenceEnergies, basisFunctions, LUT, T_lac, N_lac);
}

bool XrayPhysics::setThreeMaterialBHClookupTable(float* spectralResponse, float* gammas, int N_gamma, float referenceEnergy, float* sigmas, float* LUT, float T_atten, int N_atten)
{
    if (spectralResponse == NULL || gammas == NULL || N_gamma <= 0)
        return false;
    if (LUT == NULL || T_atten <= 0.0 || N_atten <= 0)
        return false;

    double* d = new double[N_gamma];
    double accum = 0.0;
    for (int i = 0; i < N_gamma; i++)
    {
        double T_phi;
        if (i == 0)
            T_phi = gammas[i + 1] - gammas[i];
        else if (i == N_gamma - 1)
            T_phi = gammas[i] - gammas[i - 1];
        else
            T_phi = 0.5 * (gammas[i + 1] - gammas[i - 1]);

        d[i] = T_phi * spectralResponse[i];
        accum += T_phi * spectralResponse[i];
    }
    for (int i = 0; i < N_gamma; i++)
        d[i] = d[i] / accum;

    float* sigma_1 = &sigmas[0];
    float* sigma_2 = &sigmas[N_gamma];
    float* sigma_3 = &sigmas[2*N_gamma];

    double* sigma_hat_1 = new double[N_gamma];
    double* sigma_hat_2 = new double[N_gamma];
    double* sigma_hat_3 = new double[N_gamma];
    float ind_ref = gamma_inv(referenceEnergy, gammas, N_gamma);
    int ind_lo = int(ind_ref);
    int ind_hi = min(N_gamma - 1, ind_lo + 1);
    float h = ind_ref - float(ind_lo);
    float sigma_1_ref = (1.0 - h) * sigma_1[ind_lo] + h * sigma_1[ind_hi];
    float sigma_2_ref = (1.0 - h) * sigma_2[ind_lo] + h * sigma_2[ind_hi];
    float sigma_3_ref = (1.0 - h) * sigma_3[ind_lo] + h * sigma_3[ind_hi];
    for (int i = 0; i < N_gamma; i++)
    {
        sigma_hat_1[i] = sigma_1[i] / sigma_1_ref;
        sigma_hat_2[i] = sigma_2[i] / sigma_2_ref;
        sigma_hat_3[i] = sigma_3[i] / sigma_3_ref;
    }

    bool retVal = true;

    int N_iter = 10;
    double tol = 1.0e-7;

    double T_frac = 1.0 / double(N_atten - 1);
    #ifdef USE_OPENMP
    omp_set_num_threads(omp_get_num_procs());
    #pragma omp parallel for schedule(dynamic)
    #endif
    for (int k = 0; k < N_atten; k++)
    {
        double frac_2 = double(k) * T_frac;
        float* LUT_slice = &LUT[k*N_atten*N_atten];
        for (int j = 0; j < N_atten; j++)
        {
            double frac_1 = double(j) * T_frac;
            double* normalizedCrossSection = new double[N_gamma];
            for (int l = 0; l < N_gamma; l++)
                normalizedCrossSection[l] = (1.0 - frac_1) * sigma_hat_1[l] + frac_1 * ((1.0-frac_2)*sigma_hat_2[l] + frac_2*sigma_hat_3[l]);
            for (int i = 0; i < N_atten; i++)
            {
                double polyAtten = double(i) * T_atten;
                double monoAtten = polyAtten;
                if (i > 0)
                    monoAtten = LUT_slice[j * N_atten + i-1];
                double monoAtten_save = monoAtten;
                if (BHCkernel(monoAtten, polyAtten, normalizedCrossSection, d, N_gamma) == false)
                    monoAtten = monoAtten_save;
                LUT_slice[j * N_atten + i] = monoAtten;
            }
            delete[] normalizedCrossSection;
        }
    }

    delete[] d;
    delete[] sigma_hat_1;
    delete[] sigma_hat_2;
    delete[] sigma_hat_3;
    return retVal;
}

bool XrayPhysics::setTwoMaterialBHClookupTable(float* spectralResponse, float* gammas, int N_gamma, float referenceEnergy, float* sigmas, float* LUT, float T_atten, int N_atten, bool usingBasis)
{
    if (spectralResponse == NULL || gammas == NULL || N_gamma <= 0)
        return false;
    if (LUT == NULL || T_atten <= 0.0 || N_atten <= 0)
        return false;

    double* d = new double[N_gamma];
    double accum = 0.0;
    for (int i = 0; i < N_gamma; i++)
    {
        double T_phi;
        if (i == 0)
            T_phi = gammas[i + 1] - gammas[i];
        else if (i == N_gamma - 1)
            T_phi = gammas[i] - gammas[i - 1];
        else
            T_phi = 0.5 * (gammas[i + 1] - gammas[i - 1]);

        d[i] = T_phi * spectralResponse[i];
        accum += T_phi * spectralResponse[i];
    }
    for (int i = 0; i < N_gamma; i++)
        d[i] = d[i] / accum;

    float* sigma_1 = &sigmas[0];
    float* sigma_2 = &sigmas[N_gamma];

    double* sigma_hat_1 = new double[N_gamma];
    double* sigma_hat_2 = new double[N_gamma];
    float ind_ref = gamma_inv(referenceEnergy, gammas, N_gamma);
    int ind_lo = int(ind_ref);
    int ind_hi = min(N_gamma - 1, ind_lo + 1);
    float h = ind_ref - float(ind_lo);
    float sigma_1_ref = (1.0 - h) * sigma_1[ind_lo] + h * sigma_1[ind_hi];
    float sigma_2_ref = (1.0 - h) * sigma_2[ind_lo] + h * sigma_2[ind_hi];
    for (int i = 0; i < N_gamma; i++)
    {
        if (usingBasis)
        {
            sigma_hat_1[i] = sigma_1[i];
            sigma_hat_2[i] = sigma_2[i];
        }
        else
        {
            sigma_hat_1[i] = sigma_1[i] / sigma_1_ref;
            sigma_hat_2[i] = sigma_2[i] / sigma_2_ref;
        }
    }

    bool retVal = true;

    int N_iter = 10;
    double tol = 1.0e-7;

    double T_frac = 1.0 / double(N_atten - 1);
    #ifdef USE_OPENMP
    omp_set_num_threads(omp_get_num_procs());
    #pragma omp parallel for schedule(dynamic)
    #endif
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

    delete[] d;
    delete[] sigma_hat_1;
    delete[] sigma_hat_2;
    return retVal;
}

float XrayPhysics::polychromatic_attenuation(float* spectralResponse, float* gammas, int N_gamma, float g_1, float* sigma_1, float sigma_1_ref, float g_2, float* sigma_2, float sigma_2_ref, float g_3, float* sigma_3, float sigma_3_ref)
{
	float a_1 = g_1 / sigma_1_ref;
	float a_2 = g_2 / sigma_2_ref;
	float a_3 = g_3 / sigma_3_ref;

	double val = 0.0;
	for (int l = 0; l < N_gamma; l++)
	{
		double cur = a_1*sigma_1[l];
		if (sigma_2 != nullptr)
			cur += a_2*sigma_2[l];
		if (sigma_3 != nullptr)
			cur += a_3*sigma_3[l];
		val += spectralResponse[l] * exp(-cur);
	}
	return float(-log(val));
}
