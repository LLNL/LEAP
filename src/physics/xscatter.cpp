
#include "xscatter.h"
#include "xscatter_raw.h"
#include <cstring>
#include <math.h>

#ifndef PI
#define PI 3.1415926535897932385
#endif

const xscatter::element_property_s xscatter::elementProperty_List[xscatter::nElements] = { {1, 1.008, 0.00008988, "H"}, {2, 4.0026, 0.0001785, "He"},
                                                                              {3, 6.939, 0.534, "Li"}, {4, 9.0122, 1.848, "Be"},
                                                                              {5, 10.811, 2.37, "B"}, {6, 12.011000000000001, 2.265, "C"},
                                                                              {7, 14.007000000000001, 0.0012506, "N"},{8, 15.999, 0.0014290000000000001, "O"},
                                                                              {9, 18.997999999999998, 0.001696, "F"}, {10, 20.179000000000002, 0.0008999000000000001, "Ne"},
                                                                              {11, 22.99, 0.9710000000000001, "Na"}, {12, 24.312, 1.74, "Mg"},
                                                                              {13, 26.982, 2.6989, "Al"}, {14, 28.086000000000002, 2.33, "Si"},
                                                                              {15, 30.974, 2.2, "P"}, {16, 32.064, 2.07, "S"},
                                                                              {17, 35.453, 0.003214, "Cl"}, {18, 39.948, 0.0017837, "Ar"},
                                                                              {19, 39.102000000000004, 0.862, "K"}, {20, 40.08, 1.55, "Ca"},
                                                                              {21, 44.958, 2.989, "Sc"}, {22, 47.9, 4.54, "Ti"},
                                                                              {23, 50.942, 6.11, "V"}, {24, 51.996, 7.18, "Cr"},
                                                                              {25, 54.938, 7.44, "Mn"}, {26, 55.846999999999994, 7.874, "Fe"},
                                                                              {27, 58.933, 8.9, "Co"}, {28, 58.71000000000001, 8.902, "Ni"},
                                                                              {29, 63.54, 8.96, "Cu"}, {30, 65.37, 7.133, "Zn"},
                                                                              {31, 69.72, 5.904, "Ga"}, {32, 72.59, 5.323, "Ge"},
                                                                              {33, 74.922, 5.73, "As"}, {34, 78.96, 4.5, "Se"},
                                                                              {35, 79.90899999999999, 0.00759, "Br"}, {36, 83.80000000000001, 0.003733, "Kr"},
                                                                              {37, 85.47, 1.532, "Rb"}, {38, 87.62, 2.54, "Sr"},
                                                                              {39, 88.905, 4.469, "Y"}, {40, 91.22, 6.506, "Zr"},
                                                                              {41, 92.90599999999999, 8.57, "Nb"}, {42, 95.94, 10.22, "Mo"},
                                                                              {43, 99., 11.5, "Tc"}, {44, 101.69999999999999, 12.41, "Ru"},
                                                                              {45, 102.91, 12.41, "Rh"}, {46, 106.4, 12.02, "Pd"},
                                                                              {47, 107.87, 10.5, "Ag"}, {48, 112.4, 8.65, "Cd"},
                                                                              {49, 114.82000000000001, 7.31, "In"}, {50, 118.69000000000001, 7.31, "Sn"},
                                                                              {51, 121.75, 6.691, "Sb"}, {52, 127.60000000000001, 6.24, "Te"},
                                                                              {53, 126.89999999999999, 4.93, "I"}, {54, 131.29999999999998, 0.005887, "Xe"},
                                                                              {55, 132.91, 1.873, "Cs"}, {56, 137.34, 3.5, "Ba"},
                                                                              {57, 138.91, 6.154, "La"},  {58, 140.12, 6.657, "Ce"}, {59, 140.91, 6.71, "Pr"},
                                                                              {60, 144.23999999999998, 6.9, "Nd"}, {61, 145., 7.22, "Pm"},
                                                                              {62, 150.35, 7.46, "Sm"}, {63, 151.96, 5.243, "Eu"},
                                                                              {64, 157.25, 7.9004, "Gd"}, {65, 158.92, 8.229, "Tb"},
                                                                              {66, 162.5, 8.55, "Dy"}, {67, 164.93, 8.795, "Ho"},
                                                                              {68, 167.26000000000002, 9.066, "Er"}, {69, 168.93, 9.321, "Tm"},
                                                                              {70, 173.04, 6.73, "Yb"}, {71, 174.97, 9.849, "Lu"},
                                                                              {72, 178.48999999999998, 13.309999999999999, "Hf"},
                                                                              {73, 180.95000000000002, 16.654, "Ta"}, {74, 183.85, 19.3, "W"},
                                                                              {75, 186.20000000000002, 21.02, "Re"}, {76, 190.2, 22.57, "Os"},
                                                                              {77, 192.2, 22.42, "Ir"}, {78, 195.09, 21.45, "Pt"},
                                                                              {79, 196.97, 19.32, "Au"}, {80, 200.59, 13.546, "Hg"},
                                                                              {81, 204.36999999999998, 11.719999999999999, "Tl"},
                                                                              {82, 207.19, 11.35, "Pb"}, {83, 208.98, 9.747, "Bi"},
                                                                              {84, 209., 9.32, "Po"}, {85, 210., 9., "At"},
                                                                              {86, 222.00000000000003, 0.00973, "Rn"}, {87, 223., 5., "Fr"},
                                                                              {88, 225.99999999999997, 5., "Ra"}, {89, 227., 10.069999999999999,  "Ac"},
                                                                              {90, 232.04, 11.719999999999999, "Th"},
                                                                              {91, 233., 15.37, "Pa"}, {92, 238.05, 18.95, "U"},
                                                                              {93, 237., 20.25, "Np"}, {94, 239.04999999999998, 19.84, "Pu"},
                                                                              {95, 242., 13.67, "Am"}, {96, 247.00000000000003, 13.51, "Cm"},
                                                                              {97, 247.00000000000003, 14., "Bk"}, {98, 250.99999999999997, 14., "Cf"},
                                                                              {99, 252., 14., "Es"}, {100, 257., 14., "Fm"} };

xscatter::xscatter()
{
    KNconstant = CLASSICAL_ELECTRON_RADIUS * CLASSICAL_ELECTRON_RADIUS * AVOGANDROS_NUMBER;
    two_PI_KNconstant = 2.0 * PI * KNconstant;

    relec = 0.28179403267;
    relec2 = relec * relec;

    scatterFunctions.clear();
}

xscatter::~xscatter()
{
    scatterFunctions.clear();
}

bool xscatter::isInitialized()
{
    if (scatterFunctions.size() == 0)
        return false;
    else
        return true;
}

bool xscatter::init()
{
    //printf("initializing scatterFunctions...\n");
    scatterFunctions.clear();
    xscatter_raw hardCodedTables;
    int* tempI = (int*)&hardCodedTables.data[0];
    float* tempF = (float*)&hardCodedTables.data[sizeof(int)];
    int64 curOffset = 0;
    for (int Z = 1; Z <= 100; Z++)
    {
        int N = 0;
        IncoherentAndCoherentScatterFunctions_elem_s elem_crossSects;

        // Incoherent Scatter Function
        tempI = (int*)&hardCodedTables.data[curOffset]; curOffset += sizeof(int);
        N = *tempI;
        for (int i = 0; i < N; i++)
        {
            data anEntry;
            elem_crossSects.incoherent_scatter_function.data_record.push_back(anEntry);
        }
        for (int i = 0; i < N; i++)
        {
            tempF = (float*)&hardCodedTables.data[curOffset]; curOffset += sizeof(float);
            elem_crossSects.incoherent_scatter_function.data_record[i].energy = *tempF;
        }
        for (int i = 0; i < N; i++)
        {
            tempF = (float*)&hardCodedTables.data[curOffset]; curOffset += sizeof(float);
            elem_crossSects.incoherent_scatter_function.data_record[i].value = *tempF;
        }

        // Form Factor
        tempI = (int*)&hardCodedTables.data[curOffset]; curOffset += sizeof(int);
        N = *tempI;
        for (int i = 0; i < N; i++)
        {
            data anEntry;
            elem_crossSects.form_factor.data_record.push_back(anEntry);
        }
        for (int i = 0; i < N; i++)
        {
            tempF = (float*)&hardCodedTables.data[curOffset]; curOffset += sizeof(float);
            elem_crossSects.form_factor.data_record[i].energy = *tempF;
        }
        for (int i = 0; i < N; i++)
        {
            tempF = (float*)&hardCodedTables.data[curOffset]; curOffset += sizeof(float);
            elem_crossSects.form_factor.data_record[i].value = *tempF;
        }

        // Imaginary Anomalous Scattering Factor
        tempI = (int*)&hardCodedTables.data[curOffset]; curOffset += sizeof(int);
        N = *tempI;
        for (int i = 0; i < N; i++)
        {
            data anEntry;
            elem_crossSects.imag_anomalous_scattering_factor.data_record.push_back(anEntry);
        }
        for (int i = 0; i < N; i++)
        {
            tempF = (float*)&hardCodedTables.data[curOffset]; curOffset += sizeof(float);
            elem_crossSects.imag_anomalous_scattering_factor.data_record[i].energy = *tempF;
        }
        for (int i = 0; i < N; i++)
        {
            tempF = (float*)&hardCodedTables.data[curOffset]; curOffset += sizeof(float);
            elem_crossSects.imag_anomalous_scattering_factor.data_record[i].value = *tempF;
        }

        // Real Anomalous Scattering Factor
        tempI = (int*)&hardCodedTables.data[curOffset]; curOffset += sizeof(int);
        N = *tempI;
        for (int i = 0; i < N; i++)
        {
            data anEntry;
            elem_crossSects.real_anomalous_scattering_factor.data_record.push_back(anEntry);
        }
        for (int i = 0; i < N; i++)
        {
            tempF = (float*)&hardCodedTables.data[curOffset]; curOffset += sizeof(float);
            elem_crossSects.real_anomalous_scattering_factor.data_record[i].energy = *tempF;
        }
        for (int i = 0; i < N; i++)
        {
            tempF = (float*)&hardCodedTables.data[curOffset]; curOffset += sizeof(float);
            elem_crossSects.real_anomalous_scattering_factor.data_record[i].value = *tempF;
        }

        scatterFunctions.push_back(elem_crossSects);
    }
    return true;
}

double xscatter::KleinNishinaDistribution(double gamma_in, double theta_in)
{
    // normalized distribution
    double cos_theta = cos(theta_in);
    double P = 1.0 / (1.0 + (gamma_in / ELECTRON_REST_MASS_ENERGY) * (1.0 - cos_theta));
    return 0.5 * P * P * (P + 1.0 / P - 1.0 + cos_theta * cos_theta) / (KleinNishinaCrossSection(gamma_in) / KNconstant);
}

double xscatter::KleinNishinaCrossSection(double gamma_in)
{
    double alpha = gamma_in / ELECTRON_REST_MASS_ENERGY;
    double one_plus_two_alpha = 1.0 + 2.0 * alpha;
    double log_term = log(one_plus_two_alpha);
    double retVal = (1.0 + one_plus_two_alpha) / (2.0 * one_plus_two_alpha * one_plus_two_alpha) + ((alpha * alpha - 1.0 - one_plus_two_alpha) * log_term + 4.0 * alpha) / (2.0 * alpha * alpha * alpha);
    return retVal * two_PI_KNconstant;
}

double xscatter::Smatrix(double x, int Z)
{
    if (x < 0.0 || Z < 1 || Z > 100)
        return 0.0;
    int N = scatterFunctions[Z - 1].incoherent_scatter_function.data_record.size();

    if (scatterFunctions[Z - 1].incoherent_scatter_function.data_record[N - 1].energy <= x)
        return scatterFunctions[Z - 1].incoherent_scatter_function.data_record[N - 1].value;
    else if (scatterFunctions[Z - 1].incoherent_scatter_function.data_record[0].energy > x)
        return scatterFunctions[Z - 1].incoherent_scatter_function.data_record[0].value;

    int i = 0;
    for (i = 0; i < N; i++)
    {
        if (scatterFunctions[Z - 1].incoherent_scatter_function.data_record[i].energy > x)
            break;
    }

    double x_low = scatterFunctions[Z - 1].incoherent_scatter_function.data_record[i - 1].energy;
    double x_high = scatterFunctions[Z - 1].incoherent_scatter_function.data_record[i].energy;
    double value_low = scatterFunctions[Z - 1].incoherent_scatter_function.data_record[i - 1].value;
    double value_high = scatterFunctions[Z - 1].incoherent_scatter_function.data_record[i].value;

    return (value_high - value_low) / (x_high - x_low) * (x - x_low) + value_low;
}

double xscatter::getCoherentFormFactor(double gamma_in, double theta_in, int Z)
{
    return getCoherentFormFactor(8.066e6 * gamma_in * sin(0.5 * theta_in), Z);
}

double xscatter::getCoherentFormFactor(double x, int Z)
{
    int N = scatterFunctions[Z - 1].form_factor.data_record.size();
    double modFF = 0.0;
    double imAnon = 0.0;
    double reAnon = 0.0;

    double x_low, x_high, value_low, value_high;
    int i = 0;

    if (scatterFunctions[Z - 1].form_factor.data_record[N - 1].energy <= x)
        modFF = scatterFunctions[Z - 1].form_factor.data_record[N - 1].value;
    else if (scatterFunctions[Z - 1].form_factor.data_record[0].energy > x)
        modFF = scatterFunctions[Z - 1].form_factor.data_record[0].value;
    else
    {
        i = 0;
        for (i = 0; i < N; i++)
        {
            if (scatterFunctions[Z - 1].form_factor.data_record[i].energy > x)
                break;
        }
        x_low = scatterFunctions[Z - 1].form_factor.data_record[i - 1].energy;
        x_high = scatterFunctions[Z - 1].form_factor.data_record[i].energy;
        value_low = scatterFunctions[Z - 1].form_factor.data_record[i - 1].value;
        value_high = scatterFunctions[Z - 1].form_factor.data_record[i].value;
        modFF = (value_high - value_low) / (x_high - x_low) * (x - x_low) + value_low;
    }

    N = scatterFunctions[Z - 1].imag_anomalous_scattering_factor.data_record.size();
    double xMin = 1.0e-6;
    double xMax = 10.0;
    double xx = min(x, xMax);
    if (xx >= xMin)
    {
        if (scatterFunctions[Z - 1].imag_anomalous_scattering_factor.data_record[N - 1].energy <= xx)
            imAnon = scatterFunctions[Z - 1].imag_anomalous_scattering_factor.data_record[N - 1].value;
        else if (scatterFunctions[Z - 1].imag_anomalous_scattering_factor.data_record[0].energy > xx)
            imAnon = scatterFunctions[Z - 1].imag_anomalous_scattering_factor.data_record[0].value;
        else
        {
            i = 0;
            for (i = 0; i < N; i++)
            {
                if (scatterFunctions[Z - 1].imag_anomalous_scattering_factor.data_record[i].energy > xx)
                    break;
            }
            x_low = scatterFunctions[Z - 1].imag_anomalous_scattering_factor.data_record[i - 1].energy;
            x_high = scatterFunctions[Z - 1].imag_anomalous_scattering_factor.data_record[i].energy;
            value_low = scatterFunctions[Z - 1].imag_anomalous_scattering_factor.data_record[i - 1].value;
            value_high = scatterFunctions[Z - 1].imag_anomalous_scattering_factor.data_record[i].value;
            imAnon = (value_high - value_low) / (x_high - x_low) * (xx - x_low) + value_low;
        }
    }

    N = scatterFunctions[Z - 1].real_anomalous_scattering_factor.data_record.size();
    if (xx >= xMin)
    {
        if (scatterFunctions[Z - 1].real_anomalous_scattering_factor.data_record[N - 1].energy <= xx)
            reAnon = scatterFunctions[Z - 1].real_anomalous_scattering_factor.data_record[N - 1].value;
        else if (scatterFunctions[Z - 1].real_anomalous_scattering_factor.data_record[0].energy > xx)
            reAnon = scatterFunctions[Z - 1].real_anomalous_scattering_factor.data_record[0].value;
        else
        {
            i = 0;
            for (i = 0; i < N; i++)
            {
                if (scatterFunctions[Z - 1].real_anomalous_scattering_factor.data_record[i].energy > xx)
                    break;
            }
            x_low = scatterFunctions[Z - 1].real_anomalous_scattering_factor.data_record[i - 1].energy;
            x_high = scatterFunctions[Z - 1].real_anomalous_scattering_factor.data_record[i].energy;
            value_low = scatterFunctions[Z - 1].real_anomalous_scattering_factor.data_record[i - 1].value;
            value_high = scatterFunctions[Z - 1].real_anomalous_scattering_factor.data_record[i].value;
            reAnon = (value_high - value_low) / (x_high - x_low) * (xx - x_low) + value_low;
        }
    }
    else
    {
        reAnon = -elementProperty_List[Z - 1].std_atomic_weight;
    }
    double ampSq = (modFF + reAnon) * (modFF + reAnon) + imAnon * imAnon;
    return ampSq;
}

float xscatter::incoherentScatterDistribution_normalizationFactor(int Z, float gamma_in)
{
    double accum = 0.0;
    double T_theta = 0.1 * PI / 180.0;
    int N_theta = int(floor(0.5 + PI / T_theta));
    //int N_theta = 3600; // 2*PI/(0.1*PI/180) = 360/0.1 = 3600
    for (int i = 0; i < N_theta; i++)
    {
        double theta = T_theta * double(i);
        accum += incoherentScatterDistribution(Z, gamma_in, theta) * T_theta * fabs(sin(theta)) * 2.0 * PI;
    }
    return float(accum);
}

float xscatter::coherentScatterDistribution_normalizationFactor(int Z, float gamma_in)
{
    double accum = 0.0;
    double T_theta = 0.1 * PI / 180.0;
    int N_theta = int(floor(0.5 + PI / T_theta));
    //int N_theta = 3600; // 2*PI/(0.1*PI/180) = 360/0.1 = 3600
    for (int i = 0; i < N_theta; i++)
    {
        double theta = T_theta * double(i);
        accum += coherentScatterDistribution(Z, gamma_in, theta) * T_theta * fabs(sin(theta)) * 2.0 * PI;
    }
    return float(accum);
}

float xscatter::incoherentScatterDistribution_normalizationFactor(float Z, float gamma_in)
{
    double accum = 0.0;
    double T_theta = 0.1 * PI / 180.0;
    int N_theta = int(floor(0.5 + PI / T_theta));
    //int N_theta = 3600; // 2*PI/(0.1*PI/180) = 360/0.1 = 3600
    for (int i = 0; i < N_theta; i++)
    {
        double theta = T_theta * double(i);
        accum += incoherentScatterDistribution(Z, gamma_in, theta) * T_theta * fabs(sin(theta)) * 2.0 * PI;
    }
    return float(accum);
}

float xscatter::coherentScatterDistribution_normalizationFactor(float Z, float gamma_in)
{
    double accum = 0.0;
    double T_theta = 0.1 * PI / 180.0;
    int N_theta = int(floor(0.5 + PI / T_theta));
    //int N_theta = 3600; // 2*PI/(0.1*PI/180) = 360/0.1 = 3600
    for (int i = 0; i < N_theta; i++)
    {
        double theta = T_theta * double(i);
        accum += coherentScatterDistribution(Z, gamma_in, theta) * T_theta * fabs(sin(theta)) * 2.0 * PI;
    }
    return float(accum);
}

float xscatter::incoherentScatterDistribution_normalizationFactor(float* chemForm, float gamma_in)
{
    double accum = 0.0;
    double T_theta = 0.1 * PI / 180.0;
    int N_theta = int(floor(0.5 + PI / T_theta));
    //int N_theta = 3600; // 2*PI/(0.1*PI/180) = 360/0.1 = 3600
    for (int i = 0; i < N_theta; i++)
    {
        double theta = T_theta * double(i);
        accum += incoherentScatterDistribution(chemForm, gamma_in, theta) * T_theta * fabs(sin(theta)) * 2.0 * PI;
    }
    return float(accum);
}

float xscatter::coherentScatterDistribution_normalizationFactor(float* chemForm, float gamma_in)
{
    double accum = 0.0;
    double T_theta = 0.1 * PI / 180.0;
    int N_theta = int(floor(0.5 + PI / T_theta));
    //int N_theta = 3600; // 2*PI/(0.1*PI/180) = 360/0.1 = 3600
    for (int i = 0; i < N_theta; i++)
    {
        double theta = T_theta * double(i);
        accum += coherentScatterDistribution(chemForm, gamma_in, theta) * T_theta * fabs(sin(theta)) * 2.0 * PI;
    }
    return float(accum);
}

float xscatter::incoherentScatterDistribution_normalizationFactor(const char* chemForm, float gamma_in)
{
    //float* elementCount = xsecTables.parseChemicalFormula(chemForm);
    float* elementCount = xsecTables.parseMaterialText(chemForm);
    if (elementCount == NULL)
        return 0.0;
    else
    {
        float retVal = incoherentScatterDistribution_normalizationFactor(elementCount, gamma_in);
        free(elementCount);
        return retVal;
    }
}

float xscatter::coherentScatterDistribution_normalizationFactor(const char* chemForm, float gamma_in)
{
    //float* elementCount = xsecTables.parseChemicalFormula(chemForm);
    float* elementCount = xsecTables.parseMaterialText(chemForm);
    if (elementCount == NULL)
        return 0.0;
    else
    {
        float retVal = coherentScatterDistribution_normalizationFactor(elementCount, gamma_in);
        free(elementCount);
        return retVal;
    }
}

float xscatter::incoherentScatterDistribution(int Z, float gamma_in, float theta_in)
{
    double x = 8.066e6 * gamma_in * sin(0.5 * theta_in);
    return KleinNishinaDistribution(gamma_in, theta_in) * Smatrix(x, Z);
}

float xscatter::coherentScatterDistribution(int Z, float gamma_in, double theta_in)
{
    if (gamma_in < 0.0 || Z < 1 || Z > 100)
        return 0.0;

    double ampSq = getCoherentFormFactor(gamma_in, theta_in, Z);
    return 0.5 * relec2 * (1.0 + cos(theta_in) * cos(theta_in)) * ampSq;
}

float xscatter::incoherentScatterDistribution(float Z, float gamma_in, float theta_in)
{
    float Z_near = floor(Z+0.5);
    if (fabs(Z - Z_near) < 1.0e-4)
        return incoherentScatterDistribution(int(Z_near), gamma_in, theta_in);
    else
    {
        int Z_lo = int(floor(Z));
        int Z_hi = int(ceil(Z));
        float dZ = Z - float(Z_lo);
        return (1.0 - dZ) * incoherentScatterDistribution(Z_lo, gamma_in, theta_in) + dZ * incoherentScatterDistribution(Z_hi, gamma_in, theta_in);
    }
}

float xscatter::coherentScatterDistribution(float Z, float gamma_in, double theta_in)
{
    float Z_near = floor(Z + 0.5);
    if (fabs(Z - Z_near) < 1.0e-4)
        return coherentScatterDistribution(int(Z_near), gamma_in, theta_in);
    else
    {
        int Z_lo = int(floor(Z));
        int Z_hi = int(ceil(Z));
        float dZ = Z - float(Z_lo);
        return (1.0 - dZ) * coherentScatterDistribution(Z_lo, gamma_in, theta_in) + dZ * coherentScatterDistribution(Z_hi, gamma_in, theta_in);
    }
}

float xscatter::incoherentScatterDistribution(float* chemForm, float gamma_in, float theta_in)
{
    if (chemForm == NULL)
        return 0.0;
    double sumA = 0.0;
    for (int i = 1; i <= 100; i++)
    {
        if (chemForm[i] > 0.0)
        {
            sumA += chemForm[i] * double(xsecTables.getAtomicMass(i));
        }
    }

    double retVal = 0.0;
    double sum_normalizationFactors = 0.0;
    for (int i = 1; i <= 100; i++)
    {
        if (chemForm[i] > 0.0)
        {
            float curVal = incoherentScatterDistribution(i, gamma_in, theta_in);
            double curWeight = double(xsecTables.getAtomicMass(i)) * chemForm[i] / sumA;
            sum_normalizationFactors += curWeight;
            retVal += curWeight * curVal;
        }
    }

    return float(retVal);
}

float xscatter::coherentScatterDistribution(float* chemForm, float gamma_in, float theta_in)
{
    if (chemForm == NULL)
        return 0.0;
    double sumA = 0.0;
    for (int i = 1; i <= 100; i++)
    {
        if (chemForm[i] > 0.0)
        {
            sumA += chemForm[i] * double(xsecTables.getAtomicMass(i));
        }
    }

    double retVal = 0.0;
    double sum_normalizationFactors = 0.0;
    for (int i = 1; i <= 100; i++)
    {
        if (chemForm[i] > 0.0)
        {
            float curVal = coherentScatterDistribution(i, gamma_in, theta_in);
            double curWeight = double(xsecTables.getAtomicMass(i)) * chemForm[i] / sumA;
            sum_normalizationFactors += curWeight;
            retVal += curWeight * curVal;
        }
    }
    
    return float(retVal);
}

float xscatter::incoherentScatterDistribution(const char* chemForm, float gamma_in, float theta_in)
{
    //float* elementCount = xsecTables.parseChemicalFormula(chemForm);
    float* elementCount = xsecTables.parseMaterialText(chemForm);
    if (elementCount == NULL)
        return 0.0;
    else
    {
        float retVal = incoherentScatterDistribution(elementCount, gamma_in, theta_in);
        free(elementCount);
        return retVal;
    }
}

float xscatter::coherentScatterDistribution(const char* chemForm, float gamma_in, float theta_in)
{
    //float* elementCount = xsecTables.parseChemicalFormula(chemForm);
    float* elementCount = xsecTables.parseMaterialText(chemForm);
    if (elementCount == NULL)
        return 0.0;
    else
    {
        float retVal = coherentScatterDistribution(elementCount, gamma_in, theta_in);
        free(elementCount);
        return retVal;
    }
}
