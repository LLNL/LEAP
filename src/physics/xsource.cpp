#include "xsource.h"
#include <math.h>

#ifndef PI
	#define PI 3.1415926535897932385
#endif

xraySource::xraySource()
{
    PhilibertConstant = 4.0e5;
    PhilibertExponent = 1.65;
}


xraySource::~xraySource()
{
}

bool xraySource::init(xsec* crossSectionTables)
{
    xsecTables = crossSectionTables;
    if (xsecTables != NULL)
        return true;
    else
        return false;
}

float* xraySource::takeOffAngleConversionFactor(float kVp_in, float takeOffAngle_cur, float takeOffAngle_new, int Z_in, float* gammas, int N_in)
{
    if (xsecTables == NULL)
    {
        printf("xraySource::takeOffAngleConversionFactor: Error: must provide cross section database!\n");
        return NULL;
    }

    kVp = kVp_in;
    takeOffAngle = takeOffAngle_cur * PI / 180.0;
    Z = Z_in;
    N = N_in;
    energies = gammas;

    float* retVal = (float*)malloc(sizeof(float) * N);

    if (takeOffAngle_cur == takeOffAngle_new)
    {
        for (int i = 0; i < N; i++)
            retVal[i] = 1.0;
        return retVal;
    }

    for (int i = 0; i < N; i++)
    {
        float gamma_cur = gamma(i);
        if (Z != 0 && gamma_cur > 0.0 && takeOffAngle_cur != takeOffAngle_new)
        {
            takeOffAngle = takeOffAngle_cur * PI / 180.0;
            float denom = PhilibertAbsorptionCorrectionFactor(gamma_cur);
            takeOffAngle = takeOffAngle_new * PI / 180.0;
            float num = PhilibertAbsorptionCorrectionFactor(gamma_cur);
            retVal[i] = num / denom;
        }
        else
            retVal[i] = 1.0;
    }

    return retVal;
}

float* xraySource::simulateSpectra(float kVp_in, float takeOffAngle_in, int Z_in, float* energies_in, int N_in)
{
	if (xsecTables == NULL)
    {
        printf("xraySource::simulateSpectra: Error: must provide cross section database!\n");
        return NULL;
    }
	if (kVp_in < 2.0 || kVp_in > float(MAX_ENERGY))
	{
		printf("xraySource::simulateSpectra: Error: Can only simulate spectra with kV between 2 and %d keV\n", MAX_ENERGY);
		return NULL;
	}
    if (Z_in != 29 && Z_in != 42 && Z_in != 74 && Z_in != 79)
    {
        printf("xraySource::simulateSpectra: Error: Can only simulate source spectra for Cu, Mo, W, and Au!\n");
        return NULL;
    }

    energies = energies_in;
    kVp = kVp_in;
    takeOffAngle = takeOffAngle_in * PI / 180.0;
    Z = Z_in;
    N = N_in;
    
    float* s = bremsstrahlung();

    if (s == NULL)
        return NULL;

    ////////////////////////////////////////////////
    // Get characteristic lines and amplitudes
    //int numLines = 0;
    //float** lineList = FinkLines(numLines);
    int j = getJ();
    float lineConst = 6.25e15;

    float lineList[4][2];
    int ind = 0;

    float Emin = energies[0];
    //float Eo = energies[N-1];
    float Eo = kVp;
    //float dE = (Eo - Emin)/float(N_gamma-1);
    for (int k = 0; k < 2; k++)
    {
        for (int l = 0; l < 2; l++)
        {
            if (Ejkl(j,k,l) >= Emin && Eabs(j,k,l) < Eo)
            {
                lineList[ind][0] = Ejkl(j,k,l); // energy
				float temp = fNjkl(k,l); // amplitude
                lineList[ind][1] = lineConst*temp;
                ind += 1;
            }
        }
    }
	int numLines = ind;
    ////////////////////////////////////////////////

    if (numLines > 0)
    {
        //float eMin = energies[0];
        //float eMax = energies[N-1];
        //float dE = (eMax-eMin) / float(N-1);
        for (int j = 0; j < numLines; j++)
        {
            float eLine = lineList[j][0];
            float amp = lineList[j][1];

            if (Z == 74)
            {
                //printf("kV = %f, amp(before) = %f\n", kVp, amp);
                if (numLines == 1)
                    amp *= 0.3;
                else if (numLines == 2)
                {
                    if (j == 0)
                        amp *= 0.3;
                    else
                        amp *= 0.22;
                }
                else
                {
                    if (j == 0)
                        amp *= 0.3;
                    else if (j == 1)
                        amp *= 0.22;
                }
                //printf("kV = %f, amp(after) = %f\n", kVp, amp);
            }
            
            //int iLine = int((eLine - eMin) / dE + 0.5);
            int iLine = int(0.5 + gamma_inv(eLine));
            //printf("kV = %f, iLine = %d, amp(after) = %f\n", kVp, iLine, amp);
            s[iLine] += amp;
        }
    }
    
    float c = 1;
    if (kVp <= 70.0)
        c = 1.0 / fabs(-1.525+0.06413*kVp);
    else if (kVp <= 140.0)
        c = 1.0 / (4.865-0.03825*kVp+0.0001624*kVp*kVp);
    else
        c = 1.0 / 2.693;

    for (int i = 0; i < N; i++)
        s[i] *= c;

    return s;
}

float* xraySource::bremsstrahlung()
{
    float a;
    if (Z <= 45)
        a =  0.2986 - 0.002857*float(Z);
    else
        a =  0.2010 - 0.000690*float(Z);
    
    int N_at_kVp = int(0.5+gamma_inv(kVp));
    //float d = (kVp - gamma(0)) / float(N - 1);
    float d = (kVp - gamma(0)) / float(N_at_kVp);

    float* s = (float*) malloc(sizeof(float)*N);
    for (int i = 0; i < N; i++)
    {
        float gamma_cur = gamma(i);
        
        float f = PhilibertAbsorptionCorrectionFactor(gamma_cur);
        float B = pow(kVp / (2.0*gamma_cur), a);
        float L = log(33.397*(2.0*kVp+gamma_cur)/float(Z));
        float R = Rfunc(gamma_cur);
        
        if (gamma_cur <= kVp)
            s[i] = d * 5.47152e9 * Z * (kVp / gamma_cur - 1.0) * B * R / L * f;
        else
            s[i] = 0.0;
    }

    return s;
}

float xraySource::Rfunc(float gamma_cur)
{
    return 1.0 - 0.0081517*float(Z) + 3.613e-5*float(Z*Z) + 0.009583*float(Z)*exp(-kVp/gamma_cur) + 0.001141*kVp;
}

/////////////////////////////////////////////////////////////////////////////////////////////
float xraySource::muEff(float Eedge)
{
    float Eo = kVp;
    float de = (Eo - Eedge) / (25.0 - 1.0);

    int lenLst = int(ceil((Eo - Eedge) / de)) + 1;
    float* eLst = (float*) malloc(sizeof(float)*lenLst);
    float* approxSpec = (float*) malloc(sizeof(float)*lenLst);
	float accum = 0.0;
    for (int i = 0; i < lenLst; i++)
    {
        eLst[i] = Eedge + de*float(i);
        approxSpec[i] = Eo/eLst[i] - 1.0;
		accum += approxSpec[i];
    }

	float* mus = muTotal(eLst, lenLst);
	float retVal = 0.0;
	for (int i = 0; i < lenLst; i++)
		retVal += mus[i] * approxSpec[i] / accum;

    free(approxSpec);
    free(eLst);
	free(mus);
    return retVal;
}

float xraySource::Delta(float Eedge, float Eline, int k, int l)
{
    int j = getJ();
    float c = 1.098e-5;
    
	float muEffec = muEff(Eedge);
    float mul = muTotal(Eline);
	float Leff = muEffec / mul * log(1.0 + mul / muEffec);

	return c*float(Z*Z) * rjk(j,k) * Eedge / (nq(k) * bq(k)) * Leff;
}

float xraySource::fNjkl(int k, int l)
{
    // fNjkl ==> Delta ==> Leff ==> muEff
    int j = getJ();
    //float A = axsec->getAtomicMass(Z);
    //float Eo = gamma(N-1);
    float Eo = kVp;

    float Eedge = Eabs(j,k,l);
    float Eline = Ejkl(j,k,l);
    //float muLine = muTotal(Eline);
    float Ln = log(33.3971/float(Z) * (2.0*kVp + Eedge));
    float U = Eo / Eedge;
    float Uterm = U * log(U) - (U - 1);

	float sigma_cur = muTotal_jkl(j,k,l);

    float fx = PhilibertAbsorptionCorrectionFactor_characteristic(Eline, sigma_cur);
    
    return omega_jk(j,k) * Pjkl(j,k,l) * ( nq(k) * bq(k) ) / float(2*Z) * Uterm/Ln * fx * Rfunc(Eedge)/(4.0*PI) * (1.0 + Delta(Eedge, Eline, k, l));
}

float xraySource::PhilibertAbsorptionCorrectionFactor_characteristic(float gamma_cur, float sigma_cur/* = 0.0*/)
{
	float sin_psi = sin(takeOffAngle);
	float h_local = 1.2 * xsecTables->getAtomicMass(Z) / float(Z * Z);
	float h_factor = h_local / (1.0 + h_local);
	float theRatio;
	float Z_over_A = float(Z) / xsecTables->getAtomicMass(Z);
	float kVp_e165 = pow(float(kVp), 1.65);

    float f = 0.0;
    if (gamma_cur > 0.0)
    {
		if (sigma_cur == 0.0)
			sigma_cur = xsecTables->sigma_e(Z, gamma_cur);
        theRatio = sigma_cur * Z_over_A / (4.0e5 / (kVp_e165 - pow(gamma_cur,1.65)));
        //theRatio /= xsecTables->areaConversionFactor;
        f = sin_psi*sin_psi / ((sin_psi + theRatio)*(sin_psi + h_factor * theRatio));
    }
    
	return f;
}

float xraySource::PhilibertAbsorptionCorrectionFactor(float gamma_cur, float sigma_cur/* = 0.0*/)
{
	float sin_psi = sin(takeOffAngle);
	float h_local = 1.2 * xsecTables->getAtomicMass(Z) / float(Z * Z);
	float h_factor = h_local / (1.0 + h_local);
	float theRatio;
	float Z_over_A = float(Z) / xsecTables->getAtomicMass(Z);
	float kVp_e165 = pow(float(kVp), PhilibertExponent);

    float f = 0.0;
    if (gamma_cur > 0.0)
    {
		if (sigma_cur == 0.0)
			sigma_cur = xsecTables->sigma_e(Z, gamma_cur);
        theRatio = sigma_cur * Z_over_A / (PhilibertConstant / (kVp_e165 - pow(gamma_cur,PhilibertExponent)));
        //theRatio /= xsecTables->areaConversionFactor;
        f = sin_psi*sin_psi / ((sin_psi + theRatio)*(sin_psi + h_factor * theRatio));
    }
    
	return f;
}

float xraySource::muTotal(float theEnergy)
{
    return xsecTables->sigma_e(Z, theEnergy) * float(Z) / xsecTables->getAtomicMass(Z);
}

float* xraySource::muTotal(float* Es, int N_E)
{
    float* mus = (float*) malloc(sizeof(float)*N_E);

    for (int i = 0; i < N_E; i++)
        mus[i] = muTotal(Es[i]);
    
    return mus;
}

/////////////////////////////////////////////////////////////////////////////////////////////
///   Parametric functions
/////////////////////////////////////////////////////////////////////////////////////////////
float xraySource::rjk(int j, int k)
{
    float retVal = 0.0;
    if (j == 0)
    {
        if (k == 0)
            retVal = 7.957;
        else
            retVal = 4.701;
    }
    else if (j == 1)
    {
        if (k == 0)
            retVal = 6.968;
        else
            retVal = 6.011;
    }
    else if (j == 2)
    {
        if (k == 0)
            retVal = 5.018;
        else
            retVal = 4.280;
    }
    else if (j == 3)
    {
        if (k == 0)
            retVal = 4.916;
        else
            retVal = 3.989;
    }
    else
        return 0.0;
    
    return (retVal-1.0) / retVal;
}

float xraySource::nq(int k)
{
    if (k == 0)
        return 2.0;
    else if (k == 1)
        return 8.0;
    else
        return 0.0;
}

float xraySource::bq(int k)
{
    float A = 2.519;
    float B = 26.6;
    float C = -0.0968;
    float D = 0.0103;
    
    if (k == 0)
        return 0.35*1.73;
    else
        return A/(Z-B)+C+D*float(Z);
    
}

int xraySource::getJ()
{
    switch (Z)
    {
        case 29: // Cu
            return 0;
            break;
        case 42: // Mo
            return 1;
            break;
        case 74: // W
            return 2;
            break;
        case 79: // Au
            return 3;
            break;
        default:
            return 0;
            break;
    }
}

float xraySource::omega_jk(int j, int k)
{
    if (j == 0)
    {
        if (k == 0)
            return 0.441;
        else
            return 0.0100;
    }
    else if (j == 1)
    {
        if (k == 0)
            return 0.742;
        else
            return 0.0376;
    }
    else if (j == 2)
    {
        if (k == 0)
            return 0.983;
        else
            return 0.280;
    }
    else if (j == 3)
    {
        if (k == 0)
            return 0.980;
        else
            return 0.337;
    }
    else
        return 0.0;
}

float xraySource::Ejkl(int j, int k, int l)
{
    if (j == 0)
    {
        if (k == 0)
        {
            if (l == 0)
                return 8.04;
            else
                return 8.90331;
        }
        else
        {
            if (l == 0)
                return 0.950;
            else
                return 0.930;
        }
    }
    else if (j == 1)
    {
        if (k == 0)
        {
            if (l == 0)
                return 17.441;
            else
                return 19.6575;
        }
        else
        {
            if (l == 0)
                return 2.394;
            else
                return 2.29264;
        }
    }
    else if (j == 2)
    {
        if (k == 0)
        {
            if (l == 0)
                return 58.856;
            else
                return 67.9528;
        }
        else
        {
            if (l == 0)
                return 9.671;
            else
                return 8.39036;
        }
    }
    else if (j == 3)
    {
        if (k == 0)
        {
            if (l == 0)
                return 68.177;
            else
                return 77.971;
        }
        else
        {
            if (l == 0)
                return 11.440;
            else
                return 9.703;
        }
    }
    else
        return 0.0;
}

float xraySource::muTotal_jkl(int j, int k, int l)
{
	float retVal = 0.0;
    if (j == 0)
    {
        if (k == 0)
        {
            if (l == 0)
                retVal = 51.4685;
            else
                retVal = 38.5121;
        }
        else
        {
            if (l == 0)
                retVal = 7407.29;
            else
                retVal = 1612.97;
        }
    }
    else if (j == 1)
    {
        if (k == 0)
        {
            if (l == 0)
                retVal = 18.777;
            else
                retVal = 13.3906;
        }
        else
        {
            if (l == 0)
                retVal = 612.068;
            else
                retVal = 682.347;
        }
    }
    else if (j == 2) // W
    {
        if (k == 0)
        {
            if (l == 0)
                retVal = 3.84917;
            else
                retVal = 2.62661;
        }
        else
        {
            if (l == 0)
                retVal = 104.207;
            else
                retVal = 150.188;
        }
    }
    else if (j == 3)
    {
        if (k == 0)
        {
            if (l == 0)
                retVal = 3.22008;
            else
                retVal = 2.26688;
        }
        else
        {
            if (l == 0)
                retVal = 82.885;
            else
                retVal = 126.589;
        }
    }

    //multiply by areaConversionFactor?
	retVal *= xsecTables->getAtomicMass(Z) / float(Z);

	return retVal;
}

float xraySource::Eabs(int j, int k, int l)
{
    if (j == 0)
    {
        if (k == 0)
        {
            if (l == 0)
                return 8.944;
            else
                return 8.944;
        }
        else
        {
            if (l == 0)
                return 0.958;
            else
                return 0.938;
        }
    }
    else if (j == 1)
    {
        if (k == 0)
        {
            if (l == 0)
                return 19.966;
            else
                return 19.966;
        }
        else
        {
            if (l == 0)
                return 2.63;
            else
                return 2.5234;
        }
    }
    else if (j == 2)
    {
        if (k == 0)
        {
            if (l == 0)
                return 69.69;
            else
                return 69.69;
        }
        else
        {
            if (l == 0)
                return 11.58;
            else
                return 10.21;
        }
    }
    else if (j == 3)
    {
        if (k == 0)
        {
            if (l == 0)
                return 80.98;
            else
                return 80.98;
        }
        else
        {
            if (l == 0)
                return 13.78;
            else
                return 11.94;
        }
    }
    else
        return 0.0;
}

float xraySource::Pjkl(int j, int k, int l)
{
    if (j == 0)
    {
        if (k == 0)
        {
            if (l == 0)
                return 0.852;
            else
                return 0.148;
        }
        else
        {
            if (l == 0)
                return 0.1575;
            else
                return 0.7874;
        }
    }
    else if (j == 1)
    {
        if (k == 0)
        {
            if (l == 0)
                return 0.837;
            else
                return 0.137;
        }
        else
        {
            if (l == 0)
                return 0.264;
            else
                return 0.645;
        }
    }
    else if (j == 2)
    {
        if (k == 0)
        {
            if (l == 0)
                return 0.719;
            else
                return 0.278;
        }
        else
        {
            if (l == 0)
                return 0.239;
            else
                return 0.5266;
        }
    }
    else if (j == 3)
    {
        if (k == 0)
        {
            if (l == 0)
                return 0.7115;
            else
                return 0.285;
        }
        else
        {
            if (l == 0)
                return 0.2384;
            else
                return 0.5245;
        }
    }
    else
        return 0.0;
}

float xraySource::gamma(int i)
{
    return energies[max(0,min(i,N-1))];
}

float xraySource::gamma_inv(float val)
{
    if (val <= energies[0])
        return 0.0;
    // energies[0] < val
    for (int i = 1; i < N; i++)
    {
        if (val <= energies[i])
        {
            // energies[i-1] <= val <= energies[i]
            
            float d = (val-energies[i-1]) / (energies[i] - energies[i-1]);
            
            return float(i-1)+d;
        }
    }
        
    return float(N-1);
}

float xraySource::massDensity()
{
    //return 1.0;
    return xsecTables->getMassDensity(Z);
}
