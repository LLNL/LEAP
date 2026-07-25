#ifndef XSOURCE_H
#define XSOURCE_H

#ifdef WIN32
#pragma once
#endif

//#include "crossSections.h"
#include "xsec.h"
#include <stdlib.h>

class xraySource
{
public:
    xraySource();
    ~xraySource();

    bool init(xsec*);
    
    float* simulateSpectra(float kVp_in, float takeOffAngle_in, int Z_in, float* gammas, int N_in);
    float* takeOffAngleConversionFactor(float kVp_in, float takeOffAngle_cur, float takeOffAngle_new, int Z_in, float* gammas, int N_in);
    
private:
    
    float* bremsstrahlung();

	float fNjkl(int k, int l);
    
    float muTotal(float theEnergy);
    float* muTotal(float* Es, int N_E);
    
    float Rfunc(float gamma_cur);
    int getJ();
    float omega_jk(int j, int k);
    
    float Ejkl(int j, int k, int l);
    float Eabs(int j, int k, int l);
    float Pjkl(int j, int k, int l);
	float muTotal_jkl(int j, int k, int l);
    
    float rjk(int j, int k);
    float nq(int k);
    float bq(int k);

	float PhilibertAbsorptionCorrectionFactor(float gamma_cur, float sigma_cur = 0.0);
	float PhilibertAbsorptionCorrectionFactor_characteristic(float gamma_cur, float sigma_cur = 0.0);
    
	//float Leff(float Eedge, float Eline);
    float muEff(float Eedge);
	float Delta(float Eedge, float Eline, int k, int l);
    
    float kVp;
    int Z;
    float takeOffAngle;
    
    float* energies;
    int N;
    
    float gamma(int);
    float gamma_inv(float);

    float massDensity();

	float PhilibertConstant;
	float PhilibertExponent;
    
    xsec* xsecTables;
};

#endif
