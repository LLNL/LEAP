#ifndef XSEC_H
#define XSEC_H

#ifdef WIN32
#pragma once
#endif

#define USE_OPENMP

#include <stdlib.h>
#include <vector>
#include <string>

using namespace std;

#define MAX_XRAY_ENERGY 20000 // 20 GeV
#define MAX_ENERGY 20000 // 20 GeV

#ifndef ELECTRON_REST_MASS_ENERGY
	#define ELECTRON_REST_MASS_ENERGY 510.975 // keV
#endif

#ifndef CLASSICAL_ELECTRON_RADIUS
	#define CLASSICAL_ELECTRON_RADIUS 2.8179403267e-13 // cm
#endif

#ifndef AVOGANDROS_NUMBER
	#define AVOGANDROS_NUMBER 6.0221414107e23 // mol^-1
#endif

#ifdef WIN32
	typedef unsigned long long int64;
#else
	typedef unsigned long int int64;
#endif

class xsec
{
public:
    xsec();
    ~xsec();
    bool clearAll();
    bool init();
    
    float sigma(const char* chemForm, float theEnergy, int which = -1);
    float sigma(float* chemForm, float theEnergy, int which = -1);

    float sigma_e(const char* chemForm, float theEnergy, int which = -1);
    float sigma_e(float* chemForm, float theEnergy, int which = -1);

    float sigma(float Ze, float theEnergy, int which = -1);
    float sigma_e(float Ze, float theEnergy, int which = -1);

    float sigma(int Z, float theEnergy, int which = -1);
    float sigma_e(int Z, float theEnergy, int which = -1);

    // Cross Section for each component
    float sigmaPE(const char* chemForm, float theEnergy);
    float sigmaPE(float Ze, float theEnergy);
    float sigmaPE(int Z, float theEnergy);

    float sigmaCS(const char* chemForm, float theEnergy);
    float sigmaCS(float Ze, float theEnergy);
    float sigmaCS(int Z, float theEnergy);

    float sigmaRS(const char* chemForm, float theEnergy);
    float sigmaRS(float Ze, float theEnergy);
    float sigmaRS(int Z, float theEnergy);

    float sigmaPP(const char* chemForm, float theEnergy);
    float sigmaPP(float Ze, float theEnergy);
    float sigmaPP(int Z, float theEnergy);

    float sigmaTP(const char* chemForm, float theEnergy);
    float sigmaTP(float Ze, float theEnergy);
    float sigmaTP(int Z, float theEnergy);

    float mu(int Z, float theEnergy, int which = -1);

    float sigma_inv(const char* chemForm, float val, int which = -1);
    float sigma_inv(float Ze, float val, int which = -1);
    float sigma_inv(int Z, float val, int which = -1);

    float* parseChemicalFormula(const char*);
    float* parseMaterialText(const char*);

    float getAtomicMass(int Z);
    float getMassDensity(int Z);
    float getElectronDensity(int Z);

    enum CROSS_SECTION_TYPE_LIST {PHOTOELECTRIC=0, INCOHERENT=1, COHERENT=2, PAIRPRODUCTION=3, TRIPLETPRODUCTION=4};
    
    enum elementList {H=1,He=2,Li=3,Be=4,B=5,C=6,N=7,O=8,F=9,Ne=10,Na=11,Mg=12,Al=13,Si=14,P=15,S=16,Cl=17,Ar=18,K=19,Ca=20,
                      Sc=21,Ti=22,V=23,Cr=24,Mn=25,Fe=26,Co=27,Ni=28,Cu=29,Zn=30,Ga=31,Ge=32,As=33,Se=34,Br=35,Kr=36,Rb=37,Sr=38,Y=39,Zr=40,
                      Nb=41,Mo=42,Tc=43,Ru=44,Rh=45,Pd=46,Ag=47,Cd=48,In=49,Sn=50,Sb=51,Te=52,I=53,Xe=54,Cs=55,Ba=56,La=57,Ce=58,Pr=59,Nd=60,
					  Pm=61,Sm=62,Eu=63,Gd=64,Tb=65,Dy=66,Ho=67,Er=68,Tm=69,Yb=70,Lu=71,Hf=72,Ta=73,W=74,Re=75,Os=76,Ir=77,Pt=78,Au=79,Hg=80,
					  Tl=81,Pb=82,Bi=83,Po=84,At=85,Rn=86,Fr=87,Ra=88,Ac=89,Th=90,Pa=91,U=92,Np=93,Pu=94,Am=95,Cm=96,Bk=97,Cf=98,Es=99,Fm=100};

    int elementStringToAtomicNumber(string);

private:

    size_t findNextCapitalLetter(string str, int pos = 0);
    size_t findNextPlusSign(string str, int pos = 0);
    vector<string> splitIntoElements(string str);
    vector<string> splitIntoMaterials(string str);
    //bool hasPlusSign(string str);

    float logInterp(float interp_energy, float low_energy, float high_energy, float low_crossSect, float high_crossSect);

    int N_Photoelectric[101];
    vector<float*> logPhotoelectricCrossSections; // [iz*N_e + ie]
    vector<float*> logPhotoelectricEnergies;
    vector<float*> PhotoelectricEnergies;
    
    int N_Incoherent[101];
    vector<float*> logIncoherentCrossSections;
    vector<float*> logIncoherentEnergies;
    vector<float*> IncoherentEnergies;
    
    int N_Coherent[101];
    vector<float*> logCoherentCrossSections;
    vector<float*> logCoherentEnergies;
    vector<float*> CoherentEnergies;
    
    int N_PairProduction[101];
    vector<float*> logPairProductionCrossSections;
    vector<float*> logPairProductionEnergies;
    vector<float*> PairProductionEnergies;
    
    int N_TripletProduction[101];
    vector<float*> logTripletProductionCrossSections;
    vector<float*> logTripletProductionEnergies;
    vector<float*> TripletProductionEnergies;
};

#endif

