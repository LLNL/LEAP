#ifndef XSCATTER_H
#define XSCATTER_H

#ifdef WIN32
#pragma once
#endif

#include "xsec.h"

class xscatter
{
public:
    xscatter();
    ~xscatter();

	bool isInitialized();

	bool init();

	double KleinNishinaDistribution(double gamma_in, double theta_in);
	double KleinNishinaCrossSection(double);

	float incoherentScatterDistribution(int Z, float gamma_in, float theta_in);
	float coherentScatterDistribution(int Z, float gamma_in, double theta_in);

	float incoherentScatterDistribution(float Z, float gamma_in, float theta_in);
	float coherentScatterDistribution(float Z, float gamma_in, double theta_in);

	float incoherentScatterDistribution(float* chemForm, float gamma_in, float theta_in);
	float coherentScatterDistribution(float* chemForm, float gamma_in, float theta_in);

	float incoherentScatterDistribution(const char* chemForm, float gamma_in, float theta_in);
	float coherentScatterDistribution(const char* chemForm, float gamma_in, float theta_in);

	// Normalization Factors
	float incoherentScatterDistribution_normalizationFactor(int Z, float gamma_in);
	float coherentScatterDistribution_normalizationFactor(int Z, float gamma_in);

	float incoherentScatterDistribution_normalizationFactor(float Z, float gamma_in);
	float coherentScatterDistribution_normalizationFactor(float Z, float gamma_in);

	float incoherentScatterDistribution_normalizationFactor(float* chemForm, float gamma_in);
	float coherentScatterDistribution_normalizationFactor(float* chemForm, float gamma_in);

	float incoherentScatterDistribution_normalizationFactor(const char* chemForm, float gamma_in);
	float coherentScatterDistribution_normalizationFactor(const char* chemForm, float gamma_in);

	struct header_s {
		int Z; //atomic number
		int A; //atomic mass number
		int Yi; //incident_particle designator
		int Yo; //outgoing_particle designator
		double AW; //atomic mass (amu)
		string  date;
		int Iflag;
		int C; // reaction descriptor
		int I; //reaction property
		int S; // reaction modifier
		double X1; //subshell designator.
	};

	struct data {
		double energy;
		double value;
	};

    struct reaction_record_s {
        header_s header;
        vector<data> data_record;
    };

    struct IncoherentAndCoherentScatterFunctions_elem_s {
        reaction_record_s incoherent_scatter_function;
        reaction_record_s form_factor;
        reaction_record_s imag_anomalous_scattering_factor;
        reaction_record_s real_anomalous_scattering_factor;
    };

	struct element_property_s {
		unsigned int Z;
		float std_atomic_weight;
		float density;   ///g/cm^3
		string symbol;
	};

	static const int nElements = 100;
	static const element_property_s elementProperty_List[nElements];

private:

	double Smatrix(double x, int Z);

	double getCoherentFormFactor(double gamma_in, double theta_in, int Z);
	double getCoherentFormFactor(double x, int Z);

	xsec xsecTables;

    vector<IncoherentAndCoherentScatterFunctions_elem_s> scatterFunctions;

	double KNconstant;
	double two_PI_KNconstant;

	double relec;
	double relec2;
};

#endif
