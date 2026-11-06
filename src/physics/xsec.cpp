
#include "xsec.h"
#include "xsec_raw.h"
#include <cstring>
#include <math.h>
#include <algorithm>
#include <stdexcept>

const float atomicMass[101] = {0.0, 1.008000e+00, 4.002600e+00, 6.939000e+00, 9.012200e+00, 1.081100e+01, 1.201100e+01, 1.400700e+01, 1.599900e+01, 1.899800e+01, 2.017900e+01,
														  2.299000e+01, 2.431200e+01, 2.698200e+01, 2.808600e+01, 3.097400e+01, 3.206400e+01, 3.545300e+01, 3.994800e+01, 3.910200e+01, 4.008000e+01,
														  4.495800e+01, 4.790000e+01, 5.094200e+01, 5.199600e+01, 5.493800e+01, 5.584700e+01, 5.893300e+01, 5.871000e+01, 6.354000e+01, 6.537000e+01,
														  6.972000e+01, 7.259000e+01, 7.492200e+01, 7.896000e+01, 7.990900e+01, 8.380000e+01, 8.547000e+01, 8.762000e+01, 8.890500e+01, 9.122000e+01,
														  9.290600e+01, 9.594000e+01, 9.900000e+01, 1.017000e+02, 1.029100e+02, 1.064000e+02, 1.078700e+02, 1.124000e+02, 1.148200e+02, 1.186900e+02,
														  1.217500e+02, 1.276000e+02, 1.269000e+02, 1.313000e+02, 1.329100e+02, 1.373400e+02, 1.389100e+02, 1.401200e+02, 1.409100e+02, 1.442400e+02,
														  1.450000e+02, 1.503500e+02, 1.519600e+02, 1.572500e+02, 1.589200e+02, 1.625000e+02, 1.649300e+02, 1.672600e+02, 1.689300e+02, 1.730400e+02,
														  1.749700e+02, 1.784900e+02, 1.809500e+02, 1.838500e+02, 1.862000e+02, 1.902000e+02, 1.922000e+02, 1.950900e+02, 1.969700e+02, 2.005900e+02,
														  2.043700e+02, 2.071900e+02, 2.089800e+02, 2.090000e+02, 2.100000e+02, 2.220000e+02, 2.230000e+02, 2.260000e+02, 2.270000e+02, 2.320400e+02,
														  2.330000e+02, 2.380500e+02, 2.370000e+02, 2.390500e+02, 2.420000e+02, 2.470000e+02, 2.470000e+02, 2.510000e+02, 2.520000e+02, 2.570000e+02};

const float electronDensities[101] = { 0.0, 8.916666e-05, 8.919202e-05, 2.308690e-01, 8.202215e-01, 1.096106e+00, 1.131463e+00, 6.249875e-04, 7.145447e-04, 8.034530e-04, 4.459586e-04,
                                                                4.645933e-01, 8.588351e-01, 1.300337e+00, 1.161433e+00, 1.065410e+00, 1.032934e+00, 1.541139e-03, 8.037098e-04, 4.188532e-01, 7.734530e-01,
                                                                1.396170e+00, 2.085177e+00, 2.758627e+00, 3.314101e+00, 3.385635e+00, 3.665801e+00, 4.077512e+00, 4.245546e+00, 4.089392e+00, 3.273520e+00,
                                                                2.625129e+00, 2.346549e+00, 2.523825e+00, 1.937690e+00, 3.324407e-03, 1.603675e-03, 6.632034e-01, 1.101575e+00, 1.960418e+00, 2.852883e+00,
                                                                3.781995e+00, 4.474046e+00, 4.994949e+00, 5.369125e+00, 5.426586e+00, 5.196617e+00, 4.574951e+00, 3.693950e+00, 3.119578e+00, 3.079451e+00,
                                                                2.802801e+00, 2.542947e+00, 2.059023e+00, 2.421158e-03, 7.750733e-01, 1.427115e+00, 2.525218e+00, 2.755538e+00, 2.809524e+00, 2.870216e+00,
                                                                3.037379e+00, 3.076289e+00, 2.173657e+00, 3.215425e+00, 3.365750e+00, 3.472615e+00, 3.572819e+00, 3.685807e+00, 3.807193e+00, 2.722492e+00,
                                                                3.996565e+00, 5.369040e+00, 6.718662e+00, 7.768289e+00, 8.466703e+00, 9.018507e+00, 8.981998e+00, 8.576042e+00, 7.748794e+00, 5.402463e+00,
                                                                4.645105e+00, 4.492012e+00, 3.871189e+00, 3.745837e+00, 3.642857e+00, 3.769279e-03, 1.950673e+00, 1.946903e+00, 3.948150e+00, 4.545768e+00,
                                                                6.002875e+00, 7.323672e+00, 7.946203e+00, 7.801548e+00, 5.366322e+00, 5.250850e+00, 5.497976e+00, 5.466135e+00, 5.500000e+00, 5.447471e+00 };

const char elementSymbols[101][3] = { {'-'},{'H'},{'H','e'},{'L','i'},{'B','e'},{'B'},{'C'},{'N'},{'O'},{'F'},{'N','e'},{'N','a'},{'M','g'},{'A','l'},{'S','i'},{'P'},{'S'},{'C','l'},{'A','r'},{'K'},{'C','a'},
                      {'S','c'},{'T','i'},{'V'},{'C','r'},{'M','n'},{'F','e'},{'C','o'},{'N','i'},{'C','u'},{'Z','n'},{'G','a'},{'G','e'},{'A','s'},{'S','e'},{'B','r'},{'K','r'},{'R','b'},{'S','r'},{'Y'},{'Z','r'},
                      {'N','b'},{'M','o'},{'T','c'},{'R','u'},{'R','h'},{'P','d'},{'A','g'},{'C','d'},{'I','n'},{'S','n'},{'S','b'},{'T','e'},{'I'},{'X','e'},{'C','s'},{'B','a'},{'L','a'},{'C','e'},{'P','r'},{'N','d'},
                      {'P','m'},{'S','m'},{'E','u'},{'G','d'},{'T','b'},{'D','y'},{'H','o'},{'E','r'},{'T','m'},{'Y','b'},{'L','u'},{'H','f'},{'T','a'},{'W'},{'R','e'},{'O','s'},{'I','r'},{'P','t'},{'A','u'},{'H','g'},
                      {'T','l'},{'P','b'},{'B','i'},{'P','o'},{'A','t'},{'R','n'},{'F','r'},{'R','a'},{'A','c'},{'T','h'},{'P','a'},{'U'},{'N','p'},{'P','u'},{'A','m'},{'C','m'},{'B','k'},{'C','f'},{'E','s'},{'F','m'} };

xsec::xsec()
{
    logPhotoelectricCrossSections.clear();
    logPhotoelectricEnergies.clear();
    PhotoelectricEnergies.clear();
    
    logIncoherentCrossSections.clear();
    logIncoherentEnergies.clear();
    IncoherentEnergies.clear();
    
    logCoherentCrossSections.clear();
    logCoherentEnergies.clear();
    CoherentEnergies.clear();
    
    logPairProductionCrossSections.clear();
    logPairProductionEnergies.clear();
    PairProductionEnergies.clear();
    
    logTripletProductionCrossSections.clear();
    logTripletProductionEnergies.clear();
    TripletProductionEnergies.clear();
}

xsec::~xsec()
{
    clearAll();
}

bool xsec::clearAll()
{
    // PHOTOELECTRIC
    for (int i = 0; i < int(logPhotoelectricCrossSections.size()); i++)
    {
        if (logPhotoelectricCrossSections[i] != NULL)
            free(logPhotoelectricCrossSections[i]);
        logPhotoelectricCrossSections[i] = NULL;
    }
    logPhotoelectricCrossSections.clear();
    
    for (int i = 0; i < int(logPhotoelectricEnergies.size()); i++)
    {
        if (logPhotoelectricEnergies[i] != NULL)
            free(logPhotoelectricEnergies[i]);
        logPhotoelectricEnergies[i] = NULL;
    }
    logPhotoelectricEnergies.clear();
    
    for (int i = 0; i < int(PhotoelectricEnergies.size()); i++)
    {
        if (PhotoelectricEnergies[i] != NULL)
            free(PhotoelectricEnergies[i]);
        PhotoelectricEnergies[i] = NULL;
    }
    PhotoelectricEnergies.clear();
    
    // INCOHERENT
    for (int i = 0; i < int(logIncoherentCrossSections.size()); i++)
    {
        if (logIncoherentCrossSections[i] != NULL)
            free(logIncoherentCrossSections[i]);
        logIncoherentCrossSections[i] = NULL;
    }
    logIncoherentCrossSections.clear();

    for (int i = 0; i < int(logIncoherentEnergies.size()); i++)
    {
        if (logIncoherentEnergies[i] != NULL)
            free(logIncoherentEnergies[i]);
        logIncoherentEnergies[i] = NULL;
    }
    logIncoherentEnergies.clear();
    
    for (int i = 0; i < int(IncoherentEnergies.size()); i++)
    {
        if (IncoherentEnergies[i] != NULL)
            free(IncoherentEnergies[i]);
        IncoherentEnergies[i] = NULL;
    }
    IncoherentEnergies.clear();
    
    // COHERENT
    for (int i = 0; i < int(logCoherentCrossSections.size()); i++)
    {
        if (logCoherentCrossSections[i] != NULL)
            free(logCoherentCrossSections[i]);
        logCoherentCrossSections[i] = NULL;
    }
    logCoherentCrossSections.clear();

    for (int i = 0; i < int(logCoherentEnergies.size()); i++)
    {
        if (logCoherentEnergies[i] != NULL)
            free(logCoherentEnergies[i]);
        logCoherentEnergies[i] = NULL;
    }
    logCoherentEnergies.clear();

    for (int i = 0; i < int(CoherentEnergies.size()); i++)
    {
        if (CoherentEnergies[i] != NULL)
            free(CoherentEnergies[i]);
        CoherentEnergies[i] = NULL;
    }
    CoherentEnergies.clear();
    
    // PAIR PAIRPRODUCTION
    for (int i = 0; i < int(logPairProductionCrossSections.size()); i++)
    {
        if (logPairProductionCrossSections[i] != NULL)
            free(logPairProductionCrossSections[i]);
        logPairProductionCrossSections[i] = NULL;
    }
    logPairProductionCrossSections.clear();

    for (int i = 0; i < int(logPairProductionEnergies.size()); i++)
    {
        if (logPairProductionEnergies[i] != NULL)
            free(logPairProductionEnergies[i]);
        logPairProductionEnergies[i] = NULL;
    }
    logPairProductionEnergies.clear();
    
    for (int i = 0; i < int(PairProductionEnergies.size()); i++)
    {
        if (PairProductionEnergies[i] != NULL)
            free(PairProductionEnergies[i]);
        PairProductionEnergies[i] = NULL;
    }
    PairProductionEnergies.clear();

    // TRIPLET PRODUCTION
    for (int i = 0; i < int(logTripletProductionCrossSections.size()); i++)
    {
        if (logTripletProductionCrossSections[i] != NULL)
            free(logTripletProductionCrossSections[i]);
        logTripletProductionCrossSections[i] = NULL;
    }
    logTripletProductionCrossSections.clear();

    for (int i = 0; i < int(logTripletProductionEnergies.size()); i++)
    {
        if (logTripletProductionEnergies[i] != NULL)
            free(logTripletProductionEnergies[i]);
        logTripletProductionEnergies[i] = NULL;
    }
    logTripletProductionEnergies.clear();

    for (int i = 0; i < int(TripletProductionEnergies.size()); i++)
    {
        if (TripletProductionEnergies[i] != NULL)
            free(TripletProductionEnergies[i]);
        TripletProductionEnergies[i] = NULL;
    }
    TripletProductionEnergies.clear();

    return true;
}

bool xsec::init()
{
    if (logPhotoelectricCrossSections.size() == 0 || logIncoherentCrossSections.size() == 0 || logCoherentCrossSections.size() == 0 || logPairProductionCrossSections.size() == 0 || logTripletProductionCrossSections.size() == 0)
        clearAll();
    else
        return true;
    
	//printf("loading hard-coded cross section tables!\n");
	xsec_raw hardCodedTables;
	int* tempI = (int*) &hardCodedTables.data[0];
	float* tempF = (float*) &hardCodedTables.data[sizeof(int)];

	int64 curOffset = 0;
    
    for (int iz = 0; iz < 100; iz++)
    {
        double Z = double(iz+1);
        double c = 0.6022169 / Z;
        //c = 0.1*0.6022169 /atomicMass[iz+1];
        c = 0.6022169 / atomicMass[iz + 1];
        
        // Photoelectric
		tempI = (int*) &hardCodedTables.data[curOffset]; curOffset += sizeof(int);
		N_Photoelectric[iz] = *tempI;

        logPhotoelectricCrossSections.push_back((float*)malloc(sizeof(float)*N_Photoelectric[iz]));
        logPhotoelectricEnergies.push_back((float*)malloc(sizeof(float)*N_Photoelectric[iz]));
        PhotoelectricEnergies.push_back((float*)malloc(sizeof(float)*N_Photoelectric[iz]));
        
		for (int i = 0; i < N_Photoelectric[iz]; i++)
		{
			tempF = (float*) &hardCodedTables.data[curOffset]; curOffset += sizeof(float);
			PhotoelectricEnergies[iz][i] = *tempF;
		}

		for (int i = 0; i < N_Photoelectric[iz]; i++)
		{
			tempF = (float*) &hardCodedTables.data[curOffset]; curOffset += sizeof(float);
			logPhotoelectricCrossSections[iz][i] = *tempF;
		}

        for (int i = 0; i < N_Photoelectric[iz]; i++)
        {
            logPhotoelectricCrossSections[iz][i] = log(c*logPhotoelectricCrossSections[iz][i]);
            logPhotoelectricEnergies[iz][i] = log(PhotoelectricEnergies[iz][i]);
        }

        // Incoherent
		tempI = (int*) &hardCodedTables.data[curOffset]; curOffset += sizeof(int);
		N_Incoherent[iz] = *tempI;

		logIncoherentCrossSections.push_back((float*)malloc(sizeof(float)*N_Incoherent[iz]));
        logIncoherentEnergies.push_back((float*)malloc(sizeof(float)*N_Incoherent[iz]));
        IncoherentEnergies.push_back((float*)malloc(sizeof(float)*N_Incoherent[iz]));

		for (int i = 0; i < N_Incoherent[iz]; i++)
		{
			tempF = (float*) &hardCodedTables.data[curOffset]; curOffset += sizeof(float);
			IncoherentEnergies[iz][i] = *tempF;
		}

		for (int i = 0; i < N_Incoherent[iz]; i++)
		{
			tempF = (float*) &hardCodedTables.data[curOffset]; curOffset += sizeof(float);
			logIncoherentCrossSections[iz][i] = *tempF;
		}

        for (int i = 0; i < N_Incoherent[iz]; i++)
        {
            logIncoherentCrossSections[iz][i] = log(c*logIncoherentCrossSections[iz][i]);
            logIncoherentEnergies[iz][i] = log(IncoherentEnergies[iz][i]);
        }
        
        //Coherent
		tempI = (int*) &hardCodedTables.data[curOffset]; curOffset += sizeof(int);
		N_Coherent[iz] = *tempI;

		logCoherentCrossSections.push_back((float*)malloc(sizeof(float)*N_Coherent[iz]));
        logCoherentEnergies.push_back((float*)malloc(sizeof(float)*N_Coherent[iz]));
        CoherentEnergies.push_back((float*)malloc(sizeof(float)*N_Coherent[iz]));

		for (int i = 0; i < N_Coherent[iz]; i++)
		{
			tempF = (float*) &hardCodedTables.data[curOffset]; curOffset += sizeof(float);
			CoherentEnergies[iz][i] = *tempF;
		}

		for (int i = 0; i < N_Coherent[iz]; i++)
		{
			tempF = (float*) &hardCodedTables.data[curOffset]; curOffset += sizeof(float);
			logCoherentCrossSections[iz][i] = *tempF;
		}

        for (int i = 0; i < N_Coherent[iz]; i++)
        {
            logCoherentCrossSections[iz][i] = log(c*logCoherentCrossSections[iz][i]);
            logCoherentEnergies[iz][i] = log(CoherentEnergies[iz][i]);
        }
        
        //Pair Production
		tempI = (int*) &hardCodedTables.data[curOffset]; curOffset += sizeof(int);
		N_PairProduction[iz] = *tempI;

		logPairProductionCrossSections.push_back((float*)malloc(sizeof(float)*N_PairProduction[iz]));
        logPairProductionEnergies.push_back((float*)malloc(sizeof(float)*N_PairProduction[iz]));
        PairProductionEnergies.push_back((float*)malloc(sizeof(float)*N_PairProduction[iz]));

		for (int i = 0; i < N_PairProduction[iz]; i++)
		{
			tempF = (float*) &hardCodedTables.data[curOffset]; curOffset += sizeof(float);
			PairProductionEnergies[iz][i] = *tempF;
		}

		for (int i = 0; i < N_PairProduction[iz]; i++)
		{
			tempF = (float*) &hardCodedTables.data[curOffset]; curOffset += sizeof(float);
			logPairProductionCrossSections[iz][i] = *tempF;
		}

        for (int i = 0; i < N_PairProduction[iz]; i++)
        {
            logPairProductionCrossSections[iz][i] = log(c*logPairProductionCrossSections[iz][i]);
            logPairProductionEnergies[iz][i] = log(PairProductionEnergies[iz][i]);
        }
        
        //Triplet Production
		tempI = (int*) &hardCodedTables.data[curOffset]; curOffset += sizeof(int);
		N_TripletProduction[iz] = *tempI;

		logTripletProductionCrossSections.push_back((float*)malloc(sizeof(float)*N_TripletProduction[iz]));
        logTripletProductionEnergies.push_back((float*)malloc(sizeof(float)*N_TripletProduction[iz]));
        TripletProductionEnergies.push_back((float*)malloc(sizeof(float)*N_TripletProduction[iz]));

		for (int i = 0; i < N_TripletProduction[iz]; i++)
		{
			tempF = (float*) &hardCodedTables.data[curOffset]; curOffset += sizeof(float);
			TripletProductionEnergies[iz][i] = *tempF;
		}

		for (int i = 0; i < N_TripletProduction[iz]; i++)
		{
			tempF = (float*) &hardCodedTables.data[curOffset]; curOffset += sizeof(float);
			logTripletProductionCrossSections[iz][i] = *tempF;
		}

        for (int i = 0; i < N_TripletProduction[iz]; i++)
        {
            logTripletProductionCrossSections[iz][i] = log(c*logTripletProductionCrossSections[iz][i]);
            logTripletProductionEnergies[iz][i] = log(TripletProductionEnergies[iz][i]);
        }
    }
    
	return true;
}

float xsec::logInterp(float interp_energy, float low_energy, float high_energy, float low_crossSect, float high_crossSect)
{
    if (isinf(low_crossSect) == true)
        return 0.0;
    else
    {
        float m = (high_crossSect - low_crossSect) / (high_energy - low_energy);
        return exp(low_crossSect + m*(interp_energy - low_energy));
    }
}

float* xsec::parseMaterialText(const char* material)
{
    /*
    try
    {
        float num = std::stof(material);
        printf("num = %f\n", num);
        
        float* elementCount_weighted = (float*)calloc(size_t(101), sizeof(float));
        elementCount_weighted[int(floor(num))] = 1.0-(num-floor(num));
        elementCount_weighted[int(ceil(num))] = num-floor(num);
        return elementCount_weighted;
    }
    catch (const std::invalid_argument& e)
    {
    //*/
        string material_str = material;

        // Remove spaces
        std::string::iterator end_pos = std::remove(material_str.begin(), material_str.end(), ' ');
        material_str.erase(end_pos, material_str.end());
        
        if (findNextPlusSign(material_str) == string::npos)
            return parseChemicalFormula(material);
        else
        {
            vector<string> fractionsAndFormulas = splitIntoMaterials(material_str);
            if (fractionsAndFormulas.size() == 0)
                return NULL;

            //for (int n = 0; n < int(fractionsAndFormulas.size()); n++)
            //    printf("%s\n", fractionsAndFormulas[n]);

            // Set chemFormulas and fracs
            vector<string> chemFormulas;
            float* fracs = (float*)malloc(size_t(int(fractionsAndFormulas.size())) * sizeof(float));
            for (int n = 0; n < int(fractionsAndFormulas.size()); n++)
            {
                //element = elementString.substr(0, 2);
                //countStr = elementString.substr(2, elementString.size() - 2);

                if (fractionsAndFormulas[n][0] == '*')
                {
                    free(fracs);
                    return NULL;
                }

                bool isValid = false;
                for (int i = 0; i < int(fractionsAndFormulas[n].size()); i++)
                {
                    if (fractionsAndFormulas[n][i] >= 'A' && fractionsAndFormulas[n][i] <= 'Z')
                    {
                        if (i > 0)
                        {
                            if (fractionsAndFormulas[n][i - 1] == '*')
                            {
                                string frac_str = fractionsAndFormulas[n].substr(0, i - 1);
                                fracs[n] = std::stof(frac_str);
                            }
                            else
                            {
                                string frac_str = fractionsAndFormulas[n].substr(0, i);
                                fracs[n] = std::stof(frac_str);
                            }
                        }
                        else
                            fracs[n] = 1.0;
                        isValid = true;
                        string chemFormula = fractionsAndFormulas[n].substr(i, fractionsAndFormulas[n].size() - i);
                        chemFormulas.push_back(chemFormula);
                        break;
                    }
                }
                if (isValid == false)
                {
                    free(fracs);
                    return NULL;
                }
            }

            //for (int n = 0; n < int(fractionsAndFormulas.size()); n++)
            //    printf("%s: %f\n", chemFormulas[n], fracs[n]);

            float accum = 0.0;
            for (int n = 0; n < int(chemFormulas.size()); n++)
            {
                accum += fracs[n];
                if (fracs[n] <= 0.0)
                {
                    accum = -1.0;
                    break;
                }
            }
            if (accum <= 0.0)
            {
                free(fracs);
                return NULL;
            }

            float* elementCount_weighted = (float*)calloc(size_t(101), sizeof(float));
            for (int n = 0; n < int(chemFormulas.size()); n++)
            {
                fracs[n] = fracs[n] / accum;
                float* elementCount = parseChemicalFormula(chemFormulas[n].c_str());
                if (elementCount == NULL)
                {
                    free(fracs);
                    free(elementCount_weighted);
                    return NULL;
                }
                else
                {
                    // accumulate
                    float denom = 0.0;
                    for (int Z = 1; Z <= 100; Z++)
                    {
                        if (elementCount[Z] > 0.0)
                            denom += getAtomicMass(Z) * elementCount[Z];
                    }
                    for (int Z = 1; Z <= 100; Z++)
                    {
                        if (elementCount[Z] > 0.0)
                        {
                            elementCount_weighted[Z] += fracs[n] / denom * elementCount[Z];
                        }
                    }
                }
                free(elementCount);
            }

            /*
            for (int Z = 1; Z <= 100; Z++)
            {
                if (elementCount_weighted[Z] > 0.0)
                    printf("%d: %f\n", Z, elementCount_weighted[Z]);
            }
            //*/

            free(fracs);
            return elementCount_weighted;
        }
    //}
}

float* xsec::parseChemicalFormula(const char* chemForm)
{
    float* elementCount = (float*)calloc(size_t(101), sizeof(float));

    // parse chemForm
    string chemForm_str = chemForm;
    vector<string> elementStrings = splitIntoElements(chemForm_str);
    for (int i = 0; i < int(elementStrings.size()); i++)
    {
        if (elementStrings[i].size() == 0)
        {
            printf("Error parsing chemical formula!\n");
            free(elementCount);
            return NULL;
        }

        string elementString = elementStrings[i];
        string element;
        string countStr;

        if (elementString.size() == 1)
        {
            element = elementString;
        }
        else
        {
            if (isalpha(elementString[1]))
            {
                element = elementString.substr(0, 2);
                countStr = elementString.substr(2, elementString.size() - 2);
            }
            else
            {
                element = elementString.substr(0, 1);
                countStr = elementString.substr(1, elementString.size() - 1);
            }
        }

        int Z = elementStringToAtomicNumber(element);
        float count = 1.0;
        if (countStr.size() > 0)
            count = std::stof(countStr);
        elementCount[Z] += count;
    }
    return elementCount;
}

float xsec::sigma(const char* chemForm, float theEnergy, int which)
{
    //float* elementCount = parseChemicalFormula(chemForm);
    float* elementCount = parseMaterialText(chemForm);
    if (elementCount == NULL)
        return 0.0;

    float retVal = sigma(elementCount, theEnergy, which);
    free(elementCount);
    return retVal;
}

float xsec::sigma(float* chemForm, float theEnergy, int which)
{
    float sumZ = 0.0;
    float retVal = 0.0;
    for (int i = 1; i <= 100; i++)
    {
        if (chemForm[i] > 0.0)
        {
            sumZ += chemForm[i] * atomicMass[i];
            retVal += atomicMass[i] * chemForm[i] * sigma(i, theEnergy, which);
        }
    }
    if (sumZ > 0.0)
        retVal /= sumZ;
    else
        retVal = 0.0;

    return retVal;
}

//####
float xsec::sigma_e(const char* chemForm, float theEnergy, int which)
{
    //float* elementCount = parseChemicalFormula(chemForm);
    float* elementCount = parseMaterialText(chemForm);
    if (elementCount == NULL)
        return 0.0;

    float retVal = sigma_e(elementCount, theEnergy, which);
    free(elementCount);
    return retVal;
}

float xsec::sigma_e(float* chemForm, float theEnergy, int which)
{
    float sumZ = 0.0;
    float retVal = 0.0;
    for (int i = 1; i <= 100; i++)
    {
        if (chemForm[i] > 0.0)
        {
            //sumZ += chemForm[i] * atomicMass[i];
            //retVal += atomicMass[i] * chemForm[i] * sigma_e(i, theEnergy, which);
            sumZ += chemForm[i] * float(i);
            retVal += float(i) * chemForm[i] * sigma_e(i, theEnergy, which);
        }
    }
    if (sumZ > 0.0)
        retVal /= sumZ;
    else
        retVal = 0.0;

    return retVal;
}
//####

float xsec::sigma(float Ze, float theEnergy, int which)
{
    if (fabs(Ze - floor(Ze + 0.5)) < 1.0e-8)
        return sigma(int(floor(Ze + 0.5)), theEnergy, which);
    int Z_lo = int(floor(Ze));
    int Z_hi = int(ceil(Ze));
    float dZ = Ze - float(Z_lo);
    return (1.0 - dZ) * sigma(Z_lo, theEnergy, which) + dZ * sigma(Z_hi, theEnergy, which);
}

float xsec::sigma_e(float Ze, float theEnergy, int which)
{
    if (fabs(Ze - floor(Ze + 0.5)) < 1.0e-8)
        return sigma_e(int(floor(Ze + 0.5)), theEnergy, which);
    int Z_lo = int(floor(Ze));
    int Z_hi = int(ceil(Ze));
    float dZ = Ze - float(Z_lo);
    return (1.0 - dZ) * sigma_e(Z_lo, theEnergy, which) + dZ * sigma_e(Z_hi, theEnergy, which);
}

float xsec::sigma_e(int Z, float theEnergy, int which)
{
    if (1 <= Z && Z <= 100)
        return sigma(Z, theEnergy, which) * atomicMass[Z] / float(Z);
    else
        return 0.0;
}

float xsec::sigma(int Z, float theEnergy, int which)
{
    /*
    string chemForm_str = "H2O3Al2.5";
    vector<string> elementStrings = splitIntoElements(chemForm_str);
    printf("elementStrings.size = %d\n", int(elementStrings.size()));
    for (int i = 0; i < int(elementStrings.size()); i++)
        printf("%d: %s\n", i, elementStrings[i].c_str());
    //*/

    if (logPhotoelectricCrossSections.size() == 0 || logIncoherentCrossSections.size() == 0 || logCoherentCrossSections.size() == 0 || logPairProductionCrossSections.size() == 0 || logTripletProductionCrossSections.size() == 0)
        init();
    
    if (Z < 1 || Z > 100 || theEnergy < 1.0 || theEnergy > float(MAX_ENERGY))
        return 0.0;
    else if (theEnergy > float(MAX_XRAY_ENERGY))
        return sigma(float(MAX_XRAY_ENERGY), Z, which)*(theEnergy+1.0-float(MAX_XRAY_ENERGY)) - sigma(float(MAX_XRAY_ENERGY-1), Z, which)*(theEnergy-float(MAX_XRAY_ENERGY));
    int n_min = which;
    int n_max = which;
    if (which < 0 || which > 4)
    {
        n_min = 0;
        n_max = 4;
    }
    
    Z -= 1;
    
    float retVal = 0.0;
    for (int n = n_min; n <= n_max; n++)
    {
        int N_energySamples = 0;
        float* logCrossSection = NULL;
        float* logEnergies = NULL;
        float* energies = NULL;
        switch (n)
        {
            case PHOTOELECTRIC:
                N_energySamples = N_Photoelectric[Z];
                logCrossSection = logPhotoelectricCrossSections[Z];
                logEnergies = logPhotoelectricEnergies[Z];
                energies = PhotoelectricEnergies[Z];
                break;
            case INCOHERENT:
                N_energySamples = N_Incoherent[Z];
                logCrossSection = logIncoherentCrossSections[Z];
                logEnergies = logIncoherentEnergies[Z];
                energies = IncoherentEnergies[Z];
                break;
            case COHERENT:
                N_energySamples = N_Coherent[Z];
                logCrossSection = logCoherentCrossSections[Z];
                logEnergies = logCoherentEnergies[Z];
                energies = CoherentEnergies[Z];
                break;
            case PAIRPRODUCTION:
                N_energySamples = N_PairProduction[Z];
                logCrossSection = logPairProductionCrossSections[Z];
                logEnergies = logPairProductionEnergies[Z];
                energies = PairProductionEnergies[Z];
                break;
            case TRIPLETPRODUCTION:
                N_energySamples = N_TripletProduction[Z];
                logCrossSection = logTripletProductionCrossSections[Z];
                logEnergies = logTripletProductionEnergies[Z];
                energies = TripletProductionEnergies[Z];
                break;
            default:
                break;
        }
        int index = 0;
        float interp_crossSect = 0.0;
        for(; index < N_energySamples; index++)
        {
            if(theEnergy < energies[index])
                break;
        }
        if (index > 0 && index < N_energySamples)
            interp_crossSect = logInterp(log(theEnergy), logEnergies[index-1], logEnergies[index], logCrossSection[index-1], logCrossSection[index]);
        else
            interp_crossSect = 0.0;
        retVal += interp_crossSect;
    }
    return retVal;
}

size_t xsec::findNextPlusSign(string str, int pos)
{
    for (int i = pos; i < int(str.size()); i++)
    {
        if (str[i] == '+')
            return size_t(i);
    }
    return string::npos;
}

size_t xsec::findNextCapitalLetter(string str, int pos)
{
    for (int i = pos; i < int(str.size()); i++)
    {
        if (str[i] >= 'A' && str[i] <= 'Z')
            return size_t(i);
        //if (isupper(str[i]))
        //    return size_t(i);
    }
    return string::npos;
}

vector<string> xsec::splitIntoMaterials(string str)
{
    int ntokens = 100;
    vector<string> tokens;
    if (str.empty())
        return tokens;

    //size_t pos = findNextPlusSign(str, 0);
    //if (pos != 0) // first letter must be capitalized
    //    return tokens;
    size_t pos = 0;

    while (1)
    {
        size_t nextPos = findNextPlusSign(str, pos + 1);
        if (nextPos == string::npos)
        {
            string temp = str.substr(pos, str.size() - pos);
            tokens.push_back(temp);
            break;
        }
        else
        {
            string temp = str.substr(pos, nextPos - pos);
            tokens.push_back(temp);
            pos = nextPos;
        }

        ntokens--;
        if (ntokens <= 0)
            break;
    }

    return tokens;
}

vector<string> xsec::splitIntoElements(string str)
{
    int ntokens = 100;
    vector<string> tokens;
    if (str.empty())
        return tokens;

    size_t pos = findNextCapitalLetter(str, 0);
    if (pos != 0) // first letter must be capitalized
        return tokens;

    while (1)
    {
        size_t nextPos = findNextCapitalLetter(str, pos+1);
        if (nextPos == string::npos)
        {
            string temp = str.substr(pos, str.size() - pos);
            tokens.push_back(temp);
            break;
        }
        else
        {
            string temp = str.substr(pos, nextPos - pos);
            tokens.push_back(temp);
            pos = nextPos;
        }

        ntokens--;
        if (ntokens <= 0)
            break;
    }

    return tokens;
}

int xsec::elementStringToAtomicNumber(string str)
{
    for (int i = 1; i <= 100; i++)
    {
        //printf("%s\n", elementSymbols[i]);
        if (strcmp(str.c_str(), elementSymbols[i]) == 0)
            return i;
    }
    //exit(1);
    return 0;
}

float xsec::sigmaPE(const char* chemForm, float theEnergy)
{
    return sigma(chemForm, theEnergy, PHOTOELECTRIC);
}

float xsec::sigmaPE(float Ze, float theEnergy)
{
    return sigma(Ze, theEnergy, PHOTOELECTRIC);
}

float xsec::sigmaPE(int Z, float theEnergy)
{
    return sigma(Z, theEnergy, PHOTOELECTRIC);
}

float xsec::sigmaCS(const char* chemForm, float theEnergy)
{
    return sigma(chemForm, theEnergy, INCOHERENT);
}

float xsec::sigmaCS(float Ze, float theEnergy)
{
    return sigma(Ze, theEnergy, INCOHERENT);
}

float xsec::sigmaCS(int Z, float theEnergy)
{
    return sigma(Z, theEnergy, INCOHERENT);
}

float xsec::sigmaRS(const char* chemForm, float theEnergy)
{
    return sigma(chemForm, theEnergy, COHERENT);
}

float xsec::sigmaRS(float Ze, float theEnergy)
{
    return sigma(Ze, theEnergy, COHERENT);
}

float xsec::sigmaRS(int Z, float theEnergy)
{
    return sigma(Z, theEnergy, COHERENT);
}

float xsec::sigmaPP(const char* chemForm, float theEnergy)
{
    return sigma(chemForm, theEnergy, PAIRPRODUCTION);
}

float xsec::sigmaPP(float Ze, float theEnergy)
{
    return sigma(Ze, theEnergy, PAIRPRODUCTION);
}

float xsec::sigmaPP(int Z, float theEnergy)
{
    return sigma(Z, theEnergy, PAIRPRODUCTION);
}

float xsec::sigmaTP(const char* chemForm, float theEnergy)
{
    return sigma(chemForm, theEnergy, TRIPLETPRODUCTION);
}

float xsec::sigmaTP(float Ze, float theEnergy)
{
    return sigma(Ze, theEnergy, TRIPLETPRODUCTION);
}

float xsec::sigmaTP(int Z, float theEnergy)
{
    return sigma(Z, theEnergy, TRIPLETPRODUCTION);
}

float xsec::mu(int Z, float theEnergy, int which)
{
    return sigma(Z, theEnergy, which) * getMassDensity(Z);
}

float xsec::getAtomicMass(int Z)
{
    if (1 <= Z && Z <= 100)
        return atomicMass[Z];
    else
        return 0.0;
}

float xsec::getMassDensity(int Z)
{
    //sigma_e = sigma * atomicMass[Z] / float(Z)
    //rho_e sigma_e = rho sigma
    // rho = rho_e sigma_e / sigma
    // rho = rho_e atomicMass[Z] / float(Z)
    if (1 <= Z && Z <= 100)
        return electronDensities[Z] * atomicMass[Z] / float(Z);
    else
        return 0.0;
}

float xsec::getElectronDensity(int Z)
{
    if (1 <= Z && Z <= 100)
        return electronDensities[Z];
    else
        return 0.0;
}

float xsec::sigma_inv(const char* chemForm, float val, int which)
{
    float sigma_cur = sigma(chemForm, float(1), which);
    if (val >= sigma_cur)
        return 1.0;
    float sigma_next = sigma_cur;
    for (int i = 1; i < MAX_XRAY_ENERGY; i++)
    {
        sigma_next = sigma(chemForm, float(i+1), which);
        if (sigma_cur >= val && val >= sigma_next)
        {
            float d = (sigma_cur - val) / (sigma_cur - sigma_next);
            return float(i) + d;
        }

        sigma_cur = sigma_next;
    }
    return float(MAX_XRAY_ENERGY);
}

float xsec::sigma_inv(float Ze, float val, int which)
{
    float sigma_cur = sigma(Ze, float(1), which);
    if (val >= sigma_cur)
        return 1.0;
    float sigma_next = sigma_cur;
    for (int i = 1; i < MAX_XRAY_ENERGY; i++)
    {
        sigma_next = sigma(Ze, float(i + 1), which);
        if (sigma_cur >= val && val >= sigma_next)
        {
            float d = (sigma_cur - val) / (sigma_cur - sigma_next);
            return float(i) + d;
        }

        sigma_cur = sigma_next;
    }
    return float(MAX_XRAY_ENERGY);
}

float xsec::sigma_inv(int Z, float val, int which)
{
    float sigma_cur = sigma(Z, float(1), which);
    if (val >= sigma_cur)
        return 1.0;
    float sigma_next = sigma_cur;
    for (int i = 1; i < MAX_XRAY_ENERGY; i++)
    {
        sigma_next = sigma(Z, float(i + 1), which);
        if (sigma_cur >= val && val >= sigma_next)
        {
            float d = (sigma_cur - val) / (sigma_cur - sigma_next);
            return float(i) + d;
        }

        sigma_cur = sigma_next;
    }
    return float(MAX_XRAY_ENERGY);
}
