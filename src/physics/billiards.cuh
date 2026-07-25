////////////////////////////////////////////////////////////////////////////////
// Copyright 2025 Kyle Champley
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// Cuda header for Monte-Carlo simulation of x-ray interactions with matter
// Translated from CPU-based method implemented by Kyle Champley
////////////////////////////////////////////////////////////////////////////////

#ifndef __BILLIARDS_H
#define __BILLIARDS_H

#ifdef WIN32
#pragma once
#endif

#include <curand.h>
#include <curand_kernel.h>
#include "parameters.h"

/**
 * This header and associated source file provide implementions of CUDA-based Monte Carlo 
 * simulation of x-ray interactions with matter
 */

/**
 * There are a lot of lookup tables (LUTs) needed for these calculations, so to make this more managable
 * most of these LUTs are handle here in this class
 */
class scatterSimulationTables
{
public:
    // Constructor
    scatterSimulationTables();
    
    // Destructor
    ~scatterSimulationTables();

    /**
	 * \fn          initialize
	 * \brief       copies lookup tables to GPU and initializes constant memory parameters
 	 * \param[in]   params: pointer to parameters class
     * \param[in]   chemForm: the chemical formula of the object material
     * \param[in]   max_energy: the maximum energy of the source spectra (keV)
     * \param[in]   min_scatters: the minimum number of scattering events to record
     * \param[in]   max_scatters: the maximum number of scattering events to record
     * \param[in]   num_curand_override: if &gt; 0, use this many curandState entries instead of numRows*(numCols+padding)
	 */
    void initialize(
        parameters* params,
        const char* chemForm,
        int max_energy,
        int min_scatters = 1,
        int max_scatters = 10,
        int num_curand_override = 0);

    /**
	 * \fn          initialize_multimaterial
	 * \brief       copies multi-material lookup tables to GPU and initializes constant memory parameters
	 *
	 * In addition to the common tables built by initialize(), this sets up:
	 *   - the piece-wise linear transfer function T(.) knots from mu_ref/mu_peak
	 *   - the per-material PE/CS/RS LAC component 3D texture (energy x material x component)
	 *   - the total SMB basis function 1D textures b_L, b_H
	 *
	 * \param[in]   params: pointer to parameters class
	 * \param[in]   chemForms: array of chemical formulas of the basis materials
	 * \param[in]   num_materials: number of basis materials (2-4)
	 * \param[in]   max_energy: the maximum energy of the source spectra (keV)
	 * \param[in]   reference_energy: the (low) reference energy gammaL of the fL volume
	 * \param[in]   peak_energy: the (high) peak energy gammaH used for the transfer function
	 * \param[in]   densities: [num_materials] mass density (g/cm^3) of each basis material
	 * \param[in]   b_L: [max_energy+1] total SMB low basis function (1 keV bins)
	 * \param[in]   b_H: [max_energy+1] total SMB high basis function (1 keV bins)
	 * \param[in]   min_scatters, max_scatters, num_curand_override: see initialize()
	 *
	 * The transfer-function knots (mu_ref/mu_peak) and the per-material PE/CS/RS LAC component
	 * table are computed internally from the chemical formulas, densities, and cross sections.
	 */
    void initialize_multimaterial(
        parameters* params,
        const char** chemForms,
        int num_materials,
        int max_energy,
        float reference_energy,
        float peak_energy,
        float* densities,
        float* b_L,
        float* b_H,
        int min_scatters = 1,
        int max_scatters = 10,
        int num_curand_override = 0);

    /**
	 * \fn          clear
	 * \brief       frees up all cuda memory used by this class
	 */
    void clear();

private:
    // 3D Texture memory for differential Compton and Rayleigh Scatter [which,energy,angle]
    TEX_DATA d_differential_cross_sections_txt;
    TEX_ARRAY d_differential_cross_sections_array;

    // 1D Texture memory for Photoelectric cross section (g/mm^2)
    TEX_DATA d_sigma_PE_txt;
    TEX_ARRAY d_sigma_PE_array;

    // 1D Texture memory for Compton Scatter cross section (g/mm^2)
    TEX_DATA d_sigma_CS_txt;
    TEX_ARRAY d_sigma_CS_array;

    // 1D Texture memory for Rayleigh Scatter cross section (g/mm^2)
    TEX_DATA d_sigma_RS_txt;
    TEX_ARRAY d_sigma_RS_array;

    // Multi-material: per-material PE/CS/RS LAC components, 3D texture (energy x material x component)
    TEX_DATA d_mm_LAC_components_txt;
    TEX_ARRAY d_mm_LAC_components_array;

    // Multi-material: total SMB basis functions, 1D textures (1 keV bins)
    TEX_DATA d_mm_b_L_txt;
    TEX_ARRAY d_mm_b_L_array;
    TEX_DATA d_mm_b_H_txt;
    TEX_ARRAY d_mm_b_H_array;

    // Multi-material: per-material differential angle distributions, one 3D texture per scatter
    // component (angle x energy x material)
    TEX_DATA d_mm_dsigma_Compton_txt;
    TEX_ARRAY d_mm_dsigma_Compton_array;
    TEX_DATA d_mm_dsigma_Rayleigh_txt;
    TEX_ARRAY d_mm_dsigma_Rayleigh_array;

    // The states for the random number generator for each cuda thread
    curandState* dev_states;

    float KNconstant; // = CLASSICAL_ELECTRON_RADIUS*CLASSICAL_ELECTRON_RADIUS*AVOGANDROS_NUMBER
    float two_PI_KNconstant; // 2*PI*KNconstant

    /**
	 * \fn          KleinNishinaCrossSection
	 * \brief       calculates the Klein-Nishina Cross Section
     * \param[in]   gamma: the energy (keV) of the photon
     * \return      returns the Klein-Nishina Cross Section
	 */
    float KleinNishinaCrossSection(const float gamma);

    /**
	 * \fn          KleinNishinaDistribution
	 * \brief       calculates the Differential Klein-Nishina Cross Section
     * \param[in]   gamma: the energy (keV) of the photon
     * \param[in]   theta: the angle (radians) of the scattering event
     * \return      returns the Differential Klein-Nishina Cross Section
	 */
    float KleinNishinaDistribution(const float gamma, const float theta);
};

/**
 * \fn          scatterSimulation
 * \brief       main C++ function that launches the Monte Carlo scatter simulation
 * \param[in]   params: pointer to parameters class
 * \param[in]   t: pointer to array where simulated scatter transmission is stored
 * \param[in]   f: pointer to mass density (g/mm^3) volume
 * \param[in]   source: pointer to source spectral model
 * \param[in]   energies: pointer to array of energy bins (keV) of the source spectra
 * \param[in]   N_energies: number of energy bins
 * \param[in]   detector: pointer to the detector response
 * \param[in]   chemForm: the chemical formula of the object material
 * \param[in]   data_on_cpu: true if the data is on the cpu, false if it is on the gpu
 * \param[in]   num_photons_per_pixel: the number of photon to simulate per detector pixel
 * \param[in]   min_scatters: the minimum number of scattering events to record
 * \param[in]   max_scatters: the maximum number of scattering events to record
 * \return      returns if successful, false otherwise
 */
bool scatterSimulation(parameters* params, float* t, float* f, float* source, float* energies, int N_energies, float* detector, const char* chemForm, bool data_on_cpu, int num_photons_per_pixel, int min_scatters, int max_scatters);

/**
 * \fn          scatterSimulation_multimaterial
 * \brief       multi-material Monte Carlo scatter simulation (see notes in billiards.cu)
 * \param[in]   params: pointer to parameters class
 * \param[in]   t: pointer to array where simulated scatter transmission is stored
 * \param[in]   f: fL, the Linear Attenuation Coefficient (LAC) volume at reference_energy
 * \param[in]   source, energies, N_energies, detector: source/detector model (see scatterSimulation)
 * \param[in]   reference_energy: the (low) reference energy gammaL of the fL volume
 * \param[in]   chemForms: array of chemical formulas of the basis materials
 * \param[in]   num_materials: number of basis materials (2-4)
 * \param[in]   densities: [num_materials] mass density (g/cm^3) of each basis material
 * \param[in]   b_L: [max_energy+1] total SMB low basis function (1 keV bins)
 * \param[in]   b_H: [max_energy+1] total SMB high basis function (1 keV bins)
 * \param[in]   data_on_cpu, num_photons_per_pixel, min_scatters, max_scatters: see scatterSimulation
 * \return      returns true if successful, false otherwise
 *
 * The transfer-function knots and per-material LAC component tables are computed internally from
 * the chemical formulas, densities, and cross sections; only the (SVD-derived) SMB basis functions
 * b_L/b_H are supplied by the caller.
 */
bool scatterSimulation_multimaterial(parameters* params, float* t, float* f, float* source, float* energies, int N_energies, float* detector, float reference_energy, const char** chemForms, int num_materials, float* densities, float* b_L, float* b_H, bool data_on_cpu, int num_photons_per_pixel, int min_scatters, int max_scatters);

/**
 * \fn          detectorScatterSimulation
 * \brief       main C++ function that launches the Monte Carlo detector scatter simulation
 * \param[in]   params: pointer to parameters class
 * \param[in]   thickness: the thickness of the scintillator (mm)
 * \param[in]   mass_density: bulk mass density of the scintillator (g/cm^3); used with compound cross sections
 * \param[in]   source: pointer to source spectral model
 * \param[in]   energies: pointer to array of energy bins (keV) of the source spectra
 * \param[in]   N_energies: number of energy bins
 * \param[in]   chemForm: chemical formula of the scintillator (e.g. for compound cross sections and LUTs)
 * \param[in]   num_photons: the number of photons to simulate
 * \param[in]   max_scatters: cap on stored interactions per photon history
 * \param[in]   events: (host) array of size num_photons * max_scatters * 4, each 4-tuple
 *               stores (x, y, z) position (mm) in a local scintillator frame and energy (keV) deposited
 * \param[in]   direction: the direction of the x-ray flux
 * \return      returns if successful, false otherwise
 */
bool detectorScatterSimulation(
    parameters* params,
    float thickness,
    float mass_density,
    float* source,
    float* energies,
    int N_energies,
    const char* chemForm,
    float* events,
    int num_photons,
    int max_scatters,
    float* direction = nullptr);


#endif
