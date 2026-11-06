////////////////////////////////////////////////////////////////////////////////
// Copyright 2025 Kyle Champley
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// Cuda header for Monte-Carlo simulation of x-ray interactions with matter
// Translated from CPU-based method implemented by Kyle Champley
////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <time.h>
#include <algorithm>
#include "log.h"
#include "cuda_runtime.h"
#include "cuda_utils.h"
#include "billiards.cuh"
#include "xrayphysics_c_interface.h"
#include "xsec.h"
#include "denoise/noise_filters.cuh"
#include "cpu_utils.h"
#include "fbp/ramp_filter_cpu.h"

#define PHOTON_BLOCK 128 // original value
//#define PHOTON_BLOCK 64
//#define PHOTON_BLOCK 32 // definitely slower

__constant__ float d_ELECTRON_REST_MASS_ENERGY;
__constant__ float d_KNconstant;
__constant__ float d_two_PI_KNconstant;
__constant__ float d_relec2;
__constant__ float d_Ze;
__constant__ int d_min_scatters;
__constant__ int d_max_scatters;
__constant__ float d_T_angle_inv;
// Number of cumulative-probability (u) samples in the inverse-CDF angle tables. The device samplers
// map a uniform draw u in (0,1] to a texture x-coordinate u*d_invCDF_scale (see randomComptonScatterAngle).
__constant__ float d_invCDF_scale;

__constant__ float3 d_AABB_lo;
__constant__ float3 d_AABB_hi;

// Homogeneous scintillator slab: mass density (g/mm^3 on device) and thickness (mm) for detectorScatterSimulation
__constant__ float d_slab_rho;
__constant__ float d_slab_T;
__constant__ float3 d_direction;

__device__ TEX_DATA dsigma_dOmega;
__device__ TEX_DATA d_sigma_PE;
__device__ TEX_DATA d_sigma_CS;
__device__ TEX_DATA d_sigma_RS;
__device__ curandState* d_states;

///////////////////////////////////////////////////////////////////////////////////////
// Multi-material scatter simulation (see scatterSimulation_multimaterial)
//
// The object is described by f = fL, the Linear Attenuation Coefficient (LAC) volume at
// the reference (low) energy gammaL. A piece-wise linear transfer function T(.) (the
// "change_energy_mm" knots, built from the basis-material LAC values mu_ref[i] at gammaL
// and mu_peak[i] at gammaH) maps fL -> LAC at the high energy gammaH. The total LAC at an
// arbitrary energy gamma is b_L(gamma)*fL + b_H(gamma)*T(fL), where b_L, b_H are the
// Synthesized Monochromatic Basis (SMB) functions (see xrayphysics.convert_to_smb).
//
// The per-material PE/CS/RS LAC components are stored in a 3D texture indexed
// (energy, material_index, component) so the interaction-type sampling can use the local
// material (derived from fL via the knot x-values) at the photon's current energy.
///////////////////////////////////////////////////////////////////////////////////////

// Piece-wise linear transfer function knots T(.) (same layout/semantics as change_energy
// in the polychromatic projector): .w is the lowest knot, then .x, .y, .z increasing.
__constant__ float4 d_mm_mu_ref;
__constant__ float4 d_mm_mu_slopes;
__constant__ float4 d_mm_mu_offsets;
__constant__ int d_mm_num_materials;

// Knot x-values (LAC of each basis material at gammaL) used to map fL -> fractional
// material index for the per-material component texture lookup (up to 4 materials).
__constant__ float4 d_mm_material_knots;

// Precomputed reciprocals of the knot gaps: .x=1/(x1-x0), .y=1/(x2-x1), .z=1/(x3-x2).
// Lets materialIndex_mm interpolate with multiplies instead of per-photon divisions.
__constant__ float4 d_mm_material_knot_inv_gaps;

// Precomputed reciprocals of the fL volume voxel sizes (1/T.x, 1/T.y, 1/T.z). Lets the
// world->index conversion in sample_fL_mm use multiplies instead of per-sample divisions.
__constant__ float4 d_mm_inv_T_f;

// Per-material LAC components, 3D texture indexed (x=energy, y=material, z=component)
// where component 0=PE, 1=CS, 2=RS. Values are LAC (1/mm) sampled in 1 keV bins.
__device__ TEX_DATA d_mm_LAC_components;

// Total SMB basis functions b_L(gamma), b_H(gamma), 1D textures sampled in 1 keV bins.
__device__ TEX_DATA d_mm_b_L;
__device__ TEX_DATA d_mm_b_H;

// Per-material differential scatter angle distributions, one 3D texture per scatter component
// indexed (x=angle, y=energy, z=material). Keeping Compton and Rayleigh in separate textures means
// the linear filtering on the material (z) axis can only ever blend two materials of the *same*
// component, so there is no chance of cross-component contamination at the material-block edges.
// Each (material, energy) distribution is normalized to a max of 1 over angle for rejection sampling.
__device__ TEX_DATA d_mm_dsigma_Compton;
__device__ TEX_DATA d_mm_dsigma_Rayleigh;


///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
// scatterSimulationTables
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
scatterSimulationTables::scatterSimulationTables()
{
    d_differential_cross_sections_txt = {};
    d_differential_cross_sections_array = NULL;

    d_sigma_PE_txt = {};
    d_sigma_PE_array = NULL;

    d_sigma_CS_txt = {};
    d_sigma_CS_array = NULL;

    d_sigma_RS_txt = {};
    d_sigma_RS_array = NULL;

    d_mm_LAC_components_txt = {};
    d_mm_LAC_components_array = NULL;

    d_mm_b_L_txt = {};
    d_mm_b_L_array = NULL;

    d_mm_b_H_txt = {};
    d_mm_b_H_array = NULL;

    d_mm_dsigma_Compton_txt = {};
    d_mm_dsigma_Compton_array = NULL;

    d_mm_dsigma_Rayleigh_txt = {};
    d_mm_dsigma_Rayleigh_array = NULL;

    dev_states = NULL;
}

scatterSimulationTables::~scatterSimulationTables()
{
    clear();
}

void scatterSimulationTables::clear()
{
    if (d_differential_cross_sections_array != NULL)
    {
        freeTexture(d_differential_cross_sections_array, d_differential_cross_sections_txt);
    }
    d_differential_cross_sections_txt = {};
    d_differential_cross_sections_array = NULL;

    if (d_sigma_PE_array != NULL)
    {
        freeTexture(d_sigma_PE_array, d_sigma_PE_txt);
    }
    d_sigma_PE_txt = {};
    d_sigma_PE_array = NULL;

    if (d_sigma_CS_array != NULL)
    {
        freeTexture(d_sigma_CS_array, d_sigma_CS_txt);
    }
    d_sigma_CS_txt = {};
    d_sigma_CS_array = NULL;

    if (d_sigma_RS_array != NULL)
    {
        freeTexture(d_sigma_RS_array, d_sigma_RS_txt);
    }
    d_sigma_RS_txt = {};
    d_sigma_RS_array = NULL;

    if (d_mm_LAC_components_array != NULL)
    {
        freeTexture(d_mm_LAC_components_array, d_mm_LAC_components_txt);
    }
    d_mm_LAC_components_txt = {};
    d_mm_LAC_components_array = NULL;

    if (d_mm_b_L_array != NULL)
    {
        freeTexture(d_mm_b_L_array, d_mm_b_L_txt);
    }
    d_mm_b_L_txt = {};
    d_mm_b_L_array = NULL;

    if (d_mm_b_H_array != NULL)
    {
        freeTexture(d_mm_b_H_array, d_mm_b_H_txt);
    }
    d_mm_b_H_txt = {};
    d_mm_b_H_array = NULL;

    if (d_mm_dsigma_Compton_array != NULL)
    {
        freeTexture(d_mm_dsigma_Compton_array, d_mm_dsigma_Compton_txt);
    }
    d_mm_dsigma_Compton_txt = {};
    d_mm_dsigma_Compton_array = NULL;

    if (d_mm_dsigma_Rayleigh_array != NULL)
    {
        freeTexture(d_mm_dsigma_Rayleigh_array, d_mm_dsigma_Rayleigh_txt);
    }
    d_mm_dsigma_Rayleigh_txt = {};
    d_mm_dsigma_Rayleigh_array = NULL;

    if (dev_states != NULL)
    {
        cudaFree(dev_states);
    }
    dev_states = NULL;
}

float scatterSimulationTables::KleinNishinaCrossSection(const float gamma_in)
{
    // total ops: (18,1)
    const float alpha = gamma_in / ELECTRON_REST_MASS_ENERGY; // 1 op
    const float one_plus_two_alpha = 1.0f+2.0f*alpha; // 2 ops
    return ((1.0f+one_plus_two_alpha)/(2.0f*one_plus_two_alpha*one_plus_two_alpha) + ((alpha*alpha-1.0f-one_plus_two_alpha)*log(one_plus_two_alpha) + 4.0f*alpha)/(2.0f*alpha*alpha*alpha)) * two_PI_KNconstant; // 13 ops
}

float scatterSimulationTables::KleinNishinaDistribution(const float gamma_in, const float theta_in)
{
    // normalized distribution
    const float cos_theta = cosf(theta_in);
    const float P = 1.0f / (1.0f + (gamma_in / ELECTRON_REST_MASS_ENERGY)*(1.0f-cos_theta));
    return 0.5f*P*P*(P + 1.0f/P - 1.0f + cos_theta*cos_theta) / (KleinNishinaCrossSection(gamma_in) / KNconstant);
}

// Convert a discrete angular distribution p[a] (a = 0..N_angle-1, proportional to dsigma/dOmega*sin,
// covering theta in [0, PI] with bin width T_angle) into an inverse-CDF table for sampling.
//
// invCDF[i] (i = 0..N_u-1) is the angle theta whose cumulative probability equals (i+0.5)/N_u, so
// that a uniform draw u in (0,1] maps to an angle via a single lookup at texture x-coordinate u*N_u.
// The device sampler therefore needs one RNG draw and one (linearly interpolated) texture fetch,
// with no rejection loop and no warp divergence. Degenerate rows (zero total mass, e.g. gamma == 0)
// are filled with 0.
static void buildInverseCDF(const float* p, int N_angle, float T_angle, float* invCDF, int N_u)
{
    double total = 0.0;
    for (int b = 0; b < N_angle; b++)
        if (p[b] > 0.0f)
            total += double(p[b]);

    if (total <= 0.0)
    {
        for (int i = 0; i < N_u; i++)
            invCDF[i] = 0.0f;
        return;
    }

    const double inv_total = 1.0 / total;

    // March the cumulative distribution (node n is at theta = n*T_angle, mass_n is the probability of
    // bin n spanning [theta_n, theta_{n+1}]) together with the uniform u-grid.
    int n = 0;
    double C_n = 0.0;                                              // cumulative prob at node n
    double mass_n = (p[0] > 0.0f ? double(p[0]) : 0.0) * inv_total; // mass of bin n
    for (int i = 0; i < N_u; i++)
    {
        const double P = (double(i) + 0.5) / double(N_u);
        while (C_n + mass_n < P && n < N_angle - 1)
        {
            C_n += mass_n;
            n++;
            mass_n = (p[n] > 0.0f ? double(p[n]) : 0.0) * inv_total;
        }
        double frac = (mass_n > 0.0) ? (P - C_n) / mass_n : 0.0;
        if (frac < 0.0) frac = 0.0;
        if (frac > 1.0) frac = 1.0;
        invCDF[i] = float((double(n) + frac) * double(T_angle));
    }
}

void scatterSimulationTables::initialize(
    parameters* params,
    const char* chemForm,
    int max_energy,
    int min_scatters,
    int max_scatters,
    int num_curand_override)
{
    //clear();

    int extraCols = zeroPadForOffsetScan_numberOfColsToAdd(params);

    KNconstant = CLASSICAL_ELECTRON_RADIUS*CLASSICAL_ELECTRON_RADIUS*AVOGANDROS_NUMBER;
    two_PI_KNconstant = 2.0*PI*KNconstant;

    int N_energy = max_energy + 1;
    int N_angle = 180*10;
    float T_angle = PI/float(N_angle);

    float cm_sq_to_mm_sq = 100.0;

    float* sigmaPE = new float[N_energy];
    float* sigmaCS = new float[N_energy];
    float* sigmaRS = new float[N_energy];
    // dsigma_dOmega now stores inverse-CDF tables (x-axis = cumulative probability u), so the device
    // samplers can draw an angle with a single texture lookup. cdf_tmp is scratch for the inversion.
    float* differential_cross_sections = new float[2*N_energy*N_angle];
    float* cdf_tmp = new float[N_angle];
    for (int iE = 0; iE < N_energy; iE++)
    {
        float gamma = float(iE);

        sigmaPE[iE] = sigmaCompoundPE(chemForm, gamma) * cm_sq_to_mm_sq;
        sigmaCS[iE] = sigmaCompoundCS(chemForm, gamma) * cm_sq_to_mm_sq;
        sigmaRS[iE] = sigmaCompoundRS(chemForm, gamma) * cm_sq_to_mm_sq;

        //printf("%f: %f\n", gamma, sigmaPE[iE]+sigmaCS[iE]+sigmaRS[iE]);

        float* dCompton = &differential_cross_sections[0*N_angle*N_energy + iE*N_angle];
        float* dRayleigh = &differential_cross_sections[1*N_angle*N_energy + iE*N_angle];
        for (int iAngle = 0; iAngle < N_angle; iAngle++)
        {
            float theta = float(iAngle)*T_angle*180.0/PI;
            float sin_theta = sinf(theta*PI/180.0);

            //float Compton = KleinNishinaDistribution(gamma, theta) * sin_theta;
            float Compton = incoherentScatterDistributionCompound(chemForm, gamma, theta) * sin_theta;
            float Rayleigh = coherentScatterDistributionCompound(chemForm, gamma, theta) * sin_theta;

            if (gamma == 0.0)
            {
                Compton = 0.0;
                Rayleigh = 0.0;
            }

            dCompton[iAngle] = Compton;
            dRayleigh[iAngle] = Rayleigh;
        }
        // Replace each angular pdf with its inverse CDF (theta as a function of cumulative
        // probability u). N_u == N_angle keeps the texture shape and memory unchanged.
        buildInverseCDF(dCompton, N_angle, T_angle, cdf_tmp, N_angle);
        for (int k = 0; k < N_angle; k++)
            dCompton[k] = cdf_tmp[k];
        buildInverseCDF(dRayleigh, N_angle, T_angle, cdf_tmp, N_angle);
        for (int k = 0; k < N_angle; k++)
            dRayleigh[k] = cdf_tmp[k];
    }
    delete [] cdf_tmp;

    /*
    if (has_nan(sigmaPE, N_energy))
        printf("sigmaPE is corrupted!\n");
    if (has_nan(sigmaCS, N_energy))
        printf("sigmaCS is corrupted!\n");
    if (has_nan(sigmaRS, N_energy))
        printf("sigmaRS is corrupted!\n");
    if (has_nan(differential_cross_sections, 2, N_energy, N_angle))
        printf("differential_cross_sections is corrupted!\n");
    //*/

    float Ze = effectiveZ(chemForm, 10.0, float(max_energy), 0.0);

    float temp = ELECTRON_REST_MASS_ENERGY;
    cudaMemcpyToSymbol(d_ELECTRON_REST_MASS_ENERGY, &temp, sizeof(float));
    cudaMemcpyToSymbol(d_KNconstant, &KNconstant, sizeof(float));
    cudaMemcpyToSymbol(d_two_PI_KNconstant, &two_PI_KNconstant, sizeof(float));

    temp = 1.0 / T_angle;
    cudaMemcpyToSymbol(d_T_angle_inv, &temp, sizeof(float));

    // Number of u samples in the inverse-CDF angle tables (the texture x-dimension == N_angle).
    temp = float(N_angle);
    cudaMemcpyToSymbol(d_invCDF_scale, &temp, sizeof(float));

    float relec = 0.28179403267;
    float relec2 = relec * relec;
    cudaMemcpyToSymbol(d_relec2, &relec2, sizeof(float));

    cudaMemcpyToSymbol(d_Ze, &Ze, sizeof(float));

    cudaMemcpyToSymbol(d_min_scatters, &min_scatters, sizeof(int));
    cudaMemcpyToSymbol(d_max_scatters, &max_scatters, sizeof(int));

    d_differential_cross_sections_array = loadTexture_from_cpu(d_differential_cross_sections_txt, differential_cross_sections, make_int3(2, N_energy, N_angle), true, true);
    cudaMemcpyToSymbol(dsigma_dOmega, &d_differential_cross_sections_txt, sizeof(TEX_DATA));

    d_sigma_PE_array = loadTexture1D(d_sigma_PE_txt, sigmaPE, N_energy, true, true);
    cudaMemcpyToSymbol(d_sigma_PE, &d_sigma_PE_txt, sizeof(TEX_DATA));

    d_sigma_CS_array = loadTexture1D(d_sigma_CS_txt, sigmaCS, N_energy, true, true);
    cudaMemcpyToSymbol(d_sigma_CS, &d_sigma_CS_txt, sizeof(TEX_DATA));

    d_sigma_RS_array = loadTexture1D(d_sigma_RS_txt, sigmaRS, N_energy, true, true);
    cudaMemcpyToSymbol(d_sigma_RS, &d_sigma_RS_txt, sizeof(TEX_DATA));

    float vox_eps = min(params->voxelWidth, params->voxelHeight)*0.01;
    float3 AABB_lo = make_float3(params->x_0()+vox_eps, params->y_0()+vox_eps, params->z_0()+vox_eps);
    float3 AABB_hi = make_float3(params->x_f()-vox_eps, params->y_f()-vox_eps, params->z_f()-vox_eps);
    cudaMemcpyToSymbol(d_AABB_lo, &AABB_lo, sizeof(float3));
    cudaMemcpyToSymbol(d_AABB_hi, &AABB_hi, sizeof(float3));

    {
        int n_states = params->numRows * (params->numCols + extraCols);
        if (num_curand_override > 0)
            n_states = num_curand_override;
        cudaMalloc(&dev_states, n_states * sizeof(curandState));
    }
    cudaMemcpyToSymbol(d_states, &dev_states, sizeof(curandState*));

    delete [] sigmaPE;
    delete [] sigmaCS;
    delete [] sigmaRS;
    delete [] differential_cross_sections;
}

void scatterSimulationTables::initialize_multimaterial(
    parameters* params,
    const char** chemForms,
    int num_materials,
    int max_energy,
    float reference_energy,
    float peak_energy,
    float* densities,
    float* b_L,
    float* b_H,
    int min_scatters,
    int max_scatters,
    int num_curand_override)
{
    // Build the common tables (KN constants, AABB, curand states, and the single-material 1D sigma
    // and dsigma_dOmega textures). The single-material sigma / dsigma_dOmega textures are not used
    // by the multi-material kernel (which uses the per-material d_mm_* tables built below), but the
    // rest of the GPU state (KN constants, angle binning, AABB, curand states) is shared.
    initialize(params, chemForms[0], max_energy, min_scatters, max_scatters, num_curand_override);

    const int N_energy = max_energy + 1;
    // num_materials is expected to be in [2, 4] (validated by the caller); the transfer-function
    // knots are stored in a float4 so at most 4 basis materials are supported.

    // LAC values come from (mass density) * (compound mass cross section). To match both the
    // input volume f (= fL, the reconstruction in 1/mm) and the single-material scatter path,
    // everything is in mm units: densities are passed in g/mm^3 (leapctype.massDensity in its
    // default mm mode) and the cross sections returned by sigmaCompound are in cm^2/g. Multiplying
    // by 100 (cm^2 -> mm^2) turns (g/mm^3)*(cm^2/g) into LAC in mm^-1. This mirrors the
    // single-material scatterSimulationTables::initialize, which scales its sigma tables by
    // cm_sq_to_mm_sq = 100 and likewise expects a g/mm^3 density volume.
    const float mass_xsec_cm2g_to_mm_inv = 100.0f;

    // ---- Reference-energy LAC of each basis material, sorted ascending (knot ordering) ----
    int order[4] = {0, 1, 2, 3};
    float mu_ref_unsorted[4];
    for (int i = 0; i < num_materials; i++)
        mu_ref_unsorted[i] = densities[i] * sigmaCompound(chemForms[i], reference_energy) * mass_xsec_cm2g_to_mm_inv;
    // simple insertion sort of the index array by ascending reference LAC (num_materials <= 4)
    for (int i = 1; i < num_materials; i++)
    {
        int key = order[i];
        int j = i - 1;
        while (j >= 0 && mu_ref_unsorted[order[j]] > mu_ref_unsorted[key])
        {
            order[j + 1] = order[j];
            j--;
        }
        order[j + 1] = key;
    }

    // ---- Compute knots and per-material LAC component table in sorted material order ----
    float* mu_ref = new float[num_materials];
    float* mu_peak = new float[num_materials];
    // Per-material LAC components, layout [component][material][energy] (component 0=PE,1=CS,2=RS).
    float* LAC_components = new float[3 * num_materials * N_energy];
    for (int s = 0; s < num_materials; s++)
    {
        const char* chemForm = chemForms[order[s]];
        const float rho = densities[order[s]];
        mu_ref[s] = mu_ref_unsorted[order[s]];
        mu_peak[s] = rho * sigmaCompound(chemForm, peak_energy) * mass_xsec_cm2g_to_mm_inv;

        float* dPE = &LAC_components[(0 * num_materials + s) * N_energy];
        float* dCS = &LAC_components[(1 * num_materials + s) * N_energy];
        float* dRS = &LAC_components[(2 * num_materials + s) * N_energy];
        dPE[0] = 0.0f;
        dCS[0] = 0.0f;
        dRS[0] = 0.0f;
        for (int e = 1; e < N_energy; e++)
        {
            const float gamma = float(e);
            dPE[e] = rho * sigmaCompoundPE(chemForm, gamma) * mass_xsec_cm2g_to_mm_inv;
            dCS[e] = rho * sigmaCompoundCS(chemForm, gamma) * mass_xsec_cm2g_to_mm_inv;
            dRS[e] = rho * sigmaCompoundRS(chemForm, gamma) * mass_xsec_cm2g_to_mm_inv;
        }
    }

    // ---- Transfer function T(.) knots ----
    // Knots {(mu_ref[i], mu_peak[i])}: x = LAC at gammaL, y = LAC at gammaH.
    float4 mm_mu_ref;
    float4 mm_mu_slopes;
    float4 mm_mu_offsets;

    mm_mu_ref.w = mu_ref[0];
    mm_mu_slopes.w = (mu_ref[0] != 0.0f) ? (mu_peak[0] / mu_ref[0]) : 1.0f;
    mm_mu_offsets.w = 0.0f;

    mm_mu_ref.x = mu_ref[1];
    mm_mu_slopes.x = (mu_peak[1] - mu_peak[0]) / (mu_ref[1] - mu_ref[0]);
    mm_mu_offsets.x = mu_peak[1] - mm_mu_slopes.x * mu_ref[1];

    if (num_materials == 2)
    {
        mm_mu_slopes.y = mm_mu_slopes.x;
        mm_mu_slopes.z = mm_mu_slopes.x;
        mm_mu_offsets.y = mm_mu_offsets.x;
        mm_mu_offsets.z = mm_mu_offsets.x;
        mm_mu_ref.y = 1000.0f * mm_mu_ref.x;
        mm_mu_ref.z = mm_mu_ref.y;
    }
    else
    {
        mm_mu_ref.y = mu_ref[2];
        mm_mu_slopes.y = (mu_peak[2] - mu_peak[1]) / (mu_ref[2] - mu_ref[1]);
        mm_mu_offsets.y = mu_peak[2] - mm_mu_slopes.y * mu_ref[2];

        if (num_materials == 3)
        {
            mm_mu_slopes.z = mm_mu_slopes.y;
            mm_mu_offsets.z = mm_mu_offsets.y;
            mm_mu_ref.z = 1000.0f * mm_mu_ref.y;
        }
        else
        {
            mm_mu_ref.z = mu_ref[3];
            mm_mu_slopes.z = (mu_peak[3] - mu_peak[2]) / (mu_ref[3] - mu_ref[2]);
            mm_mu_offsets.z = mu_peak[3] - mm_mu_slopes.z * mu_ref[3];
        }
    }

    cudaMemcpyToSymbol(d_mm_mu_ref, &mm_mu_ref, sizeof(float4));
    cudaMemcpyToSymbol(d_mm_mu_slopes, &mm_mu_slopes, sizeof(float4));
    cudaMemcpyToSymbol(d_mm_mu_offsets, &mm_mu_offsets, sizeof(float4));
    cudaMemcpyToSymbol(d_mm_num_materials, &num_materials, sizeof(int));

    // Material knot x-values (LAC at gammaL) used to map fL -> fractional material index.
    float4 material_knots = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float* knot_ptr = &material_knots.x;
    for (int i = 0; i < num_materials; i++)
        knot_ptr[i] = mu_ref[i];
    cudaMemcpyToSymbol(d_mm_material_knots, &material_knots, sizeof(float4));

    // Precompute reciprocals of the knot gaps so materialIndex_mm avoids per-photon divisions.
    float4 material_knot_inv_gaps = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float* inv_gap_ptr = &material_knot_inv_gaps.x;
    for (int i = 0; i + 1 < num_materials; i++)
    {
        const float gap = knot_ptr[i + 1] - knot_ptr[i];
        inv_gap_ptr[i] = (gap > 0.0f) ? (1.0f / gap) : 0.0f;
    }
    cudaMemcpyToSymbol(d_mm_material_knot_inv_gaps, &material_knot_inv_gaps, sizeof(float4));

    // ---- Per-material LAC component 3D texture (x=energy, y=material, z=component) ----
    // Layout is [component][material][energy] (C-contiguous), which matches the
    // (depth=component, height=material, width=energy) ordering expected by loadTexture_from_cpu.
    d_mm_LAC_components_array = loadTexture_from_cpu(
        d_mm_LAC_components_txt, LAC_components, make_int3(3, num_materials, N_energy), true, true);
    cudaMemcpyToSymbol(d_mm_LAC_components, &d_mm_LAC_components_txt, sizeof(TEX_DATA));

    // ---- Total SMB basis functions b_L, b_H (1D textures, 1 keV bins) ----
    d_mm_b_L_array = loadTexture1D(d_mm_b_L_txt, b_L, N_energy, true, true);
    cudaMemcpyToSymbol(d_mm_b_L, &d_mm_b_L_txt, sizeof(TEX_DATA));

    d_mm_b_H_array = loadTexture1D(d_mm_b_H_txt, b_H, N_energy, true, true);
    cudaMemcpyToSymbol(d_mm_b_H, &d_mm_b_H_txt, sizeof(TEX_DATA));

    // ---- Per-material differential Compton/Rayleigh angle distributions ----
    // Mirrors the single-material dsigma_dOmega build in initialize(), but evaluated per basis
    // material (in the same sorted order as the LAC components / material knots) so the angle
    // sampling can interpolate over the fractional material index. Compton and Rayleigh are stored
    // in two separate textures, each laid out [material][energy][angle] (C-contiguous), matching the
    // (depth=num_materials, height=N_energy, width=N_angle) ordering expected by loadTexture_from_cpu.
    // Using one texture per component means the material-axis (depth) linear filtering can only ever
    // blend two materials of the same component. Each (material, energy) angular distribution is
    // stored as an inverse-CDF table (x-axis = cumulative probability u) so the device samplers draw
    // an angle with a single lookup instead of rejection sampling.
    const int N_angle = 180 * 10;
    const float T_angle = PI / float(N_angle);
    float* dCompton_mm = new float[num_materials * N_energy * N_angle];
    float* dRayleigh_mm = new float[num_materials * N_energy * N_angle];
    float* cdf_tmp = new float[N_angle];
    for (int s = 0; s < num_materials; s++)
    {
        const char* chemForm = chemForms[order[s]];
        for (int iE = 0; iE < N_energy; iE++)
        {
            const float gamma = float(iE);
            float* dCompton = &dCompton_mm[(s * N_energy + iE) * N_angle];
            float* dRayleigh = &dRayleigh_mm[(s * N_energy + iE) * N_angle];
            for (int iAngle = 0; iAngle < N_angle; iAngle++)
            {
                const float theta = float(iAngle) * T_angle * 180.0 / PI;
                const float sin_theta = sinf(theta * PI / 180.0);

                float Compton = incoherentScatterDistributionCompound(chemForm, gamma, theta) * sin_theta;
                float Rayleigh = coherentScatterDistributionCompound(chemForm, gamma, theta) * sin_theta;

                if (gamma == 0.0f)
                {
                    Compton = 0.0f;
                    Rayleigh = 0.0f;
                }

                dCompton[iAngle] = Compton;
                dRayleigh[iAngle] = Rayleigh;
            }
            // Replace each per-(material, energy) angular pdf with its inverse CDF so the device
            // samplers (randomComptonScatterAngle_mm / randomRayleighScatterAngle_mm) need only a
            // single texture lookup. Interpolating inverse CDFs across the material axis is a
            // quantile blend of the two bracketing materials' angular distributions.
            buildInverseCDF(dCompton, N_angle, T_angle, cdf_tmp, N_angle);
            for (int k = 0; k < N_angle; k++)
                dCompton[k] = cdf_tmp[k];
            buildInverseCDF(dRayleigh, N_angle, T_angle, cdf_tmp, N_angle);
            for (int k = 0; k < N_angle; k++)
                dRayleigh[k] = cdf_tmp[k];
        }
    }
    delete [] cdf_tmp;
    d_mm_dsigma_Compton_array = loadTexture_from_cpu(
        d_mm_dsigma_Compton_txt, dCompton_mm, make_int3(num_materials, N_energy, N_angle), true, true);
    cudaMemcpyToSymbol(d_mm_dsigma_Compton, &d_mm_dsigma_Compton_txt, sizeof(TEX_DATA));
    d_mm_dsigma_Rayleigh_array = loadTexture_from_cpu(
        d_mm_dsigma_Rayleigh_txt, dRayleigh_mm, make_int3(num_materials, N_energy, N_angle), true, true);
    cudaMemcpyToSymbol(d_mm_dsigma_Rayleigh, &d_mm_dsigma_Rayleigh_txt, sizeof(TEX_DATA));
    delete [] dCompton_mm;
    delete [] dRayleigh_mm;

    delete [] mu_ref;
    delete [] mu_peak;
    delete [] LAC_components;
}


///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
// Device Functions
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
/*
__device__ float KleinNishinaCrossSection(const float gamma_in)
{
    // total ops: (18,1)
    const float alpha = gamma_in / d_ELECTRON_REST_MASS_ENERGY; // 1 op
    const float one_plus_two_alpha = 1.0f+2.0f*alpha; // 2 ops
    return ((1.0f+one_plus_two_alpha)/(2.0f*one_plus_two_alpha*one_plus_two_alpha) + ((alpha*alpha-1.0f-one_plus_two_alpha)*log(one_plus_two_alpha) + 4.0f*alpha)/(2.0f*alpha*alpha*alpha)) * d_two_PI_KNconstant; // 13 ops
}

__device__ float KleinNishinaDistribution(const float gamma_in, const float theta_in)
{
    // normalized distribution
    const float cos_theta = cos(theta_in);
    const float P = 1.0f / (1.0f + (gamma_in / d_ELECTRON_REST_MASS_ENERGY)*(1.0f-cos_theta));
    return 0.5f*P*P*(P + 1.0f/P - 1.0f + cos_theta*cos_theta) / (KleinNishinaCrossSection(gamma_in) / d_KNconstant);
}
//*/

__device__ float fullLineIntegral(TEX_DATA mu, const int4 N, const float4 T, const float4 startVal, const float3 p, const float3 dst)
{
    // NOTE: assumes that T.x == T.y == T.z
    const float3 r = make_float3(dst.x - p.x, dst.y - p.y, dst.z - p.z);  // points from voxel to pixel

    if (fabsf(r.x) >= fabsf(r.y) && fabsf(r.x) >= fabsf(r.z))
    {
        // integral in x direction
        const float3 ip = make_float3((p.x - startVal.x) / T.x, (p.y - startVal.y) / T.y,
            (p.z - startVal.z) / T.z);

        const float3 ir = make_float3(r.x / fabsf(r.x), r.y / fabsf(r.x), r.z / fabsf(r.x));
        const int ix_start = max(0, min(N.x - 1, int(floor(0.5f + ip.x))));

        // The above nearest neighbor calculation will have move the "true" starting x position by a small
        // amount. Make sure this small shift is also accounted for in the y and z dimensions.
        // ip+ir*t = ix_start
        const float t = (ix_start - ip.x) / ir.x;
        const float iy_start = ip.y + t * ir.y;
        const float iz_start = ip.z + t * ir.z;

        float val = 0.0f;
        if (r.x > 0.0f)
        {
            if (ip.x >= float(N.x) - 0.5f) return 0.0f;
            int ix_max = min(N.x - 1, int(ceil((dst.x - startVal.x) / T.x)));

            val = TEX3D(mu, float(ix_start) + 0.5f, iy_start + 0.5f, iz_start + 0.5f) *
                ((float(ix_start) - 0.5f) - max(-0.5f, ip.x));

            const float iy_offset = iy_start - ir.y * float(ix_start) + 0.5f;
            const float iz_offset = iz_start - ir.z * float(ix_start) + 0.5f;

            for (int ix = ix_start; ix <= ix_max; ix++)
                val += TEX3D(mu, float(ix) + 0.5f, iy_offset + ir.y * float(ix), iz_offset + ir.z * float(ix));
        }
        else
        {
            if (ip.x <= -0.5f) return 0.0f;
            int ix_min = max(0, int(floor((dst.x - startVal.x) / T.x)));

            val = TEX3D(mu, float(ix_start) + 0.5f, iy_start + 0.5f, iz_start + 0.5f) *
                (min((float(N.x) - 0.5f), ip.x) - (float(ix_start) + 0.5f));

            const float iy_offset = iy_start + ir.y * float(ix_start) + 0.5f;
            const float iz_offset = iz_start + ir.z * float(ix_start) + 0.5f;
            for (int ix = ix_start; ix >= ix_min; ix--)
                val += TEX3D(mu, float(ix) + 0.5f, iy_offset - ir.y * float(ix), iz_offset - ir.z * float(ix));
        }
        return val * sqrtf(1.0f + ir.y * ir.y + ir.z * ir.z) * T.x;
    }
    else if (fabsf(r.y) >= fabsf(r.z))
    {
        // integral in y direction
        const float3 ip = make_float3((p.x - startVal.x) / T.x, (p.y - startVal.y) / T.y,
            (p.z - startVal.z) / T.z);

        const float3 ir = make_float3(r.x / fabsf(r.y), r.y / fabsf(r.y), r.z / fabsf(r.y));
        const int iy_start = max(0, min(N.y - 1, int(floor(0.5f + ip.y))));

        const float t = (iy_start - ip.y) / ir.y;
        const float ix_start = ip.x + t * ir.x;
        const float iz_start = ip.z + t * ir.z;

        float val = 0.0f;
        if (r.y > 0.0f)
        {
            if (ip.y >= float(N.y) - 0.5f) return 0.0f;
            int iy_max = min(N.y - 1, int(ceil((dst.y - startVal.y) / T.y)));

            val = TEX3D(mu, ix_start + 0.5f, float(iy_start) + 0.5f, iz_start + 0.5f) *
                ((float(iy_start) - 0.5f) - max(-0.5f, ip.y));

            const float ix_offset = ix_start - ir.x * float(iy_start) + 0.5f;
            const float iz_offset = iz_start - ir.z * float(iy_start) + 0.5f;
            //printf("p = %f, %f, %f to dst = %f, %f, %f\n", p.x, p.y, p.z, dst.x, dst.y, dst.z);
            //printf("istarts = %f, %d, %f\n", ix_start, iy_start, iz_start);
            for (int iy = iy_start; iy <= iy_max; iy++)
            {
                //printf("iy = %d: indices = %f, %f, %f; update: %f\n", iy, ix_offset + ir.x * float(iy)-0.5f, float(iy) + 0.5f - 0.5f, iz_offset + ir.z * float(iy) - 0.5f, TEX3D(mu, ix_offset + ir.x * float(iy), float(iy) + 0.5f, iz_offset + ir.z * float(iy)));
                val += TEX3D(mu, ix_offset + ir.x * float(iy), float(iy) + 0.5f, iz_offset + ir.z * float(iy));
            }
        }
        else
        {
            if (ip.y <= -0.5f) return 0.0f;
            int iy_min = max(0, int(floor((dst.y - startVal.y) / T.y)));

            val = TEX3D(mu, ix_start + 0.5f, iy_start + 0.5f, iz_start + 0.5f) *
                (min((float(N.y) - 0.5f), ip.y) - (float(iy_start) + 0.5f));

            const float ix_offset = ix_start + ir.x * float(iy_start) + 0.5f;
            const float iz_offset = iz_start + ir.z * float(iy_start) + 0.5f;
            for (int iy = iy_start; iy >= iy_min; iy--)
                val += TEX3D(mu, ix_offset - ir.x * float(iy), float(iy) + 0.5f, iz_offset - ir.z * float(iy));
        }
        //printf("left edge = %f\n", TEX3D(mu, -0.9f+0.5f, 0.5f, 0.5f));
        //printf("forward project rayWeight = sqrt(1.0 + (%f)^2 + (%f)^2) = %f\n", ir.x, ir.z, sqrt(1.0f + ir.x * ir.x + ir.z * ir.z));
        return val * sqrtf(1.0f + ir.x * ir.x + ir.z * ir.z) * T.y;
    }
    else
    {
        // integral in z direction
        const float3 ip = make_float3((p.x - startVal.x) / T.x, (p.y - startVal.y) / T.y,
            (p.z - startVal.z) / T.z);

        const float3 ir = make_float3(r.x / fabsf(r.z), r.y / fabsf(r.z), r.z / fabsf(r.z));
        const int iz_start = max(0, min(N.z - 1, int(floor(0.5f + ip.z))));

        const float t = (iz_start - ip.z) / ir.z;
        const float ix_start = ip.x + t * ir.x;
        const float iy_start = ip.y + t * ir.y;

        float val = 0.0f;
        if (r.z > 0.0f)
        {
            if (ip.z >= float(N.z) - 0.5f) return 0.0f;
            int iz_max = min(N.z - 1, int(ceil((dst.z - startVal.z) / T.z)));

            val = TEX3D(mu, ix_start + 0.5f, iy_start + 0.5f, iz_start + 0.5f) *
                ((float(iz_start) - 0.5f) - max(-0.5f, ip.z));

            const float ix_offset = ix_start - ir.x * float(iz_start) + 0.5f;
            const float iy_offset = iy_start - ir.y * float(iz_start) + 0.5f;
            for (int iz = iz_start; iz <= iz_max; iz++)
                val += TEX3D(mu, ix_offset + ir.x * float(iz), iy_offset + ir.y * float(iz), float(iz) + 0.5f);
        }
        else
        {
            if (ip.z <= -0.5f) return 0.0f;
            int iz_min = max(0, int(floor((dst.z - startVal.z) / T.z)));

            val = TEX3D(mu, ix_start + 0.5f, iy_start + 0.5f, float(iz_start) + 0.5f) *
                (min((float(N.z) - 0.5f), ip.z) - (float(iz_start) + 0.5f));

            const float ix_offset = ix_start + ir.x * float(iz_start) + 0.5f;
            const float iy_offset = iy_start + ir.y * float(iz_start) + 0.5f;
            for (int iz = iz_start; iz >= iz_min; iz--)
                val += TEX3D(mu, ix_offset - ir.x * float(iz), iy_offset - ir.y * float(iz), float(iz) + 0.5f);
        }
        return val * sqrtf(1.0f + ir.x * ir.x + ir.y * ir.y) * T.z;
    }
}

__device__ float3 divergentBeamTransform(TEX_DATA mu, const int4 N, const float4 T, const float4 startVal, const float3 p, const float3 r, const float rhoL)
{
    /*
    mu: mass density distribution
    N:  dimensions of mu
    T:  voxel pitch
    startVal: first samples in (x,y,z)
    p: starting position of line integral
    r: direction of line integral
    rhoL: stop line integral when this value is crossed

    returns the final destination
    */

    // NOTE: assumes that T.x == T.y == T.z
    const float3 ip = make_float3((p.x - startVal.x) / T.x, (p.y - startVal.y) / T.y,
        (p.z - startVal.z) / T.z);
    //const float3 r = make_float3(dst.x - p.x, dst.y - p.y, dst.z - p.z);  // points from voxel to pixel

    if (fabsf(r.x) >= fabsf(r.y) && fabsf(r.x) >= fabsf(r.z))
    {
        // integral in x direction
        const float3 ir = make_float3(r.x / fabsf(r.x), r.y / fabsf(r.x), r.z / fabsf(r.x));
        const float increment_amount = sqrtf(1.0f + ir.y * ir.y + ir.z * ir.z) * T.x;
        const float threshold = rhoL / increment_amount;

        const int ix_start = max(0, min(N.x - 1, int(floor(0.5f + ip.x))));

        // The above nearest neighbor calculation will have move the "true" starting x position by a small
        // amount. Make sure this small shift is also accounted for in the y and z dimensions.
        // ip+ir*t = ix_start
        const float t = (ix_start - ip.x) / ir.x;
        const float iy_start = ip.y + t * ir.y;
        const float iz_start = ip.z + t * ir.z;

        float val = 0.0f;
        if (r.x > 0.0f)
        {
            if (ip.x >= float(N.x) - 0.5f) return p;
            int ix_max = N.x - 1;

            val = TEX3D(mu, float(ix_start) + 0.5f, iy_start + 0.5f, iz_start + 0.5f) *
                ((float(ix_start) - 0.5f) - max(-0.5f, ip.x));

            const float iy_offset = iy_start - ir.y * float(ix_start) + 0.5f;
            const float iz_offset = iz_start - ir.z * float(ix_start) + 0.5f;

            float ix_opt = float(N.x); // maybe ix_max?
            for (int ix = ix_start; ix <= ix_max; ix++)
            {
                const float next = TEX3D(mu, float(ix) + 0.5f, iy_offset + ir.y * float(ix), iz_offset + ir.z * float(ix));
                val += next;
                if (val > threshold)
                {
                    ix_opt = float(ix) - (val-threshold)/next;
                    break;
                }
            }
            return make_float3(ix_opt*T.x + startVal.x, (iy_offset + ir.y * ix_opt-0.5f)*T.y + startVal.y, (iz_offset + ir.z * ix_opt-0.5f)*T.z + startVal.z);
        }
        else
        {
            if (ip.x <= -0.5f) return p;
            int ix_min = 0;

            val = TEX3D(mu, float(ix_start) + 0.5f, iy_start + 0.5f, iz_start + 0.5f) *
                (min((float(N.x) - 0.5f), ip.x) - (float(ix_start) + 0.5f));

            const float iy_offset = iy_start + ir.y * float(ix_start) + 0.5f;
            const float iz_offset = iz_start + ir.z * float(ix_start) + 0.5f;

            float ix_opt = float(-1); // maybe ix_min?
            for (int ix = ix_start; ix >= ix_min; ix--)
            {
                const float next = TEX3D(mu, float(ix) + 0.5f, iy_offset - ir.y * float(ix), iz_offset - ir.z * float(ix));
                val += next;
                if (val > threshold)
                {
                    ix_opt = float(ix) + (val-threshold)/next;
                    break;
                }
            }
            return make_float3(ix_opt*T.x + startVal.x, (iy_offset - ir.y * ix_opt-0.5f)*T.y + startVal.y, (iz_offset - ir.z * ix_opt-0.5f)*T.z + startVal.z);
        }
    }
    else if (fabsf(r.y) >= fabsf(r.z))
    {
        // integral in y direction
        const float3 ir = make_float3(r.x / fabsf(r.y), r.y / fabsf(r.y), r.z / fabsf(r.y));
        const float increment_amount = sqrtf(1.0f + ir.x * ir.x + ir.z * ir.z) * T.y;
        const float threshold = rhoL / increment_amount;

        const int iy_start = max(0, min(N.y - 1, int(floor(0.5f + ip.y))));

        const float t = (iy_start - ip.y) / ir.y;
        const float ix_start = ip.x + t * ir.x;
        const float iz_start = ip.z + t * ir.z;

        float val = 0.0f;
        if (r.y > 0.0f)
        {
            if (ip.y >= float(N.y) - 0.5f) return p;
            int iy_max = N.y - 1;

            val = TEX3D(mu, ix_start + 0.5f, float(iy_start) + 0.5f, iz_start + 0.5f) *
                ((float(iy_start) - 0.5f) - max(-0.5f, ip.y));

            const float ix_offset = ix_start - ir.x * float(iy_start) + 0.5f;
            const float iz_offset = iz_start - ir.z * float(iy_start) + 0.5f;

            float iy_opt = float(N.y);
            for (int iy = iy_start; iy <= iy_max; iy++)
            {
                const float next = TEX3D(mu, ix_offset + ir.x * float(iy), float(iy) + 0.5f, iz_offset + ir.z * float(iy));
                val += next;
                if (val > threshold)
                {
                    iy_opt = float(iy) - (val-threshold)/next;
                    break;
                }
            }
            return make_float3((ix_offset-0.5f + ir.x * float(iy_opt))*T.x + startVal.x, (float(iy_opt))*T.y + startVal.y, (iz_offset-0.5f + ir.z * float(iy_opt))*T.z + startVal.z);
        }
        else
        {
            if (ip.y <= -0.5f) return p;
            int iy_min = 0;

            val = TEX3D(mu, ix_start + 0.5f, iy_start + 0.5f, iz_start + 0.5f) *
                (min((float(N.y) - 0.5f), ip.y) - (float(iy_start) + 0.5f));

            const float ix_offset = ix_start + ir.x * float(iy_start) + 0.5f;
            const float iz_offset = iz_start + ir.z * float(iy_start) + 0.5f;

            float iy_opt = float(-1);
            for (int iy = iy_start; iy >= iy_min; iy--)
            {
                const float next = TEX3D(mu, ix_offset - ir.x * float(iy), float(iy) + 0.5f, iz_offset - ir.z * float(iy));
                val += next;
                if (val > threshold)
                {
                    iy_opt = float(iy) + (val-threshold)/next;
                    break;
                }
            }
            return make_float3((ix_offset-0.5f - ir.x * float(iy_opt))*T.x + startVal.x, (float(iy_opt))*T.y + startVal.y, (iz_offset-0.5f - ir.z * float(iy_opt))*T.z + startVal.z);
        }
    }
    else
    {
        // integral in z direction
        const float3 ir = make_float3(r.x / fabsf(r.z), r.y / fabsf(r.z), r.z / fabsf(r.z));
        const float increment_amount = sqrtf(1.0f + ir.x * ir.x + ir.y * ir.y) * T.z;
        const float threshold = rhoL / increment_amount;

        const int iz_start = max(0, min(N.z - 1, int(floor(0.5f + ip.z))));

        const float t = (iz_start - ip.z) / ir.z;
        const float ix_start = ip.x + t * ir.x;
        const float iy_start = ip.y + t * ir.y;

        float val = 0.0f;
        if (r.z > 0.0f)
        {
            if (ip.z >= float(N.z) - 0.5f) return p;
            int iz_max = N.z - 1;

            val = TEX3D(mu, ix_start + 0.5f, iy_start + 0.5f, iz_start + 0.5f) *
                ((float(iz_start) - 0.5f) - max(-0.5f, ip.z));

            const float ix_offset = ix_start - ir.x * float(iz_start) + 0.5f;
            const float iy_offset = iy_start - ir.y * float(iz_start) + 0.5f;

            float iz_opt = float(N.z);
            for (int iz = iz_start; iz <= iz_max; iz++)
            {
                const float next = TEX3D(mu, ix_offset + ir.x * float(iz), iy_offset + ir.y * float(iz), float(iz) + 0.5f);
                val += next;
                if (val > threshold)
                {
                    iz_opt = float(iz) - (val-threshold)/next;
                    break;
                }
            }
            return make_float3((ix_offset-0.5f + ir.x * float(iz_opt))*T.x + startVal.x, (iy_offset-0.5f + ir.y * float(iz_opt))*T.y + startVal.y, float(iz_opt)*T.z + startVal.z);
        }
        else
        {
            if (ip.z <= -0.5f) return p;
            int iz_min = 0;

            val = TEX3D(mu, ix_start + 0.5f, iy_start + 0.5f, float(iz_start) + 0.5f) *
                (min((float(N.z) - 0.5f), ip.z) - (float(iz_start) + 0.5f));

            const float ix_offset = ix_start + ir.x * float(iz_start) + 0.5f;
            const float iy_offset = iy_start + ir.y * float(iz_start) + 0.5f;

            float iz_opt = float(-1);
            for (int iz = iz_start; iz >= iz_min; iz--)
            {
                const float next = TEX3D(mu, ix_offset - ir.x * float(iz), iy_offset - ir.y * float(iz), float(iz) + 0.5f);
                val += next;
                if (val > threshold)
                {
                    iz_opt = float(iz) + (val-threshold)/next;
                    break;
                }
            }
            return make_float3((ix_offset-0.5f - ir.x * float(iz_opt))*T.x + startVal.x, (iy_offset-0.5f - ir.y * float(iz_opt))*T.y + startVal.y, float(iz_opt)*T.z + startVal.z);
        }
    }
}

__global__ void setup_uRand(const int4 N, const int extraCols, uint64 seed)
{
    const int m = threadIdx.x + blockIdx.x * blockDim.x;
    const int n = threadIdx.y + blockIdx.y * blockDim.y;
    if (m >= N.y || n >= N.z+extraCols) return;
    const int ind = m*(N.z+extraCols) + n;
    
    curand_init(seed, ind, 0, &d_states[ind]);
}

__device__ float uRand(const uint64 tid)
{
    return curand_uniform(&(d_states[tid]));
}

__device__ float randomAngle(const uint64 tid)
{
    return uRand(tid)*2.0f*PI;
}

__device__ float ExpRand(const uint64 tid)
{
    return -logf(1.0f-uRand(tid));
}

// Given there was an interaction, is it PE, CS, or RS?
__device__ int typeRand(const float mu_p, const float mu_c, const float mu_r, const uint64 tid)
{
    const float y = uRand(tid);
    const float s = mu_p + mu_c + mu_r;
    
    //printf("%f %f %f\n", y, mu_p/s, (mu_p+mu_c)/s);
    
    /*
    if (y < mu_p/s)
        return 0; // PE
    else if (y < (mu_p+mu_c)/s)
        return 1; // CS
    else
        return 2; // RS
    //*/
    //*
    if (y*s < mu_p)
        return 0; // PE
    else if (y*s < (mu_p+mu_c))
        return 1; // CS
    else
        return 2; // RS
    //*/
}

__device__ float3 setTrajectory(const float theta, const float phi)
{
    return make_float3(-cosf(theta)*sinf(phi), cosf(theta)*cosf(phi), sinf(theta));
}

__device__ void updateTrajectory(float3& r, const float theta, const float phi)
{
    const float cos_theta = cosf(theta);
    const float sin_theta = sinf(theta);
    const float cos_phi = cosf(phi);
    const float sin_phi = sinf(phi);

    const float3 v = r;
    r.x = v.x*cos_theta - (v.y*sin_phi + v.x*v.z*cos_phi)*sin_theta;
    r.y = v.y*cos_theta + (v.x*sin_phi - v.y*v.z*cos_phi)*sin_theta;
    r.z = v.z*cos_theta + (v.x*v.x + v.y*v.y)*cos_phi*sin_theta;
}

/**
 * For the pencil scintillator beam, r is (0,0,1); updateTrajectory is degenerate there
 * and leaves x and y at zero. Compton/Rayleigh deflection after scatter_simulation uses
 * updateTrajectory in trackPhoton (non-axial source-to-pixel directions).
 */
__device__ void scatterDeflectUnitVector(float3& w, const float theta, const float phi)
{
    const float cth = cosf(theta);
    const float sth = sinf(theta);
    const float cph = cosf(phi);
    const float sph = sinf(phi);
    // t not collinear with w: cross(t,w) is a vector perpendicular to w
    const float3 t = (fabsf(w.x) < 0.9f) ? make_float3(1.0f, 0.0f, 0.0f) : make_float3(0.0f, 1.0f, 0.0f);
    const float3 axb = make_float3(
        t.y * w.z - t.z * w.y, t.z * w.x - t.x * w.z, t.x * w.y - t.y * w.x);
    const float n_a = rsqrtf(axb.x * axb.x + axb.y * axb.y + axb.z * axb.z);
    const float3 a = make_float3(axb.x * n_a, axb.y * n_a, axb.z * n_a);
    // b = w x a (perpendicular, unit; w, a, b is right-handed)
    const float3 b = make_float3(
        w.y * a.z - w.z * a.y, w.z * a.x - w.x * a.z, w.x * a.y - w.y * a.x);
    w = make_float3(
        cth * w.x + sth * (cph * a.x + sph * b.x),
        cth * w.y + sth * (cph * a.y + sph * b.y),
        cth * w.z + sth * (cph * a.z + sph * b.z));
    const float nw = rsqrtf(w.x * w.x + w.y * w.y + w.z * w.z);
    w.x *= nw;
    w.y *= nw;
    w.z *= nw;
}

// Inverse-CDF angle samplers. The dsigma_dOmega / d_mm_dsigma_* textures store, per (energy[,
// material], component), the inverse CDF of the angular distribution (x-axis = cumulative
// probability u, in [0, d_invCDF_scale] texels). A single uniform draw u in (0,1] therefore maps
// directly to a scatter angle via one (hardware-interpolated) texture lookup at x = u*d_invCDF_scale:
// no rejection loop, no warp divergence. The old rejection-sampling versions are kept (disabled) just
// below for easy revert.
__device__ float randomComptonScatterAngle(const float energy, const uint64 tid)
{
    return TEX3D(dsigma_dOmega, uRand(tid) * d_invCDF_scale, energy + 0.5f, 0.5f);
}

__device__ float randomRayleighScatterAngle(const float energy, const uint64 tid)
{
    return TEX3D(dsigma_dOmega, uRand(tid) * d_invCDF_scale, energy + 0.5f, 1.5f);
}

// Multi-material angle samplers: read the inverse CDF from the per-component, per-material textures
// at the (fractional) material index (z coordinate). Compton and Rayleigh live in separate textures,
// so the material-axis linear filtering only ever blends materials of the same component.
__device__ float randomComptonScatterAngle_mm(const float energy, const float matIndex, const uint64 tid)
{
    return TEX3D(d_mm_dsigma_Compton, uRand(tid) * d_invCDF_scale, energy + 0.5f, matIndex + 0.5f);
}

__device__ float randomRayleighScatterAngle_mm(const float energy, const float matIndex, const uint64 tid)
{
    return TEX3D(d_mm_dsigma_Rayleigh, uRand(tid) * d_invCDF_scale, energy + 0.5f, matIndex + 0.5f);
}

#if 0  // ---- OLD rejection-sampling angle samplers (kept for easy revert) ----
       // NOTE: reverting these also requires reverting the table builds in
       // scatterSimulationTables::initialize / initialize_multimaterial back to storing the
       // peak-normalized angular distributions (instead of the inverse-CDF tables).
__device__ float randomComptonScatterAngle(const float energy, const uint64 tid)
{
    //* REJCTION METHOD
    float theta = 0.0f;
    do
    {
        theta = uRand(tid) * PI;
    } while (uRand(tid) > TEX3D(dsigma_dOmega, theta*d_T_angle_inv+0.5f, energy+0.5f, 0.5f));
    return theta;
    //*/

    /* INVERSE CDF METHOD
    const float u = uRand();
    float accum = 0.0f;
    const float T_angle = PI/1800.0;
    for (int itheta = 0; itheta < 1800; itheta++)
    {
        accum += TEX3D(dsigma_dOmega, itheta+0.5f, energy+0.5f, 0.5f);
        if (accum > u)
            return float(itheta)*T_angle;
    }
    return PI;
    //*/

    /* EGS-5 METHOD
    float retVal = 0.0f;
    
    const float eps0 = 511.0f/(511.0f+2.0f*energy);
    float eps;
    
    const float alpha1 = log(1.0f/eps0);
    const float alpha2 = 0.5f*(1.0f-eps0*eps0);
    float r1,r2,r3;
    float fFactor;
    do
    {
        do
        {
            //3 (0,1) uniform random
            r1 = uRand();
            r2 = uRand();
            r3 = uRand();
            
            if(r1 < alpha1/(alpha1+alpha2))    //f1
            {
                eps = expf(-r2*alpha1);
                //f1 = 1.0/(alpha1*eps);
            }
            else    //f2
            {
                eps = sqrtf(eps0*eps0+(1.0f-eps0*eps0)*r2);
                //f2 = eps/alpha2;
            }
            retVal = acos(1.0f-511.0f*(1.0f-eps)/(energy*eps));

        } while(1.0F - eps/(1.0F+eps*eps)*sin(retVal)*sin(retVal) < r3);
        //fFactor = incoherentFF->getData(sin(0.5f*retVal) * 8.066e6f*energy);
        fFactor = incoherentFF(energy, retVal);
    } while (fFactor < uRand() * d_Ze);
    
    return retVal;
    //*/
}

__device__ float randomRayleighScatterAngle(const float energy, const uint64 tid)
{
    float theta = 0.0f;
    do
    {
        theta = uRand(tid) * PI;
    } while (uRand(tid) > TEX3D(dsigma_dOmega, theta*d_T_angle_inv+0.5f, energy+0.5f, 1.5f));
    return theta;
}

__device__ float randomComptonScatterAngle_mm(const float energy, const float matIndex, const uint64 tid)
{
    float theta = 0.0f;
    do
    {
        theta = uRand(tid) * PI;
    } while (uRand(tid) > TEX3D(d_mm_dsigma_Compton, theta*d_T_angle_inv+0.5f, energy+0.5f, matIndex+0.5f));
    return theta;
}

__device__ float randomRayleighScatterAngle_mm(const float energy, const float matIndex, const uint64 tid)
{
    float theta = 0.0f;
    do
    {
        theta = uRand(tid) * PI;
    } while (uRand(tid) > TEX3D(d_mm_dsigma_Rayleigh, theta*d_T_angle_inv+0.5f, energy+0.5f, matIndex+0.5f));
    return theta;
}
#endif  // ---- end OLD rejection-sampling angle samplers ----

__device__ bool insideObject(const float3 pos)
{
    if (d_AABB_lo.x <= pos.x && d_AABB_lo.y <= pos.y && d_AABB_lo.z <= pos.z && pos.x <= d_AABB_hi.x && pos.y <= d_AABB_hi.y && pos.z <= d_AABB_hi.z)
        return true;
    else
        return false;
}

__device__ void trackPhoton(float3& position, float3& r, float& energy, int& numberOfInteractions, bool& hasRayleighEvent, TEX_DATA f, const int4 N_f, const float4 T_f, const float4 startVal_f, const uint64 tid, const float rhoL_full, bool printDebug)
{
    numberOfInteractions = 0;
    hasRayleighEvent = false;

    do
    {
        // Calculate path length until next interaction and update position
        float sigma_PE = TEX1D(d_sigma_PE, energy+0.5f);
        float sigma_CS = TEX1D(d_sigma_CS, energy+0.5f);
        float sigma_RS = TEX1D(d_sigma_RS, energy+0.5f);
        float sigma_tot = sigma_PE + sigma_CS + sigma_RS;
        float rhoL = ExpRand(tid)/sigma_tot; // path traveled by photon

        //*
        if (rhoL > rhoL_full)
        {
            // photon passes through object
            break;
        }
        //*/

        int interactionType = typeRand(sigma_PE, sigma_CS, sigma_RS, tid);
        //*
        if (interactionType == 0)
        {
            // we know from the above condition, that there is
            // an interaction within the volume, so
            // don't bother ray-tracing if the photon just gets absorbed
            energy = 0.0f;
            break;
        }
        //*/

        float3 nextPos = divergentBeamTransform(f, N_f, T_f, startVal_f, position, r, rhoL);
        if (printDebug)
            printf("DBG2 interactionType=%d rhoL_samp=%.3f pos=(%.2f,%.2f,%.2f) r=(%.3f,%.3f,%.3f) -> nextPos=(%.2f,%.2f,%.2f) inside=%d\n",
                interactionType, rhoL, position.x, position.y, position.z, r.x, r.y, r.z, nextPos.x, nextPos.y, nextPos.z, int(insideObject(nextPos)));
        
        // if passed outside object end simulation
        if (/*numberOfInteractions > 0 &&*/ insideObject(nextPos) == false)
        {
            if (printDebug)
            {
                printf("photon escaped with energy %f\n", energy);
                printf("rhoL = %f, curPos = (%f, %f, %f)\n", rhoL, nextPos.x, nextPos.y, nextPos.z);
            }
            position = nextPos;
            break;
        }
        
        float theta_new = 0.0;
        switch (interactionType)
        {
            case 0: // PE
                if (printDebug)
                    printf("PE\n");
                energy = 0.0f;
                break;
            case 1: // CS
                theta_new = randomComptonScatterAngle(energy, tid);
                //energy = ELECTRON_REST_MASS_ENERGY / (ELECTRON_REST_MASS_ENERGY/energy + 1.0f-cosf(theta_new));
                energy *= ELECTRON_REST_MASS_ENERGY / (ELECTRON_REST_MASS_ENERGY + energy*(1.0f-cosf(theta_new)));
                updateTrajectory(r, theta_new, randomAngle(tid));
                numberOfInteractions += 1;
                if (printDebug)
                    printf("CS (%f degrees, energy -> %f)\n", theta_new*180.0/PI, energy);
                break;
            case 2: // RS
                theta_new = randomRayleighScatterAngle(energy, tid);
                updateTrajectory(r, theta_new, randomAngle(tid));
                hasRayleighEvent = true;
                numberOfInteractions += 1;
                if (printDebug)
                    printf("RS (%f)\n", theta_new*180.0/PI);
                break;
            default:
                theta_new = 0.0f;
        }
        
        position = nextPos;
        if (numberOfInteractions > d_max_scatters)
        {
            energy = 0.0f;
            break;
        }
    } while (energy > 0.0f);
}

__global__ void setup_uRand1D(int n, uint64 seed)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
        curand_init(seed, (uint64_t)i, 0, &d_states[i]);
}

/**
 * Local coordinates: x,y in the scintillator plane (mm), z = depth (0 = entry face, +z = into slab).
 * Pencil beam: initial direction (0,0,1). Mass density d_slab_rho is uniform (g/mm^3).
 */
__device__ void trackPhotonScintillator(
    float* __restrict__ event_buf,
    int max_scatters,
    const uint64 tid,
    float energy,
    int& nwritten)
{
    float3 p = make_float3(0.0f, 0.0f, 0.0f);
    //float3 r = make_float3(0.0f, 0.0f, 1.0f);
    float3 r = d_direction;
    int numberOfInteractions = 0; // Compton + Rayleigh (same as trackPhoton)
    nwritten = 0;

    while (energy > 0.0f)
    {
        if (nwritten >= max_scatters)
            break;

        const float sigma_PE = TEX1D(d_sigma_PE, energy + 0.5f);
        const float sigma_CS = TEX1D(d_sigma_CS, energy + 0.5f);
        const float sigma_RS = TEX1D(d_sigma_RS, energy + 0.5f);
        const float sigma_tot = sigma_PE + sigma_CS + sigma_RS;
        if (sigma_tot <= 0.0f)
            break;

        // Mean free path in mm for constant density: (g/mm^2 line integral) / (g/mm^3) = mm
        const float rhoL = ExpRand(tid) / sigma_tot;
        const float s_mfp = rhoL / d_slab_rho;

        // Distance to exit the slab (front at z=0, back at z=d_slab_T)
        float s_to_exit = 1.0e30f;
        if (r.z > 1.0e-6f)
        {
            s_to_exit = (d_slab_T - p.z) / r.z;
        }
        else if (r.z < -1.0e-6f)
        {
            s_to_exit = (0.0f - p.z) / r.z;
        }
        if (s_to_exit < 0.0f)
            s_to_exit = 0.0f;

        if (s_mfp >= s_to_exit)
        {
            // Leak out of front/back face before interacting
            break;
        }

        p = make_float3(p.x + r.x * s_mfp, p.y + r.y * s_mfp, p.z + r.z * s_mfp);

        const int interactionType = typeRand(sigma_PE, sigma_CS, sigma_RS, tid);

        if (interactionType == 0)
        {
            // Photoelectric: local energy deposit (keV)
            event_buf[4 * nwritten + 0] = p.x;
            event_buf[4 * nwritten + 1] = p.y;
            event_buf[4 * nwritten + 2] = p.z;
            event_buf[4 * nwritten + 3] = energy;
            nwritten++;
            energy = 0.0f;
            break;
        }
        else if (interactionType == 1)
        {
            const float energy_before = energy;
            const float theta_new = randomComptonScatterAngle(energy, tid);
            //energy = ELECTRON_REST_MASS_ENERGY
            //    / (ELECTRON_REST_MASS_ENERGY / energy + 1.0f - cosf(theta_new));
            energy *= ELECTRON_REST_MASS_ENERGY / (ELECTRON_REST_MASS_ENERGY + energy*(1.0f-cosf(theta_new)));
            const float energy_dep = energy_before - energy;
            numberOfInteractions += 1;
            event_buf[4 * nwritten + 0] = p.x;
            event_buf[4 * nwritten + 1] = p.y;
            event_buf[4 * nwritten + 2] = p.z;
            event_buf[4 * nwritten + 3] = energy_dep;
            nwritten++;
            if (nwritten >= max_scatters)
            {
                energy = 0.0f;
                break;
            }
            scatterDeflectUnitVector(r, theta_new, randomAngle(tid));
        }
        else
        {
            const float theta_new = randomRayleighScatterAngle(energy, tid);
            numberOfInteractions += 1;
            event_buf[4 * nwritten + 0] = p.x;
            event_buf[4 * nwritten + 1] = p.y;
            event_buf[4 * nwritten + 2] = p.z;
            event_buf[4 * nwritten + 3] = 0.0f;
            nwritten++;
            if (nwritten >= max_scatters)
            {
                energy = 0.0f;
                break;
            }
            scatterDeflectUnitVector(r, theta_new, randomAngle(tid));
        }

        if (numberOfInteractions > d_max_scatters)
        {
            energy = 0.0f;
            break;
        }
    }
}

__global__ void detector_scatter_simulation_kernel(
    float* __restrict__ events,
    uint64 num_photons,
    int N_energies,
    int max_scatters,
    float wsum,
    TEX_DATA source_spectra,
    TEX_DATA source_energies)
{
    const uint64 pid = (uint64)threadIdx.x + (uint64)blockIdx.x * (uint64)blockDim.x;
    if (pid >= num_photons)
        return;
    const uint64 tid = pid;
    float* __restrict__ event_buf = events + pid * (uint64)max_scatters * 4;

    // Sample one energy bin weighted by the source spectrum (same weights as scatterSimulation)
    const float u = uRand(tid) * wsum;
    float acc = 0.0f;
    int iE0 = 0;
    bool chosen = false;
    for (int i = 0; i < N_energies; i++)
    {
        acc += TEX1D(source_spectra, i + 0.5f);
        if (u < acc)
        {
            iE0 = i;
            chosen = true;
            break;
        }
    }
    if (N_energies > 0 && !chosen)
        iE0 = N_energies - 1;
    if (N_energies > 0 && iE0 > N_energies - 1)
        iE0 = N_energies - 1;
    const float initial_energy = TEX1D(source_energies, iE0 + 0.5f);

    int nwritten = 0;
    trackPhotonScintillator(event_buf, max_scatters, tid, initial_energy, nwritten);
}

// Precompute the primary-ray line integral (rhoL) for every detector element, once per view. This is
// the expensive full ray-cast through the object; it depends only on geometry (source/pixel position),
// not on energy or the photon RNG, so it can be reused by every photon of that pixel. Computing it in
// its own kernel (one thread per detector element) lets the scatter kernel parallelize the photon
// histories across many more threads WITHOUT recomputing this integral per photon.
__global__ void precompute_line_integral_kernel(float* rhoL_map, const int4 N_g, const float4 T_g, const float4 startVal_g, TEX_DATA f, const int4 N_f, const float4 T_f, const float4 startVal_f, const float3 sourcePos, const float3 moduleCenter, const float3 v_vec, const float3 u_vec,
    const int extraCols, const bool padOnLeft)
{
    const uint64 m = threadIdx.x + blockIdx.x * blockDim.x;
    const uint64 n = threadIdx.y + blockIdx.y * blockDim.y;
    if (m >= N_g.y || n >= N_g.z+extraCols) return;
    const uint64 pixel = m*(N_g.z+extraCols) + n;

    float u_shift = 0.0f;
    if (padOnLeft)
        u_shift = -extraCols*T_g.z;
    const float v = m * T_g.y + startVal_g.y;
    const float u = n * T_g.z + startVal_g.z + u_shift;

    const float3 endPos = make_float3(moduleCenter.x + v*v_vec.x + u*u_vec.x, moduleCenter.y + v*v_vec.y + u*u_vec.y, moduleCenter.z + v*v_vec.z + u*u_vec.z);
    rhoL_map[pixel] = fullLineIntegral(f, N_f, T_f, startVal_f, sourcePos, endPos);
}

// Photon-parallel scatter kernel. Each thread processes a contiguous chunk of "photons_per_thread"
// photon histories for a single detector element (pixel), reading that pixel's precomputed line
// integral from rhoL_map. Threads are indexed as (pixel, photon-group) so the (typically 500-5000)
// histories per pixel are spread across many threads (raising occupancy and cutting warp divergence)
// while the expensive fullLineIntegral is still computed only once per pixel (in the kernel above).
// One curand state is used per thread (state index == global thread index).
__global__ void scatter_simulation_kernel(float* scatter_data, float* scatter_data_high_order, const float* __restrict__ rhoL_map, const int4 N_g, const float4 T_g, const float4 startVal_g, TEX_DATA f, const int4 N_f, const float4 T_f, const float4 startVal_f, const float3 sourcePos, const float3 moduleCenter, const float3 v_vec, const float3 u_vec,
    TEX_DATA source_spectra, TEX_DATA source_energies, TEX_DATA detector_response, const int total_photons_per_pixel, const int photons_per_thread, const int groups_per_pixel,
    const int extraCols, const bool padOnLeft)
{
    const uint64 gid = (uint64)threadIdx.x + (uint64)blockIdx.x * (uint64)blockDim.x;
    const uint64 num_cols = (uint64)(N_g.z + extraCols);
    const uint64 num_pixels = (uint64)N_g.y * num_cols;
    const uint64 total_threads = num_pixels * (uint64)groups_per_pixel;
    if (gid >= total_threads) return;

    const uint64 pixel = gid / (uint64)groups_per_pixel;
    const uint64 group = gid - pixel * (uint64)groups_per_pixel;

    // Photon-index range [p0, p1) this thread is responsible for, within its pixel.
    int p0 = int(group) * photons_per_thread;
    if (p0 >= total_photons_per_pixel) return;
    int p1 = p0 + photons_per_thread;
    if (p1 > total_photons_per_pixel) p1 = total_photons_per_pixel;

    // Primary-ray line integral through the object, precomputed once per pixel. rhoL <= 0 means the
    // primary ray traverses no material, so every photon would pass straight through without interacting
    // (in trackPhoton, tau = ExpRand()/sigma_tot > 0 > rhoL_full) and no scatter can be generated.
    const float rhoL = rhoL_map[pixel];
    if (rhoL <= 0.0f)
        return;

    const uint64 m = pixel / num_cols;
    const uint64 n = pixel - m * num_cols;
    const uint64 tid = gid;

    // if padOnLeft, params->centerCol += extraCols;
    //startVals.z = -(params->centerCol + params->colShiftFromFilter) * params->pixelWidth;
    float u_shift = 0.0f;
    if (padOnLeft)
        u_shift = -extraCols*T_g.z;
    const float v = m * T_g.y + startVal_g.y;
    const float u = n * T_g.z + startVal_g.z + u_shift;

    const float3 n_vec = make_float3(u_vec.y * v_vec.z - u_vec.z * v_vec.y,
        u_vec.z * v_vec.x - u_vec.x * v_vec.z,
        u_vec.x * v_vec.y - u_vec.y * v_vec.x);
    
    const float3 endPos = make_float3(moduleCenter.x + v*v_vec.x + u*u_vec.x, moduleCenter.y + v*v_vec.y + u*u_vec.y, moduleCenter.z + v*v_vec.z + u*u_vec.z);
    float3 r_init = make_float3(endPos.x - sourcePos.x, endPos.y - sourcePos.y, endPos.z - sourcePos.z);
    const float r_mag_inv = rsqrtf(r_init.x*r_init.x + r_init.y*r_init.y + r_init.z*r_init.z);
    r_init.x *= r_mag_inv;
    r_init.y *= r_mag_inv;
    r_init.z *= r_mag_inv;

    const float T_u_inv = 1.0f / T_g.z;
    const float T_v_inv = 1.0f / T_g.y;

    // Map the flat photon index range [p0, p1) onto source energy bins: photon j belongs to the energy
    // bin ienergy where the running sum of per-energy photon counts (round(source_spectra)) first
    // exceeds j. This reproduces the original nested (energy, event) loop, just sliced across threads.
    int ienergy = 0;
    int cum = 0;
    int cnt = (N_g.w > 0) ? int(0.5f + TEX1D(source_spectra, 0.5f)) : 0;
    while (ienergy < N_g.w - 1 && p0 >= cum + cnt)
    {
        cum += cnt;
        ienergy++;
        cnt = int(0.5f + TEX1D(source_spectra, ienergy + 0.5f));
    }

    bool printDebug = false;
    for (int ievent = p0; ievent < p1; ievent++)
    {
        while (ienergy < N_g.w - 1 && ievent >= cum + cnt)
        {
            cum += cnt;
            ienergy++;
            cnt = int(0.5f + TEX1D(source_spectra, ienergy + 0.5f));
        }

        int numberOfInteractions;
        bool hasRayleighEvent;
        float energy = TEX1D(source_energies, ienergy + 0.5f);
        float3 startPos = make_float3(sourcePos.x, sourcePos.y, sourcePos.z);
        float3 r = make_float3(r_init.x, r_init.y, r_init.z);

        trackPhoton(startPos, r, energy, numberOfInteractions, hasRayleighEvent, f, N_f, T_f, startVal_f, tid, rhoL, printDebug);

        //if (energy > 0.0f && (d_min_scatters <= numberOfInteractions || numberOfInteractions == 0))
        if (energy > 0.0f && d_min_scatters <= numberOfInteractions)
        {
            // now calculate where startPos + t*r hits the detector.
            // The ray tracer can march an escaping photon to the edge of the volume box, which may lie
            // *behind* the detector plane (more so at oblique views / larger sod). startInFront tells us
            // which side of the detector plane the exit point is on (relative to the source): a photon
            // heading toward the detector reaches the plane at t >= 0 when in front, but at t <= 0 when
            // it has already passed behind it. Rejecting backscatter therefore depends on that side.
            const float startDotN = (startPos.x-moduleCenter.x)*n_vec.x + (startPos.y-moduleCenter.y)*n_vec.y + (startPos.z-moduleCenter.z)*n_vec.z;
            const float srcDotN = (sourcePos.x-moduleCenter.x)*n_vec.x + (sourcePos.y-moduleCenter.y)*n_vec.y + (sourcePos.z-moduleCenter.z)*n_vec.z;
            const float t = -startDotN / (r.x*n_vec.x + r.y*n_vec.y + r.z*n_vec.z);
            const bool startInFront = (startDotN * srcDotN > 0.0f);
            if ((t >= 0.0f && startInFront) || (t <= 0.0f && !startInFront))
            {
                const float3 finalPosition = make_float3(startPos.x + t*r.x, startPos.y + t*r.y, startPos.z + t*r.z);
                const float u_ind = ((finalPosition.x-moduleCenter.x) * u_vec.x + (finalPosition.y-moduleCenter.y) * u_vec.y + (finalPosition.z-moduleCenter.z) * u_vec.z - startVal_g.z) * T_u_inv;
                const float v_ind = ((finalPosition.x-moduleCenter.x) * v_vec.x + (finalPosition.y-moduleCenter.y) * v_vec.y + (finalPosition.z-moduleCenter.z) * v_vec.z - startVal_g.y) * T_v_inv;
                if (-0.5f < u_ind && u_ind < N_g.z-0.5f && -0.5f < v_ind && v_ind < N_g.y-0.5f)
                {
                    const uint64 ind = int(0.5f+v_ind)*N_g.z + int(0.5f+u_ind);
                    if (numberOfInteractions > 1)
                        atomicAdd(&scatter_data_high_order[ind], TEX1D(detector_response, energy+0.5f));
                    else
                        atomicAdd(&scatter_data[ind], TEX1D(detector_response, energy+0.5f));
                }
            }
        }
    }
}

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
    float* direction)
{
    /*
    The purpose of this function is to simulate a polychromatic (described by the source, energies, and N_energies)
    input parameters) pencil beam of x-rays that hit a scintillator. chemForm is a chemical formula for compound
    cross sections; mass_density (g/cm^3) and thickness (mm) describe the uniform slab, otherwise infinite in extent.
    The x-ray beam hits the scintillator orthogonal.  This function will then record
    the location (x,y,z) and amount of energy deposited (Compton or Photoelectric event) for every interaction
    in the scintillator.   The simulation will stop if the x-ray leaves the front or back face of the scintillator
    or there is a photoelectric event, or we reach the "max_scatters" number of scattering events.  These events are
    to be stored in the "events" input parameter which is of size num_photons * max_scatters * 4.
    The "4" here is to save the (x,y,z) position of the interaction and the amount of energy deposited.
    Each cuda thread is to write their simulated interactions in a contiguous chunk with offset given by
    thread_index * max_scatters * 4.
    There is another function in this file called scatterSimulation.  I want you to use many of the same functions
    that are used in this function to perform this task.
    */
    if (params == NULL || events == NULL || source == NULL || energies == NULL || N_energies <= 0 || chemForm == NULL)
        return false;
    if (thickness <= 0.0f || mass_density <= 0.0f || num_photons <= 0)
        return false;
    max_scatters = std::max(0, min(max_scatters, 100));
    if (max_scatters == 0)
        return false;

    cudaError_t cudaStatus = cudaSuccess;

    float direction_local[3] = {0.0f, 0.0f, 1.0f};
    if (direction == nullptr)
        direction = direction_local;
    if (direction[2] < 0.0)
    {
        direction[0] *= -1.0;
        direction[1] *= -1.0;
        direction[2] *= -1.0;
    }
    float direction_length = sqrt(direction[0]*direction[0] + direction[1]*direction[1] + direction[2]*direction[2]);
    if (direction_length == 0.0)
    {
        direction[0] = 0.0;
        direction[1] = 0.0;
        direction[2] = 1.0;
        direction_length = 1.0;
        //print("Error: invalid x-ray flux direction!\n");
        //return false;
    }
    float3 direction3 = make_float3(direction[0]/direction_length, direction[1]/direction_length, direction[2]/direction_length);
    cudaStatus = cudaMemcpyToSymbol(d_direction, &direction3, sizeof(float3));

    float wsum = 0.0f;
    for (int i = 0; i < N_energies; i++)
        wsum += source[i];
    if (wsum <= 0.0f)
    {
        fprintf(stderr, "detectorScatterSimulation: non-positive source spectrum\n");
        return false;
    }

    cudaSetDevice(params->whichGPU);

    int max_energy = int(ceil(energies[N_energies - 1]));
    if (max_energy < 0)
        max_energy = 0;

    // Uniform slab: user-supplied mass density (g/cm^3) -> g/mm^3 for mean free path (sigma tables are g/mm^2)
    {
        const float rho_g_mm3 = mass_density * 1.0e-3f;
        cudaStatus = cudaMemcpyToSymbol(d_slab_rho, &rho_g_mm3, sizeof(float));
        if (cudaStatus != cudaSuccess)
        {
            fprintf(
                stderr,
                "detectorScatterSimulation: cudaMemcpyToSymbol(d_slab_rho) failed! %s\n",
                cudaGetErrorString(cudaStatus));
            return false;
        }
    }
    cudaStatus = cudaMemcpyToSymbol(d_slab_T, &thickness, sizeof(float));
    if (cudaStatus != cudaSuccess)
    {
        fprintf(
            stderr, "detectorScatterSimulation: cudaMemcpyToSymbol(d_slab_T) failed! %s\n", cudaGetErrorString(cudaStatus));
        return false;
    }

    scatterSimulationTables scatterTables;
    // min arg only affects scatter_simulation; detector slab records every CS/RS (pass 1 for d_min_scatters)
    scatterTables.initialize(params, chemForm, max_energy, 1, max_scatters, num_photons);

    float* dev_events = 0;
    if ((cudaStatus = cudaMalloc((void**)&dev_events, 4 * max_scatters * num_photons * sizeof(float)))
        != cudaSuccess)
    {
        fprintf(stderr, "detectorScatterSimulation: cudaMalloc(events) failed! %s\n", cudaGetErrorString(cudaStatus));
        return false;
    }
    if ((cudaStatus = cudaMemset(dev_events, 0, 4 * max_scatters * num_photons * sizeof(float))) != cudaSuccess)
    {
        fprintf(stderr, "detectorScatterSimulation: cudaMemset failed! %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(dev_events);
        return false;
    }

    // CDF for initial energy: use unscaled spectrum weights (same as scatterSimulation source_sum)
    float* dev_source = copy1DdataToGPU(source, N_energies, params->whichGPU);
    if (dev_source == NULL)
    {
        fprintf(stderr, "detectorScatterSimulation: copy1DdataToGPU(source) failed!\n");
        cudaFree(dev_events);
        return false;
    }
    TEX_DATA source_txt = {};
    TEX_ARRAY source_array = loadTexture1D(source_txt, dev_source, N_energies, false, true);

    float* dev_energies = copy1DdataToGPU(energies, N_energies, params->whichGPU);
    if (dev_energies == NULL)
    {
        fprintf(stderr, "detectorScatterSimulation: copy1DdataToGPU(energies) failed!\n");
        freeTexture(source_array, source_txt);
        cudaFree(dev_source);
        cudaFree(dev_events);
        return false;
    }
    TEX_DATA energies_txt = {};
    TEX_ARRAY energies_array = loadTexture1D(energies_txt, dev_energies, N_energies, false, true);

    const int threads = 256;
    const int blocks = int(ceil(double(num_photons) / double(threads)));
    const uint64 tseed = (uint64)time(NULL);
    setup_uRand1D<<<blocks, threads>>>(num_photons, tseed);
    if ((cudaStatus = cudaGetLastError()) != cudaSuccess)
    {
        fprintf(stderr, "detectorScatterSimulation: setup_uRand1D failed! %s\n", cudaGetErrorString(cudaStatus));
        goto cleanup;
    }

    detector_scatter_simulation_kernel<<<blocks, threads>>>(dev_events, (uint64)num_photons, N_energies, max_scatters,
        wsum, source_txt, energies_txt);
    if ((cudaStatus = cudaGetLastError()) != cudaSuccess)
    {
        fprintf(stderr, "detectorScatterSimulation: kernel launch failed! %s\n", cudaGetErrorString(cudaStatus));
        goto cleanup;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(
            stderr, "detectorScatterSimulation: kernel synchronization failed! %s\n", cudaGetErrorString(cudaStatus));
        goto cleanup;
    }

    if ((cudaStatus
            = cudaMemcpy(events, dev_events, 4 * max_scatters * num_photons * sizeof(float), cudaMemcpyDeviceToHost))
        != cudaSuccess)
    {
        fprintf(stderr, "detectorScatterSimulation: cudaMemcpy(events) failed! %s\n", cudaGetErrorString(cudaStatus));
        goto cleanup;
    }

cleanup:
    freeTexture(energies_array, energies_txt);
    cudaFree(dev_energies);
    freeTexture(source_array, source_txt);
    cudaFree(dev_source);
    cudaFree(dev_events);

    return cudaStatus == cudaSuccess;
}

bool scatterSimulation(parameters* params, float* g, float* f, float* source, float* energies, int N_energies, float* detector, const char* chemForm, bool data_on_cpu, int num_photons_per_pixel, int min_scatters, int max_scatters)
{
    if (params == NULL || g == NULL || f == NULL || source == NULL || energies == NULL || N_energies <= 0 || detector == NULL || chemForm == NULL)
        return false;
    min_scatters = max(0, min(min_scatters, 100));
    max_scatters = max(0, min(max_scatters, 100));
    bool do_smoothing = true;
    //bool do_smoothing = false; // FIXME
    cudaSetDevice(params->whichGPU);

    int max_energy = int(ceil(energies[N_energies-1]));

    // Photon-parallel scatter simulation. The expensive primary-ray line integral is computed once per
    // detector element (precompute_line_integral_kernel) and reused by all photons of that pixel, while
    // the many photon histories are spread across many threads in scatter_simulation_kernel. Each
    // photon-thread needs its own curand state, so size the state pool by the total number of threads.
    // photons_per_thread is the tuning knob: smaller -> more parallelism but more curand-state memory
    // and per-thread setup; larger -> less state memory but longer per-thread serial work.
    const int photons_per_thread = 64;
    bool padOnLeft;
    int extraCols = zeroPadForOffsetScan_numberOfColsToAdd(params, padOnLeft);
    int total_photons_per_pixel = 0;
    {
        float src_sum = 0.0f;
        for (int i = 0; i < N_energies; i++)
            src_sum += source[i];
        for (int i = 0; i < N_energies; i++)
            total_photons_per_pixel += int(0.5f + source[i] * num_photons_per_pixel / src_sum);
    }
    const uint64 num_pixels = (uint64)params->numRows * (uint64)(params->numCols + extraCols);
    int groups_per_pixel = (total_photons_per_pixel + photons_per_thread - 1) / photons_per_thread;
    if (groups_per_pixel < 1)
        groups_per_pixel = 1;
    const uint64 total_threads = num_pixels * (uint64)groups_per_pixel;

    scatterSimulationTables scatterTables;
    scatterTables.initialize(params, chemForm, max_energy, min_scatters, max_scatters, (int)total_threads);

    if (params->geometry != parameters::MODULAR)
    {
        printf("Error: scatter estimation algorithm only implemented for modular-beam geometries. Please convert to modular-beam before running this algorithm.\n");
        return false;
    }
    
    //cudaSetDevice(params->whichGPU);
    cudaError_t cudaStatus;

    // Allocate projection data on GPU
    int4 N_g; float4 T_g; float4 startVal_g;
    setProjectionGPUparams(params, N_g, T_g, startVal_g, false);
    N_g.w = N_energies;
    float* dev_g = 0;
    if (data_on_cpu)
    {
        if ((cudaStatus = cudaMalloc((void**)&dev_g, params->projectionData_numberOfElements() * sizeof(float))) != cudaSuccess)
        {
            fprintf(stderr, "scatterSimulation: cudaMalloc(projections) failed!\n");
        }
    }
    else
        dev_g = g;
    //cudaMemset(dev_g, 0, params->projectionData_numberOfElements() * sizeof(float));

    // Allocate volume data on GPU
    int4 N_f; float4 T_f; float4 startVal_f;
    setVolumeGPUparams(params, N_f, T_f, startVal_f);
    float* dev_f = 0;
    if (data_on_cpu)
        dev_f = copyVolumeDataToGPU(f, params, params->whichGPU);
    else
        dev_f = f;
    TEX_DATA f_data_txt = {};
    TEX_ARRAY f_data_array = loadTexture(f_data_txt, dev_f, N_f, false, true, bool(params->volumeDimensionOrder == 1));

    // source: the source spectra
    // energies: the energies of the source spectra
    // detector: the detector response sampled in 1 keV bins
    float source_sum = 0.0;
    for (int i = 0; i < N_energies; i++)
        source_sum += source[i];
    float* source_scaled = new float[N_energies];
    for (int i = 0; i < N_energies; i++)
        source_scaled[i] = source[i]*num_photons_per_pixel/source_sum;

    float air_scan_inv = 0.0;
    for (int i = 0; i < N_energies; i++)
    {
        float gamma = energies[i];
        air_scan_inv += floor(0.5 + source_scaled[i]) * detector[int(0.5 + gamma)];
    }
    //printf("normalization factor = %f\n", air_scan_inv);
    air_scan_inv = 1.0 / air_scan_inv;

    float* dev_source = copy1DdataToGPU(source_scaled, N_energies, params->whichGPU);
    TEX_DATA source_txt = {};
    TEX_ARRAY source_array = loadTexture1D(source_txt, dev_source, N_energies, false, true);
    delete [] source_scaled;

    float* dev_energies = copy1DdataToGPU(energies, N_energies, params->whichGPU);
    TEX_DATA energies_txt = {};
    TEX_ARRAY energies_array = loadTexture1D(energies_txt, dev_energies, N_energies, false, true);

    float* dev_detector = copy1DdataToGPU(detector, max_energy+1, params->whichGPU);
    TEX_DATA detector_txt = {};
    TEX_ARRAY detector_array = loadTexture1D(detector_txt, dev_detector, max_energy+1, true, true);

    float* dev_high_order = 0;
    if (cudaSuccess != cudaMalloc((void**)&dev_high_order, params->numAngles*params->numRows*params->numCols * sizeof(float)))
        fprintf(stderr, "scatterSimulation: cudaMalloc failed!\n");

    cudaMemset(dev_g, 0, params->numAngles*params->numRows*params->numCols*sizeof(float));
    cudaMemset(dev_high_order, 0, params->numAngles*params->numRows*params->numCols*sizeof(float));

    //printf("extraCols = %d\n", extraCols);

    // Per-pixel primary-ray line integral, recomputed for each view.
    float* dev_rhoL = 0;
    if (cudaSuccess != cudaMalloc((void**)&dev_rhoL, num_pixels * sizeof(float)))
        fprintf(stderr, "scatterSimulation: cudaMalloc(dev_rhoL) failed!\n");

    // ********* CALL KERNEL *********
    // 2D launch for the per-pixel line-integral precompute (one thread per detector element).
    dim3 dimBlock(8, 8);
    dim3 dimGrid(int(ceil(double(params->numRows) / double(dimBlock.x))), int(ceil(double(params->numCols+extraCols) / double(dimBlock.y))));

    // 1D launch for the photon-parallel scatter kernel (one thread per photon-group).
    const int photon_block = PHOTON_BLOCK;
    const int photon_grid = int((total_threads + (uint64)photon_block - 1) / (uint64)photon_block);

    setup_uRand1D <<< photon_grid, photon_block >>> ((int)total_threads, time(NULL));

    printf("Monte-Carlo simulation...\n");
    for (int i = 0; i < params->numAngles; i++)
    {
        //printf("Monte-Carlo simulation view %d of %d\n", i+1, params->numAngles);

        float3 sourcePosition = make_float3(params->sourcePositions[3 * i + 0], params->sourcePositions[3 * i + 1], params->sourcePositions[3 * i + 2]);
        float3 moduleCenter = make_float3(params->moduleCenters[3 * i + 0], params->moduleCenters[3 * i + 1], params->moduleCenters[3 * i + 2]);
        float3 rowVector = make_float3(params->rowVectors[3 * i + 0], params->rowVectors[3 * i + 1], params->rowVectors[3 * i + 2]);
        float3 colVector = make_float3(params->colVectors[3 * i + 0], params->colVectors[3 * i + 1], params->colVectors[3 * i + 2]);

        const uint64 ind_offset = uint64(i)*uint64(params->numRows*params->numCols);

        precompute_line_integral_kernel <<< dimGrid, dimBlock >>> (dev_rhoL, N_g, T_g, startVal_g, f_data_txt, N_f, T_f, startVal_f, sourcePosition, moduleCenter, rowVector, colVector, extraCols, padOnLeft);
        scatter_simulation_kernel <<< photon_grid, photon_block >>> (&dev_g[ind_offset], &dev_high_order[ind_offset], dev_rhoL, N_g, T_g, startVal_g, f_data_txt, N_f, T_f, startVal_f, sourcePosition, moduleCenter, rowVector, colVector, source_txt, energies_txt, detector_txt, total_photons_per_pixel, photons_per_thread, groups_per_pixel, extraCols, padOnLeft);
        //cudaStatus = cudaDeviceSynchronize();
    }
    cudaStatus = cudaDeviceSynchronize();
    printf("done\n");
    cudaFree(dev_rhoL);

    if (max_scatters > 1)
    {
        if (do_smoothing && min_scatters < 2)
            blurFilter(dev_high_order, params->numAngles, params->numRows, params->numCols, 8.0, 3, 0, false, params->whichGPU);
        add(dev_g, dev_high_order, make_int3(params->numAngles, params->numRows, params->numCols), params->whichGPU);
    }
    cudaFree(dev_high_order);

    scale(dev_g, air_scan_inv, make_int3(params->numAngles, params->numRows, params->numCols), params->whichGPU);

    // pull result off GPU
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "scatterSimulation: kernel failed!\n");
        fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
        fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
    }

    // Clean up
    freeTexture(f_data_array, f_data_txt);
    if (data_on_cpu)
        cudaFree(dev_f);
    
    freeTexture(source_array, source_txt);
    cudaFree(dev_source);

    freeTexture(energies_array, energies_txt);
    cudaFree(dev_energies);

    freeTexture(detector_array, detector_txt);
    cudaFree(dev_detector);

    /*
    // target: use 12 for 900 projections
    //12.0*float(params->numAngles)/900.0
    //float FWHM = min(0.5*float(params->numAngles), 12.0);
    float FWHM = min(0.5*float(params->numAngles), 12.0*float(params->numAngles)/900.0);
    if (FWHM > 1.0 && do_smoothing)
    {
        // sigma^2 ==> 3/(4*FWHM) * sigma^2 or sigma ==> sqrt(0.75/FWHM)*sigma
        // so for FWHM = 12, the standard deviation is reduced by a factor of 4
        blurFilter(dev_g, params->numAngles, params->numRows, params->numCols, FWHM, 1, 0, false, params->whichGPU);
        //blurFilter(dev_direct_data, params->numAngles, params->numRows, params->numCols, FWHM, 1, 0, false, params->whichGPU);
    }
    //*/

    //cudaFree(dev_direct_data);

    if (data_on_cpu)
        pullProjectionDataFromGPU(g, params, dev_g, params->whichGPU);
    else
        g = dev_g;

    if (data_on_cpu)
        cudaFree(dev_g);

    return true;
}

///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
// Multi-material Device Functions
///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

// Piece-wise linear transfer function T(.) mapping fL (LAC at gammaL) -> LAC at gammaH.
// Mirrors the polychromatic change_energy() setup.
__device__ __forceinline__ float change_energy_mm(float mu_ref)
{
    if (mu_ref <= d_mm_mu_ref.w)
        return fmaf(d_mm_mu_slopes.w, mu_ref, 0.0f);
    else if (mu_ref <= d_mm_mu_ref.x)
        return fmaf(d_mm_mu_slopes.x, mu_ref, d_mm_mu_offsets.x);
    else if (mu_ref <= d_mm_mu_ref.y)
        return fmaf(d_mm_mu_slopes.y, mu_ref, d_mm_mu_offsets.y);
    else
        return fmaf(d_mm_mu_slopes.z, mu_ref, d_mm_mu_offsets.z);
}

// Total LAC (1/mm) at the photon's current energy for a voxel whose reference-energy LAC is fL.
// Uses the SMB decomposition: LAC(gamma) = b_L(gamma)*fL + b_H(gamma)*T(fL).
__device__ __forceinline__ float localLAC_mm(float fL, float bL, float bH)
{
    return bL * fL + bH * change_energy_mm(fL);
}

// Map fL (LAC at gammaL) to a fractional material index in [0, num_materials-1] using the
// knot x-values (the basis-material LAC values at gammaL), assumed ascending.
__device__ __forceinline__ float materialIndex_mm(float fL)
{
    const int M = d_mm_num_materials;
    const float x0 = d_mm_material_knots.x;
    const float x1 = d_mm_material_knots.y;
    const float x2 = d_mm_material_knots.z;
    const float x3 = d_mm_material_knots.w;

    if (fL <= x0)
        return 0.0f;
    if (fL <= x1)
        return (fL - x0) * d_mm_material_knot_inv_gaps.x;
    if (M == 2)
        return 1.0f;
    if (fL <= x2)
        return 1.0f + (fL - x1) * d_mm_material_knot_inv_gaps.y;
    if (M == 3)
        return 2.0f;
    if (fL <= x3)
        return 2.0f + (fL - x2) * d_mm_material_knot_inv_gaps.z;
    return float(M - 1);
}

// Fetch a per-material LAC component (0=PE, 1=CS, 2=RS) at a fractional material index and energy.
__device__ __forceinline__ float sigma_component_mm(int component, float matIndex, float energy)
{
    return TEX3D(d_mm_LAC_components, energy + 0.5f, matIndex + 0.5f, float(component) + 0.5f);
}

// Sample fL at a world position from the fL volume texture (same indexing convention as
// fullLineIntegral / divergentBeamTransform).
__device__ __forceinline__ float sample_fL_mm(TEX_DATA f, const int4 N, const float4 T, const float4 startVal, const float3 pos)
{
    const float ix = (pos.x - startVal.x) * d_mm_inv_T_f.x;
    const float iy = (pos.y - startVal.y) * d_mm_inv_T_f.y;
    const float iz = (pos.z - startVal.z) * d_mm_inv_T_f.z;
    return TEX3D(f, ix + 0.5f, iy + 0.5f, iz + 0.5f);
}

// Pair of line integrals (P*fL, P*T(fL)) from p to dst through the fL volume.
// Mirrors the polychromatic lineIntegralPairKernel (uses change_energy_mm for the
// transfer function). The total optical depth at energy gamma is b_L(gamma)*x + b_H(gamma)*y.
__device__ float2 lineIntegralPair_mm(TEX_DATA mu, const int4 N, const float4 T, const float4 startVal, const float3 p, const float3 dst)
{
    // NOTE: assumes that T.x == T.y == T.z
    const float3 r = make_float3(dst.x - p.x, dst.y - p.y, dst.z - p.z);

    if (fabsf(r.x) >= fabsf(r.y) && fabsf(r.x) >= fabsf(r.z))
    {
        const float3 ip = make_float3((p.x - startVal.x) / T.x, (p.y - startVal.y) / T.y, (p.z - startVal.z) / T.z);
        const float3 ir = make_float3(r.x / fabsf(r.x), r.y / fabsf(r.x), r.z / fabsf(r.x));
        const int ix_start = max(0, min(N.x - 1, int(floor(0.5f + ip.x))));
        const float tt = (ix_start - ip.x) / ir.x;
        const float iy_start = ip.y + tt * ir.y;
        const float iz_start = ip.z + tt * ir.z;

        float curVal = 0.0f;
        float2 val = make_float2(0.0f, 0.0f);
        if (r.x > 0.0f)
        {
            int ix_max = min(N.x - 1, int(ceil((dst.x - startVal.x) / T.x)));
            const float iy_offset = iy_start - ir.y * float(ix_start) + 0.5f;
            const float iz_offset = iz_start - ir.z * float(ix_start) + 0.5f;
            for (int ix = ix_start; ix <= ix_max; ix++)
            {
                curVal = TEX3D(mu, float(ix) + 0.5f, iy_offset + ir.y * float(ix), iz_offset + ir.z * float(ix));
                val.x += curVal;
                val.y += change_energy_mm(curVal);
            }
        }
        else
        {
            int ix_min = max(0, int(floor((dst.x - startVal.x) / T.x)));
            const float iy_offset = iy_start + ir.y * float(ix_start) + 0.5f;
            const float iz_offset = iz_start + ir.z * float(ix_start) + 0.5f;
            for (int ix = ix_start; ix >= ix_min; ix--)
            {
                curVal = TEX3D(mu, float(ix) + 0.5f, iy_offset - ir.y * float(ix), iz_offset - ir.z * float(ix));
                val.x += curVal;
                val.y += change_energy_mm(curVal);
            }
        }
        const float line_length = sqrtf(1.0f + ir.y * ir.y + ir.z * ir.z) * T.x;
        val.x *= line_length;
        val.y *= line_length;
        return val;
    }
    else if (fabsf(r.y) >= fabsf(r.z))
    {
        const float3 ip = make_float3((p.x - startVal.x) / T.x, (p.y - startVal.y) / T.y, (p.z - startVal.z) / T.z);
        const float3 ir = make_float3(r.x / fabsf(r.y), r.y / fabsf(r.y), r.z / fabsf(r.y));
        const int iy_start = max(0, min(N.y - 1, int(floor(0.5f + ip.y))));
        const float tt = (iy_start - ip.y) / ir.y;
        const float ix_start = ip.x + tt * ir.x;
        const float iz_start = ip.z + tt * ir.z;

        float curVal = 0.0f;
        float2 val = make_float2(0.0f, 0.0f);
        if (r.y > 0.0f)
        {
            int iy_max = min(N.y - 1, int(ceil((dst.y - startVal.y) / T.y)));
            const float ix_offset = ix_start - ir.x * float(iy_start) + 0.5f;
            const float iz_offset = iz_start - ir.z * float(iy_start) + 0.5f;
            for (int iy = iy_start; iy <= iy_max; iy++)
            {
                curVal = TEX3D(mu, ix_offset + ir.x * float(iy), float(iy) + 0.5f, iz_offset + ir.z * float(iy));
                val.x += curVal;
                val.y += change_energy_mm(curVal);
            }
        }
        else
        {
            int iy_min = max(0, int(floor((dst.y - startVal.y) / T.y)));
            const float ix_offset = ix_start + ir.x * float(iy_start) + 0.5f;
            const float iz_offset = iz_start + ir.z * float(iy_start) + 0.5f;
            for (int iy = iy_start; iy >= iy_min; iy--)
            {
                curVal = TEX3D(mu, ix_offset - ir.x * float(iy), float(iy) + 0.5f, iz_offset - ir.z * float(iy));
                val.x += curVal;
                val.y += change_energy_mm(curVal);
            }
        }
        const float line_length = sqrtf(1.0f + ir.x * ir.x + ir.z * ir.z) * T.y;
        val.x *= line_length;
        val.y *= line_length;
        return val;
    }
    else
    {
        const float3 ip = make_float3((p.x - startVal.x) / T.x, (p.y - startVal.y) / T.y, (p.z - startVal.z) / T.z);
        const float3 ir = make_float3(r.x / fabsf(r.z), r.y / fabsf(r.z), r.z / fabsf(r.z));
        const int iz_start = max(0, min(N.z - 1, int(floor(0.5f + ip.z))));
        const float tt = (iz_start - ip.z) / ir.z;
        const float ix_start = ip.x + tt * ir.x;
        const float iy_start = ip.y + tt * ir.y;

        float curVal = 0.0f;
        float2 val = make_float2(0.0f, 0.0f);
        if (r.z > 0.0f)
        {
            int iz_max = min(N.z - 1, int(ceil((dst.z - startVal.z) / T.z)));
            const float ix_offset = ix_start - ir.x * float(iz_start) + 0.5f;
            const float iy_offset = iy_start - ir.y * float(iz_start) + 0.5f;
            for (int iz = iz_start; iz <= iz_max; iz++)
            {
                curVal = TEX3D(mu, ix_offset + ir.x * float(iz), iy_offset + ir.y * float(iz), float(iz) + 0.5f);
                val.x += curVal;
                val.y += change_energy_mm(curVal);
            }
        }
        else
        {
            int iz_min = max(0, int(floor((dst.z - startVal.z) / T.z)));
            const float ix_offset = ix_start + ir.x * float(iz_start) + 0.5f;
            const float iy_offset = iy_start + ir.y * float(iz_start) + 0.5f;
            for (int iz = iz_start; iz >= iz_min; iz--)
            {
                curVal = TEX3D(mu, ix_offset - ir.x * float(iz), iy_offset - ir.y * float(iz), float(iz) + 0.5f);
                val.x += curVal;
                val.y += change_energy_mm(curVal);
            }
        }
        const float line_length = sqrtf(1.0f + ir.x * ir.x + ir.y * ir.y) * T.z;
        val.x *= line_length;
        val.y *= line_length;
        return val;
    }
}

// Multi-material analogue of divergentBeamTransform: ray-march from p in direction r through the
// fL volume, accumulating the local total LAC (b_L*fL + b_H*T(fL)) * dl until the running optical
// depth crosses tau. Returns the world position where that threshold is reached (or p if the ray
// immediately exits the grid). bL/bH are the SMB coefficients at the photon's current energy.
__device__ float3 divergentBeamTransform_multimaterial(TEX_DATA mu, const int4 N, const float4 T, const float4 startVal, const float3 p, const float3 r, const float tau, const float bL, const float bH)
{
    const float3 ip = make_float3((p.x - startVal.x) / T.x, (p.y - startVal.y) / T.y, (p.z - startVal.z) / T.z);

    if (fabsf(r.x) >= fabsf(r.y) && fabsf(r.x) >= fabsf(r.z))
    {
        const float3 ir = make_float3(r.x / fabsf(r.x), r.y / fabsf(r.x), r.z / fabsf(r.x));
        const float increment_amount = sqrtf(1.0f + ir.y * ir.y + ir.z * ir.z) * T.x;
        const float threshold = tau / increment_amount;
        const int ix_start = max(0, min(N.x - 1, int(floor(0.5f + ip.x))));
        const float tt = (ix_start - ip.x) / ir.x;
        const float iy_start = ip.y + tt * ir.y;
        const float iz_start = ip.z + tt * ir.z;

        float val = 0.0f;
        if (r.x > 0.0f)
        {
            if (ip.x >= float(N.x) - 0.5f) return p;
            int ix_max = N.x - 1;
            const float iy_offset = iy_start - ir.y * float(ix_start) + 0.5f;
            const float iz_offset = iz_start - ir.z * float(ix_start) + 0.5f;
            float ix_opt = float(N.x);
            for (int ix = ix_start; ix <= ix_max; ix++)
            {
                const float next = localLAC_mm(TEX3D(mu, float(ix) + 0.5f, iy_offset + ir.y * float(ix), iz_offset + ir.z * float(ix)), bL, bH);
                val += next;
                if (val > threshold)
                {
                    ix_opt = float(ix) - (val - threshold) / next;
                    break;
                }
            }
            return make_float3(ix_opt * T.x + startVal.x, (iy_offset + ir.y * ix_opt - 0.5f) * T.y + startVal.y, (iz_offset + ir.z * ix_opt - 0.5f) * T.z + startVal.z);
        }
        else
        {
            if (ip.x <= -0.5f) return p;
            int ix_min = 0;
            const float iy_offset = iy_start + ir.y * float(ix_start) + 0.5f;
            const float iz_offset = iz_start + ir.z * float(ix_start) + 0.5f;
            float ix_opt = float(-1);
            for (int ix = ix_start; ix >= ix_min; ix--)
            {
                const float next = localLAC_mm(TEX3D(mu, float(ix) + 0.5f, iy_offset - ir.y * float(ix), iz_offset - ir.z * float(ix)), bL, bH);
                val += next;
                if (val > threshold)
                {
                    ix_opt = float(ix) + (val - threshold) / next;
                    break;
                }
            }
            return make_float3(ix_opt * T.x + startVal.x, (iy_offset - ir.y * ix_opt - 0.5f) * T.y + startVal.y, (iz_offset - ir.z * ix_opt - 0.5f) * T.z + startVal.z);
        }
    }
    else if (fabsf(r.y) >= fabsf(r.z))
    {
        const float3 ir = make_float3(r.x / fabsf(r.y), r.y / fabsf(r.y), r.z / fabsf(r.y));
        const float increment_amount = sqrtf(1.0f + ir.x * ir.x + ir.z * ir.z) * T.y;
        const float threshold = tau / increment_amount;
        const int iy_start = max(0, min(N.y - 1, int(floor(0.5f + ip.y))));
        const float tt = (iy_start - ip.y) / ir.y;
        const float ix_start = ip.x + tt * ir.x;
        const float iz_start = ip.z + tt * ir.z;

        float val = 0.0f;
        if (r.y > 0.0f)
        {
            if (ip.y >= float(N.y) - 0.5f) return p;
            int iy_max = N.y - 1;
            const float ix_offset = ix_start - ir.x * float(iy_start) + 0.5f;
            const float iz_offset = iz_start - ir.z * float(iy_start) + 0.5f;
            float iy_opt = float(N.y);
            for (int iy = iy_start; iy <= iy_max; iy++)
            {
                const float next = localLAC_mm(TEX3D(mu, ix_offset + ir.x * float(iy), float(iy) + 0.5f, iz_offset + ir.z * float(iy)), bL, bH);
                val += next;
                if (val > threshold)
                {
                    iy_opt = float(iy) - (val - threshold) / next;
                    break;
                }
            }
            return make_float3((ix_offset - 0.5f + ir.x * float(iy_opt)) * T.x + startVal.x, (float(iy_opt)) * T.y + startVal.y, (iz_offset - 0.5f + ir.z * float(iy_opt)) * T.z + startVal.z);
        }
        else
        {
            if (ip.y <= -0.5f) return p;
            int iy_min = 0;
            const float ix_offset = ix_start + ir.x * float(iy_start) + 0.5f;
            const float iz_offset = iz_start + ir.z * float(iy_start) + 0.5f;
            float iy_opt = float(-1);
            for (int iy = iy_start; iy >= iy_min; iy--)
            {
                const float next = localLAC_mm(TEX3D(mu, ix_offset - ir.x * float(iy), float(iy) + 0.5f, iz_offset - ir.z * float(iy)), bL, bH);
                val += next;
                if (val > threshold)
                {
                    iy_opt = float(iy) + (val - threshold) / next;
                    break;
                }
            }
            return make_float3((ix_offset - 0.5f - ir.x * float(iy_opt)) * T.x + startVal.x, (float(iy_opt)) * T.y + startVal.y, (iz_offset - 0.5f - ir.z * float(iy_opt)) * T.z + startVal.z);
        }
    }
    else
    {
        const float3 ir = make_float3(r.x / fabsf(r.z), r.y / fabsf(r.z), r.z / fabsf(r.z));
        const float increment_amount = sqrtf(1.0f + ir.x * ir.x + ir.y * ir.y) * T.z;
        const float threshold = tau / increment_amount;
        const int iz_start = max(0, min(N.z - 1, int(floor(0.5f + ip.z))));
        const float tt = (iz_start - ip.z) / ir.z;
        const float ix_start = ip.x + tt * ir.x;
        const float iy_start = ip.y + tt * ir.y;

        float val = 0.0f;
        if (r.z > 0.0f)
        {
            if (ip.z >= float(N.z) - 0.5f) return p;
            int iz_max = N.z - 1;
            const float ix_offset = ix_start - ir.x * float(iz_start) + 0.5f;
            const float iy_offset = iy_start - ir.y * float(iz_start) + 0.5f;
            float iz_opt = float(N.z);
            for (int iz = iz_start; iz <= iz_max; iz++)
            {
                const float next = localLAC_mm(TEX3D(mu, ix_offset + ir.x * float(iz), iy_offset + ir.y * float(iz), float(iz) + 0.5f), bL, bH);
                val += next;
                if (val > threshold)
                {
                    iz_opt = float(iz) - (val - threshold) / next;
                    break;
                }
            }
            return make_float3((ix_offset - 0.5f + ir.x * float(iz_opt)) * T.x + startVal.x, (iy_offset - 0.5f + ir.y * float(iz_opt)) * T.y + startVal.y, float(iz_opt) * T.z + startVal.z);
        }
        else
        {
            if (ip.z <= -0.5f) return p;
            int iz_min = 0;
            const float ix_offset = ix_start + ir.x * float(iz_start) + 0.5f;
            const float iy_offset = iy_start + ir.y * float(iz_start) + 0.5f;
            float iz_opt = float(-1);
            for (int iz = iz_start; iz >= iz_min; iz--)
            {
                const float next = localLAC_mm(TEX3D(mu, ix_offset - ir.x * float(iz), iy_offset - ir.y * float(iz), float(iz) + 0.5f), bL, bH);
                val += next;
                if (val > threshold)
                {
                    iz_opt = float(iz) + (val - threshold) / next;
                    break;
                }
            }
            return make_float3((ix_offset - 0.5f - ir.x * float(iz_opt)) * T.x + startVal.x, (iy_offset - 0.5f - ir.y * float(iz_opt)) * T.y + startVal.y, float(iz_opt) * T.z + startVal.z);
        }
    }
}

// Multi-material analogue of trackPhoton. Works directly in LAC (1/mm) units: the optical depth
// to the next interaction is sampled as tau = ExpRand(), and the interaction point is found by
// ray-marching the spatially-varying LAC. The interaction type (PE/CS/RS) is then drawn from the
// local per-material components at the photon's current energy.
//
// TODO(physics): this is the core routine to refine together. Open items include:
//   - confirming the LAC-unit optical-depth convention end to end (vs the g/mm^2 sigma + density
//     convention used by the single-material trackPhoton),
//   - whether the per-energy pass-through early-out (rhoL_full) should be recomputed each step.
// Note: the Compton/Rayleigh scatter angle distributions are material-dependent: they are sampled
// from the per-component d_mm_dsigma_Compton / d_mm_dsigma_Rayleigh textures at the local fractional
// material index (the same matIndex used for the PE/CS/RS interaction-type sampling), linearly
// interpolating between the two bracketing basis materials.
__device__ void trackPhoton_multimaterial(float3& position, float3& r, float& energy, int& numberOfInteractions, bool& hasRayleighEvent, TEX_DATA f, const int4 N_f, const float4 T_f, const float4 startVal_f, const uint64 tid, const float2 lineIntegralPair, bool printDebug)
{
    numberOfInteractions = 0;
    hasRayleighEvent = false;

    do
    {
        const float bL = TEX1D(d_mm_b_L, energy + 0.5f);
        const float bH = TEX1D(d_mm_b_H, energy + 0.5f);

        // Total optical depth through the remaining object at the current energy (used only as a
        // first-segment pass-through early-out, matching single-material trackPhoton's rhoL_full).
        const float rhoL_full = bL * lineIntegralPair.x + bH * lineIntegralPair.y;

        // Sample optical depth to next interaction (LAC units => dimensionless optical depth).
        const float tau = ExpRand(tid);
        if (tau > rhoL_full)
        {
            // photon passes through object
            break;
        }

        const float3 nextPos = divergentBeamTransform_multimaterial(f, N_f, T_f, startVal_f, position, r, tau, bL, bH);

        if (insideObject(nextPos) == false)
        {
            if (printDebug)
                printf("photon escaped with energy %f\n", energy);
            position = nextPos;
            break;
        }

        // Determine interaction type from the local per-material LAC components at this point.
        const float fL_local = sample_fL_mm(f, N_f, T_f, startVal_f, nextPos);
        const float matIndex = materialIndex_mm(fL_local);
        const float sigma_PE = sigma_component_mm(0, matIndex, energy);
        const float sigma_CS = sigma_component_mm(1, matIndex, energy);
        const float sigma_RS = sigma_component_mm(2, matIndex, energy);

        const int interactionType = typeRand(sigma_PE, sigma_CS, sigma_RS, tid);

        float theta_new = 0.0f;
        switch (interactionType)
        {
            case 0: // PE
                energy = 0.0f;
                break;
            case 1: // CS
                theta_new = randomComptonScatterAngle_mm(energy, matIndex, tid);
                //energy = ELECTRON_REST_MASS_ENERGY / (ELECTRON_REST_MASS_ENERGY / energy + 1.0f - cosf(theta_new));
                energy *= ELECTRON_REST_MASS_ENERGY / (ELECTRON_REST_MASS_ENERGY + energy*(1.0f-cosf(theta_new)));
                updateTrajectory(r, theta_new, randomAngle(tid));
                numberOfInteractions += 1;
                break;
            case 2: // RS
                theta_new = randomRayleighScatterAngle_mm(energy, matIndex, tid);
                updateTrajectory(r, theta_new, randomAngle(tid));
                hasRayleighEvent = true;
                numberOfInteractions += 1;
                break;
            default:
                theta_new = 0.0f;
        }

        position = nextPos;
        if (numberOfInteractions > d_max_scatters)
        {
            energy = 0.0f;
            break;
        }
    } while (energy > 0.0f);
}

// Multi-material analogue of precompute_line_integral_kernel: computes the primary-ray line-integral
// pair (P*fL, P*T(fL)) for every detector element once per view, storing it as a float2 per pixel.
__global__ void precompute_line_integral_pair_kernel(float2* lineIntegral_map, const int4 N_g, const float4 T_g, const float4 startVal_g, TEX_DATA f, const int4 N_f, const float4 T_f, const float4 startVal_f, const float3 sourcePos, const float3 moduleCenter, const float3 v_vec, const float3 u_vec,
    const int extraCols, const bool padOnLeft)
{
    const uint64 m = threadIdx.x + blockIdx.x * blockDim.x;
    const uint64 n = threadIdx.y + blockIdx.y * blockDim.y;
    if (m >= N_g.y || n >= N_g.z + extraCols) return;
    const uint64 pixel = m * (N_g.z + extraCols) + n;

    float u_shift = 0.0f;
    if (padOnLeft)
        u_shift = -extraCols * T_g.z;
    const float v = m * T_g.y + startVal_g.y;
    const float u = n * T_g.z + startVal_g.z + u_shift;

    const float3 endPos = make_float3(moduleCenter.x + v * v_vec.x + u * u_vec.x, moduleCenter.y + v * v_vec.y + u * u_vec.y, moduleCenter.z + v * v_vec.z + u * u_vec.z);
    lineIntegral_map[pixel] = lineIntegralPair_mm(f, N_f, T_f, startVal_f, sourcePos, endPos);
}

// Photon-parallel multi-material scatter kernel; see scatter_simulation_kernel for the threading model.
// Reads the per-pixel (P*fL, P*T(fL)) pair precomputed by precompute_line_integral_pair_kernel.
__global__ void scatter_simulation_multimaterial_kernel(float* scatter_data, float* scatter_data_high_order, const float2* __restrict__ lineIntegral_map, const int4 N_g, const float4 T_g, const float4 startVal_g, TEX_DATA f, const int4 N_f, const float4 T_f, const float4 startVal_f, const float3 sourcePos, const float3 moduleCenter, const float3 v_vec, const float3 u_vec,
    TEX_DATA source_spectra, TEX_DATA source_energies, TEX_DATA detector_response, const int total_photons_per_pixel, const int photons_per_thread, const int groups_per_pixel,
    const int extraCols, const bool padOnLeft)
{
    const uint64 gid = (uint64)threadIdx.x + (uint64)blockIdx.x * (uint64)blockDim.x;
    const uint64 num_cols = (uint64)(N_g.z + extraCols);
    const uint64 num_pixels = (uint64)N_g.y * num_cols;
    const uint64 total_threads = num_pixels * (uint64)groups_per_pixel;
    if (gid >= total_threads) return;

    const uint64 pixel = gid / (uint64)groups_per_pixel;
    const uint64 group = gid - pixel * (uint64)groups_per_pixel;

    int p0 = int(group) * photons_per_thread;
    if (p0 >= total_photons_per_pixel) return;
    int p1 = p0 + photons_per_thread;
    if (p1 > total_photons_per_pixel) p1 = total_photons_per_pixel;

    // (P*fL, P*T(fL)) through the object along the primary ray, precomputed once per pixel. Combined
    // with the SMB coefficients b_L(gamma), b_H(gamma) inside trackPhoton this gives the total optical
    // depth at any energy. P*fL <= 0 means the ray traverses no material, so no scatter is possible.
    const float2 lineIntegralPair = lineIntegral_map[pixel];
    if (lineIntegralPair.x <= 0.0f)
        return;

    const uint64 m = pixel / num_cols;
    const uint64 n = pixel - m * num_cols;
    const uint64 tid = gid;

    float u_shift = 0.0f;
    if (padOnLeft)
        u_shift = -extraCols * T_g.z;
    const float v = m * T_g.y + startVal_g.y;
    const float u = n * T_g.z + startVal_g.z + u_shift;

    const float3 n_vec = make_float3(u_vec.y * v_vec.z - u_vec.z * v_vec.y,
        u_vec.z * v_vec.x - u_vec.x * v_vec.z,
        u_vec.x * v_vec.y - u_vec.y * v_vec.x);

    const float3 endPos = make_float3(moduleCenter.x + v * v_vec.x + u * u_vec.x, moduleCenter.y + v * v_vec.y + u * u_vec.y, moduleCenter.z + v * v_vec.z + u * u_vec.z);
    float3 r_init = make_float3(endPos.x - sourcePos.x, endPos.y - sourcePos.y, endPos.z - sourcePos.z);
    const float r_mag_inv = rsqrtf(r_init.x * r_init.x + r_init.y * r_init.y + r_init.z * r_init.z);
    r_init.x *= r_mag_inv;
    r_init.y *= r_mag_inv;
    r_init.z *= r_mag_inv;

    const float T_u_inv = 1.0f / T_g.z;
    const float T_v_inv = 1.0f / T_g.y;

    // Map the flat photon index range [p0, p1) onto source energy bins (see scatter_simulation_kernel).
    int ienergy = 0;
    int cum = 0;
    int cnt = (N_g.w > 0) ? int(0.5f + TEX1D(source_spectra, 0.5f)) : 0;
    while (ienergy < N_g.w - 1 && p0 >= cum + cnt)
    {
        cum += cnt;
        ienergy++;
        cnt = int(0.5f + TEX1D(source_spectra, ienergy + 0.5f));
    }

    bool printDebug = false;
    for (int ievent = p0; ievent < p1; ievent++)
    {
        while (ienergy < N_g.w - 1 && ievent >= cum + cnt)
        {
            cum += cnt;
            ienergy++;
            cnt = int(0.5f + TEX1D(source_spectra, ienergy + 0.5f));
        }

        int numberOfInteractions;
        bool hasRayleighEvent;
        float energy = TEX1D(source_energies, ienergy + 0.5f);
        float3 startPos = make_float3(sourcePos.x, sourcePos.y, sourcePos.z);
        float3 r = make_float3(r_init.x, r_init.y, r_init.z);

        trackPhoton_multimaterial(startPos, r, energy, numberOfInteractions, hasRayleighEvent, f, N_f, T_f, startVal_f, tid, lineIntegralPair, printDebug);

        if (energy > 0.0f && d_min_scatters <= numberOfInteractions)
        {
            // See the single-material kernel for the rationale: the escaping photon's exit point may lie
            // behind the detector plane (volume box extends past it at oblique views / larger sod), so the
            // backscatter rejection must account for which side of the plane the exit point is on.
            const float startDotN = (startPos.x - moduleCenter.x) * n_vec.x + (startPos.y - moduleCenter.y) * n_vec.y + (startPos.z - moduleCenter.z) * n_vec.z;
            const float srcDotN = (sourcePos.x - moduleCenter.x) * n_vec.x + (sourcePos.y - moduleCenter.y) * n_vec.y + (sourcePos.z - moduleCenter.z) * n_vec.z;
            const float t = -startDotN / (r.x * n_vec.x + r.y * n_vec.y + r.z * n_vec.z);
            const bool startInFront = (startDotN * srcDotN > 0.0f);
            if ((t >= 0.0f && startInFront) || (t <= 0.0f && !startInFront))
            {
                const float3 finalPosition = make_float3(startPos.x + t * r.x, startPos.y + t * r.y, startPos.z + t * r.z);
                const float u_ind = ((finalPosition.x - moduleCenter.x) * u_vec.x + (finalPosition.y - moduleCenter.y) * u_vec.y + (finalPosition.z - moduleCenter.z) * u_vec.z - startVal_g.z) * T_u_inv;
                const float v_ind = ((finalPosition.x - moduleCenter.x) * v_vec.x + (finalPosition.y - moduleCenter.y) * v_vec.y + (finalPosition.z - moduleCenter.z) * v_vec.z - startVal_g.y) * T_v_inv;
                if (-0.5f < u_ind && u_ind < N_g.z - 0.5f && -0.5f < v_ind && v_ind < N_g.y - 0.5f)
                {
                    const uint64 ind = int(0.5f + v_ind) * N_g.z + int(0.5f + u_ind);
                    if (numberOfInteractions > 1)
                        atomicAdd(&scatter_data_high_order[ind], TEX1D(detector_response, energy + 0.5f));
                    else
                        atomicAdd(&scatter_data[ind], TEX1D(detector_response, energy + 0.5f));
                }
            }
        }
    }
}

bool scatterSimulation_multimaterial(parameters* params, float* t, float* f, float* source, float* energies, int N_energies, float* detector, float reference_energy, const char** chemForms, int num_materials, float* densities, float* b_L, float* b_H, bool data_on_cpu, int num_photons_per_pixel, int min_scatters, int max_scatters)
{
    // f = fL, the LAC volume at reference_energy (gammaL).
    if (params == NULL || t == NULL || f == NULL || source == NULL || energies == NULL || N_energies <= 0 || detector == NULL || chemForms == NULL)
        return false;
    if (densities == NULL || b_L == NULL || b_H == NULL)
    {
        printf("Error: scatterSimulation_multimaterial requires basis-material densities and SMB basis functions.\n");
        return false;
    }
    if (num_materials < 2 || num_materials > 4)
    {
        printf("Error: scatterSimulation_multimaterial requires between 2 and 4 basis materials.\n");
        return false;
    }
    min_scatters = max(0, min(min_scatters, 100));
    max_scatters = max(0, min(max_scatters, 100));
    bool do_smoothing = true;
    cudaSetDevice(params->whichGPU);

    int max_energy = int(ceil(energies[N_energies - 1]));
    float peak_energy = energies[N_energies - 1];

    // Photon-parallel scatter simulation: see scatterSimulation for the threading model. The primary-ray
    // line-integral pair is precomputed once per pixel (precompute_line_integral_pair_kernel) and the
    // photon histories are spread across many threads, one curand state per thread.
    const int photons_per_thread = 64;
    bool padOnLeft;
    int extraCols = zeroPadForOffsetScan_numberOfColsToAdd(params, padOnLeft);
    int total_photons_per_pixel = 0;
    {
        float src_sum = 0.0f;
        for (int i = 0; i < N_energies; i++)
            src_sum += source[i];
        for (int i = 0; i < N_energies; i++)
            total_photons_per_pixel += int(0.5f + source[i] * num_photons_per_pixel / src_sum);
    }
    const uint64 num_pixels = (uint64)params->numRows * (uint64)(params->numCols + extraCols);
    int groups_per_pixel = (total_photons_per_pixel + photons_per_thread - 1) / photons_per_thread;
    if (groups_per_pixel < 1)
        groups_per_pixel = 1;
    const uint64 total_threads = num_pixels * (uint64)groups_per_pixel;

    scatterSimulationTables scatterTables;
    scatterTables.initialize_multimaterial(params, chemForms, num_materials, max_energy, reference_energy, peak_energy, densities, b_L, b_H, min_scatters, max_scatters, (int)total_threads);

    if (params->geometry != parameters::MODULAR)
    {
        printf("Error: scatter estimation algorithm only implemented for modular-beam geometries. Please convert to modular-beam before running this algorithm.\n");
        return false;
    }

    cudaError_t cudaStatus;

    // Allocate projection data on GPU
    int4 N_g; float4 T_g; float4 startVal_g;
    setProjectionGPUparams(params, N_g, T_g, startVal_g, false);
    N_g.w = N_energies;
    float* dev_g = 0;
    if (data_on_cpu)
    {
        if ((cudaStatus = cudaMalloc((void**)&dev_g, params->projectionData_numberOfElements() * sizeof(float))) != cudaSuccess)
        {
            fprintf(stderr, "scatterSimulation_multimaterial: cudaMalloc(projections) failed!\n");
        }
    }
    else
        dev_g = t;

    // Allocate volume data (fL) on GPU as a texture
    int4 N_f; float4 T_f; float4 startVal_f;
    setVolumeGPUparams(params, N_f, T_f, startVal_f);

    // Precompute reciprocal voxel sizes so sample_fL_mm avoids per-sample divisions.
    float4 inv_T_f = make_float4(
        (T_f.x != 0.0f) ? (1.0f / T_f.x) : 0.0f,
        (T_f.y != 0.0f) ? (1.0f / T_f.y) : 0.0f,
        (T_f.z != 0.0f) ? (1.0f / T_f.z) : 0.0f,
        0.0f);
    cudaMemcpyToSymbol(d_mm_inv_T_f, &inv_T_f, sizeof(float4));

    float* dev_f = 0;
    if (data_on_cpu)
        dev_f = copyVolumeDataToGPU(f, params, params->whichGPU);
    else
        dev_f = f;
    TEX_DATA f_data_txt = {};
    TEX_ARRAY f_data_array = loadTexture(f_data_txt, dev_f, N_f, false, true, bool(params->volumeDimensionOrder == 1));

    float source_sum = 0.0;
    for (int i = 0; i < N_energies; i++)
        source_sum += source[i];
    float* source_scaled = new float[N_energies];
    for (int i = 0; i < N_energies; i++)
        source_scaled[i] = source[i] * num_photons_per_pixel / source_sum;

    float air_scan_inv = 0.0;
    for (int i = 0; i < N_energies; i++)
    {
        float gamma = energies[i];
        air_scan_inv += floor(0.5 + source_scaled[i]) * detector[int(0.5 + gamma)];
    }
    air_scan_inv = 1.0 / air_scan_inv;

    float* dev_source = copy1DdataToGPU(source_scaled, N_energies, params->whichGPU);
    TEX_DATA source_txt = {};
    TEX_ARRAY source_array = loadTexture1D(source_txt, dev_source, N_energies, false, true);
    delete [] source_scaled;

    float* dev_energies = copy1DdataToGPU(energies, N_energies, params->whichGPU);
    TEX_DATA energies_txt = {};
    TEX_ARRAY energies_array = loadTexture1D(energies_txt, dev_energies, N_energies, false, true);

    float* dev_detector = copy1DdataToGPU(detector, max_energy + 1, params->whichGPU);
    TEX_DATA detector_txt = {};
    TEX_ARRAY detector_array = loadTexture1D(detector_txt, dev_detector, max_energy + 1, true, true);

    float* dev_high_order = 0;
    if (cudaSuccess != cudaMalloc((void**)&dev_high_order, params->numAngles * params->numRows * params->numCols * sizeof(float)))
        fprintf(stderr, "scatterSimulation_multimaterial: cudaMalloc failed!\n");

    cudaMemset(dev_g, 0, params->numAngles * params->numRows * params->numCols * sizeof(float));
    cudaMemset(dev_high_order, 0, params->numAngles * params->numRows * params->numCols * sizeof(float));

    // Per-pixel primary-ray line-integral pair, recomputed for each view.
    float2* dev_lineIntegral = 0;
    if (cudaSuccess != cudaMalloc((void**)&dev_lineIntegral, num_pixels * sizeof(float2)))
        fprintf(stderr, "scatterSimulation_multimaterial: cudaMalloc(dev_lineIntegral) failed!\n");

    // 2D launch for the per-pixel line-integral precompute (one thread per detector element).
    dim3 dimBlock(8, 8);
    dim3 dimGrid(int(ceil(double(params->numRows) / double(dimBlock.x))), int(ceil(double(params->numCols + extraCols) / double(dimBlock.y))));

    // 1D launch for the photon-parallel scatter kernel (one thread per photon-group).
    const int photon_block = PHOTON_BLOCK;
    const int photon_grid = int((total_threads + (uint64)photon_block - 1) / (uint64)photon_block);

    setup_uRand1D <<< photon_grid, photon_block >>> ((int)total_threads, time(NULL));

    printf("Monte-Carlo multi-material simulation...\n");
    for (int i = 0; i < params->numAngles; i++)
    {
        float3 sourcePosition = make_float3(params->sourcePositions[3 * i + 0], params->sourcePositions[3 * i + 1], params->sourcePositions[3 * i + 2]);
        float3 moduleCenter = make_float3(params->moduleCenters[3 * i + 0], params->moduleCenters[3 * i + 1], params->moduleCenters[3 * i + 2]);
        float3 rowVector = make_float3(params->rowVectors[3 * i + 0], params->rowVectors[3 * i + 1], params->rowVectors[3 * i + 2]);
        float3 colVector = make_float3(params->colVectors[3 * i + 0], params->colVectors[3 * i + 1], params->colVectors[3 * i + 2]);

        const uint64 ind_offset = uint64(i) * uint64(params->numRows * params->numCols);

        precompute_line_integral_pair_kernel <<< dimGrid, dimBlock >>> (dev_lineIntegral, N_g, T_g, startVal_g, f_data_txt, N_f, T_f, startVal_f, sourcePosition, moduleCenter, rowVector, colVector, extraCols, padOnLeft);
        scatter_simulation_multimaterial_kernel <<< photon_grid, photon_block >>> (&dev_g[ind_offset], &dev_high_order[ind_offset], dev_lineIntegral, N_g, T_g, startVal_g, f_data_txt, N_f, T_f, startVal_f, sourcePosition, moduleCenter, rowVector, colVector, source_txt, energies_txt, detector_txt, total_photons_per_pixel, photons_per_thread, groups_per_pixel, extraCols, padOnLeft);
    }
    cudaStatus = cudaDeviceSynchronize();
    printf("done\n");
    cudaFree(dev_lineIntegral);

    if (max_scatters > 1)
    {
        if (do_smoothing && min_scatters < 2)
            blurFilter(dev_high_order, params->numAngles, params->numRows, params->numCols, 8.0, 3, 0, false, params->whichGPU);
        add(dev_g, dev_high_order, make_int3(params->numAngles, params->numRows, params->numCols), params->whichGPU);
    }
    cudaFree(dev_high_order);

    scale(dev_g, air_scan_inv, make_int3(params->numAngles, params->numRows, params->numCols), params->whichGPU);

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "scatterSimulation_multimaterial: kernel failed!\n");
        fprintf(stderr, "error name: %s\n", cudaGetErrorName(cudaStatus));
        fprintf(stderr, "error msg: %s\n", cudaGetErrorString(cudaStatus));
    }

    // Clean up
    freeTexture(f_data_array, f_data_txt);
    if (data_on_cpu)
        cudaFree(dev_f);

    freeTexture(source_array, source_txt);
    cudaFree(dev_source);

    freeTexture(energies_array, energies_txt);
    cudaFree(dev_energies);

    freeTexture(detector_array, detector_txt);
    cudaFree(dev_detector);

    if (data_on_cpu)
        pullProjectionDataFromGPU(t, params, dev_g, params->whichGPU);
    else
        t = dev_g;

    if (data_on_cpu)
        cudaFree(dev_g);

    return true;
}
