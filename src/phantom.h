////////////////////////////////////////////////////////////////////////////////
// Copyright 2023-2024 Kyle Champley
// See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
// sets voxelized phantoms of 3D geometric shapes to assist in algorithm
// development and testing
////////////////////////////////////////////////////////////////////////////////
#ifndef __PHANTOM_H
#define __PHANTOM_H

#ifdef WIN32
#pragma once
#endif


#include <stdlib.h>
#include "parameters.h"

/**
 * This class provides CPU-based implementations (accelerated by OpenMP) to specify voxelized phantoms as a collection of geometric shapes.
 */

class geometricObject
{
public:
    geometricObject();
    geometricObject(int type_in, float* c_in, float* r_in, float val_in, float* A_in, float* clip_in);
    ~geometricObject();

    void reset();
    bool init(int type_in, float* c_in, float* r_in, float val_in, float* A_in = NULL, float* clip_in = NULL);

    bool intersectionEndPoints(double* p, double* r, double* ts);
    bool intersectionEndPoints_centeredAndNormalized(double* p, double* r, double* ts);
    bool parametersOfIntersection_1D(double* ts, double p, double r);
    bool parametersOfClippingPlaneIntersections(double* ts, double* p, double* r);

    int type;
    float centers[3];
    float radii[3];
    float val;
    float A[9];
    float clippingPlanes[6][4];
    
    bool isRotated;
    int numClippingPlanes;
    float clipCone[2];

    void restore_cone_params();

private:
    double dot(double* x, double* y, int N = 3);
    float centers_save[3];
    float radii_save[3];
};

class phantom
{
public:

    // Constructor and destructor; these do nothing
    phantom();
    phantom(parameters*);
    ~phantom();

    /**
     * \fn          addObject
     * \brief       Changes the voxels values to a specified value inside a 3D geometric object
                    (any voxel values inside the object before this function are ignored, i.e., this function does not accumulate the values, it replaces them)
     * \param[in]   f, pointer to the volume data (on the CPU)
     * \param[in]   params, pointer to an instance of the parameters class
     * \param[in]   type, the enumerated object type; see enum objectType_list
     * \param[in]   c, pointer to a three-element array of the (x,y,z) coordinates of the center of the object
     * \param[in]   r, pointer to a three-element array of the (x,y,z) coordinates of the radii of the object
     * \param[in]   val, value of the voxels inside the object
     * \param[in]   A, pointer to a 3X3 rotation matrix of the object
     * \param[in]   clip, pointer to a three-element array that specifies the clipping planes along the (x,y,z) coordinates
     * \param[in]   oversampling, voxels are broken up into oversampling X oversampling X oversampling subvoxels to model the partial volume effect on voxels on the edge of the object
     * \return      returns true if all input are valid and CT volume parameter values are defined and valid, false otherwise
     */
    bool addObject(float* f, parameters* params, int type, float* c, float* r, float val, float* A = NULL, float* clip = NULL, int oversampling = 1);

    /**
     * \fn          addObject
     * \brief       Changes the voxels values to a specified value inside a 3D geometric object
                    (any voxel values inside the object before this function are ignored, i.e., this function does not accumulate the values, it replaces them)
     * \param[in]   type, the enumerated object type; see enum objectType_list
     * \param[in]   c, pointer to a three-element array of the (x,y,z) coordinates of the center of the object
     * \param[in]   r, pointer to a three-element array of the (x,y,z) coordinates of the radii of the object
     * \param[in]   val, value of the voxels inside the object
     * \param[in]   A, pointer to a 3X3 rotation matrix of the object
     * \param[in]   clip, pointer to a three-element array that specifies the clipping planes along the (x,y,z) coordinates
     * \return      returns true if all input are valid and CT volume parameter values are defined and valid, false otherwise
     */
    bool addObject(int type, float* c, float* r, float val, float* A = NULL, float* clip = NULL);

    void clearObjects();

    double lineIntegral(double* p, double* r);

    bool synthesizeSymmetry(float* f_radial, float* f);

    bool scale_phantom(float scale_x, float scale_y, float scale_z);
    bool voxelize(float* f, parameters* params_in, int oversampling = 1);

    // enumerated list of all the 3D geometric shapes that are supported
    enum objectType_list { ELLIPSOID = 0, PARALLELEPIPED = 1, CYLINDER_X = 2, CYLINDER_Y = 3, CYLINDER_Z = 4, CONE_X = 5, CONE_Y = 6, CONE_Z = 7 };

    std::vector<geometricObject> objects;

    bool makeTempData(int num_threads);

private:

    /**
     * \fn          x_inv
     * \return      returns (x_val - x_0) / T_x
     */
    float x_inv(float x_val);

    /**
     * \fn          y_inv
     * \return      returns (y_val - y_0) / T_y
     */
    float y_inv(float y_val);

    /**
     * \fn          z_inv
     * \return      returns (z_val - z_0) / T_z
     */
    float z_inv(float z_val);

    float clipCone[2];

    float x_0; // copy of parameters::x_0()
    float y_0; // copy of parameters::y_0()
    float z_0; // copy of parameters::z_0()

    int numX; // copy of parameters::numX
    int numY; // copy of parameters::numY
    int numZ; // copy of parameters::numZ
    float T_x; // copy of parameters::voxelWidth
    float T_y; // copy of parameters::voxelWidth
    float T_z; // copy of parameters::voxelHeight

    /**
     * \fn          isInside
     * \brief       This function is called by addObject, where in addObject, the coordinate system is shifted so the object is centered on the origin,
     *              scaled, so that all the object axes are 1.0, and rotated.  Then this function is called to test whether the given location is inside
     *              or outside the given shape.
     * \param[in]   x, x-coordinate
     * \param[in]   y, y-coordinate
     * \param[in]   z, z-coordinate
     * \param[in]   type, the enumerated object type; see enum objectType_list
     * \param[in]   clip, pointer to a three-element array that specifies the clipping planes along the (x,y,z) coordinates
     * \return      returns true the given (x,y,z) coordinates are inside the the shifted, rotated, and normalized 3D geometric shape
     */
    bool isInside(float x, float y, float z, int type, float* clip);

    int* intData;
    double* floatData;

    // local copy of the pointer to the parameters class that is passed by the addObject function
    parameters* params;
};

#endif
