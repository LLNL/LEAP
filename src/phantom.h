////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
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


class phantom
{
public:
    phantom();
    ~phantom();

    bool addObject(float* f, parameters*, int type, float* c, float* r, float val, float* A = NULL, float* clip = NULL, int oversampling = 1);

private:
    float x_inv(float);
    float y_inv(float);
    float z_inv(float);

    float clipCone[2];

    float x_0, y_0, z_0;
    int numX, numY, numZ;
    float T_x, T_y, T_z;

    bool isInside(float x, float y, float z, int type, float* clip);

    enum objectType_list {ELLIPSOID=0, PARALLELEPIPED=1, CYLINDER_X=2, CYLINDER_Y=3, CYLINDER_Z=4, CONE_X=5, CONE_Y=6, CONE_Z=7};
    parameters* params;
};

#endif
