////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
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

    bool addObject(float* f, parameters*, int type, float* c, float* r, float val);

private:
    float x_inv(float);
    float y_inv(float);
    float z_inv(float);

    enum objectType_list {ELLIPSOID=0, PARALLELEPIPED, CYLINDER_X, CYLINDER_Y, CYLINDER_Z};
    parameters* params;
};

#endif
