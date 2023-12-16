////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
////////////////////////////////////////////////////////////////////////////////
#ifndef __LIST_OF_TOMOGRAPHIC_MODELS_H
#define __LIST_OF_TOMOGRAPHIC_MODELS_H

#ifdef WIN32
#pragma once
#endif

//*
#include <vector>
#include <stdlib.h>
class tomographicModels;

class listOfTomographicModels
{
public:
    listOfTomographicModels();
    ~listOfTomographicModels();

    void clear();
    int append();
    tomographicModels* get(int);
    
private:
    std::vector<tomographicModels*> list;
};
//*/

#endif
