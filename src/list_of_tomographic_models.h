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

#include <vector>
#include <stdlib.h>
class tomographicModels;

/**
 *  listOfTomographicModels class
 * This class maintains several instances of the tomographicModels class in a stack, where each instance allows users to specify unique
 * CT geometry and CT volume parameters.
 */

class listOfTomographicModels
{
public:
    // Constructor; these do nothing
    listOfTomographicModels();

    // Destructor, just called clear()
    ~listOfTomographicModels();

    /**
     * \fn          clear
     * \brief       Clears the list member variable and calls the tomographicModels destructor for all elements in the list.
     */
    void clear();

    /**
     * \fn          append
     * \brief       Adds another instance of tomographicModels class to the end of the list
     */
    int append();

    /**
     * \fn          size
     * \return      returns list.size()
     */
    int size();

    /**
     * \fn          get
     * \param[in]   i, the index of the list to get
     * \return      returns list[i % size()]
     */
    tomographicModels* get(int i);
    
private:

    // Tracks a list (as a stack) of tomographicModels instances
    std::vector<tomographicModels*> list;
};

#endif
