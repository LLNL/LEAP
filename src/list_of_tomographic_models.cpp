////////////////////////////////////////////////////////////////////////////////
// Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
// LEAP project developers. See the LICENSE file for details.
// SPDX-License-Identifier: MIT
//
// LivermorE AI Projector for Computed Tomography (LEAP)
////////////////////////////////////////////////////////////////////////////////

#include "list_of_tomographic_models.h"
#include "tomographic_models.h"

listOfTomographicModels::listOfTomographicModels()
{
}

listOfTomographicModels::~listOfTomographicModels()
{
    clear();
}

void listOfTomographicModels::clear()
{
    for (int i = 0; i < int(list.size()); i++)
    {
        tomographicModels* p_model = list[i];
        delete p_model;
        list[i] = NULL;
    }
    list.clear();
}

int listOfTomographicModels::append()
{
    tomographicModels* p_model = new tomographicModels;
    list.push_back(p_model);
    return int(list.size()-1);
}

int listOfTomographicModels::size()
{
    return int(list.size());
}

tomographicModels* listOfTomographicModels::get(int i)
{
    if (list.size() == 0)
    {
        append();
        return list[0];
    }
    else
        return list[i % list.size()];
}
