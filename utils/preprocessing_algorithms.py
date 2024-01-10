################################################################################
# Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
# LEAP project developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# LivermorE AI Projector for Computed Tomography (LEAP)
#
# This file contains preprocessing algorithms for CT projection data
# All algorithms assume the input data is in attenuation space (-log of transmission)
# If one does not have enough CPU memory to process the entire data set, then we
# recommend splitting the data across projections for the outlierCorrection
# algorithms and splitting the data across detector rows for the ringRemoval
# algorithms.
# For beam hardening correction, see the demo_leapctype/test_beam_hardening.py
#
# Unfortunately at this time these routines only work with numpy arrays.
# Please submit a feature request if you want these to work for PyTorch tensors.
################################################################################

import sys
import os
import time
import numpy as np
from leapctype import *

def outlierCorrection(leapct, g, threshold=0.03, windowSize=3):
    """Removes outliers (zingers) from CT projections
    
    Assumes the input data is in attenuation space.
    No LEAP parameters need to be set for this function to work 
    and can be applied to any CT geometry type.
    This algorithm processes each projection independently
    and removes outliers by a thresholded median filter
    
    Args:
        leapct (tomographicModels object): This is just needed to access LEAP algorithms
        g (contiguous float32 numpy array): attenuation projection data
        threshold (float): A pixel will be replaced by the median of its neighbors if |g - median(g)|/median(g) > threshold
        windowSize (int): The window size of the median filter applied is windowSize x windowSize
        
    Returns:
        The corrected data
    """
    # This algorithm processes each transmission
    g = np.exp(-g)
    leapct.MedianFilter2D(g, threshold, windowSize)
    g = -np.log(g)
    return g

def outlierCorrection_highEnergy(leapct, g):
    """Removes outliers (zingers) from CT projections
    
    Assumes the input data is in attenuation space.
    No LEAP parameters need to be set for this function to work 
    and can be applied to any CT geometry type.
    This algorithm processes each projection independently
    and removes outliers by a series of three thresholded median filters
    
    This outlier correction algorithm should most be used for MV CT or neutron CT
    where outliers effect a larger neighborhood of pixels.
    
    Args:
        leapct (tomographicModels object): This is just needed to access LEAP algorithms
        g (contiguous float32 numpy array): attenuation projection data
        
    Returns:
        The corrected data
    """
    g = np.exp(-g)
    leapct.MedianFilter2D(g, 0.08, 7)
    leapct.MedianFilter2D(g, 0.024, 5)
    leapct.MedianFilter2D(g, 0.0032, 3)
    g = -np.log(g)
    return g
    

def ringRemoval_fast(leapct, g, delta, numIter, maxChange):
    """Removes detector pixel-to-pixel gain variations that cause ring artifacts in reconstructed images
    
    Assumes the input data is in attenuation space.
    No LEAP parameters need to be set for this function to work 
    and can be applied to any CT geometry type.
    This algorithm is effective at removing ring artifacts and runs fast, but can
    sometimes create new rings of tangents of sharp transitions.
    
    Args:
        leapct (tomographicModels object): This is just needed to access LEAP algorithms
        g (contiguous float32 numpy array): attenuation projection data
        delta (float): The delta parameter of the Total Variation Functional
        numIter (int): Number of iterations
        maxChange (float): An upper limit on the maximum difference that can be applied to a detector pixels
    
    Returns:
        The corrected data
    """
    g_sum = np.zeros((1,g.shape[1],g.shape[2]), dtype=np.float32)
    g_sum[0,:] = np.mean(g,axis=0)
    g_sum_save = g_sum.copy()
    leapct.diffuse(g_sum,delta,numIter)
    gainMap = g_sum - g_sum_save
    
    gainMap[gainMap>maxChange] = maxChange
    gainMap[gainMap<-maxChange] = -maxChange
    g = g + gainMap[None,:,:]
    return g

def ringRemoval(leapct, g, delta, beta, numIter):
    """Removes detector pixel-to-pixel gain variations that cause ring artifacts in reconstructed images
    
    Assumes the input data is in attenuation space.
    No LEAP parameters need to be set for this function to work 
    and can be applied to any CT geometry type.
    This algorithm is effective at removing ring artifacts without creating new ring artifacts, 
    but is computationally expensive.
    
    Args:
        leapct (tomographicModels object): This is just needed to access LEAP algorithms
        g (contiguous float32 numpy array): attenuation projection data
        delta (float): The delta parameter of the Total Variation Functional
        beta (float): The strength of the regularization
        numIter (int): Number of iterations
    
    Returns:
        The corrected data
    """
    g_0 = g.copy()
    #Dg_sum = np.zeros((1,g.shape[1],g.shape[2]), dtype=np.float32)
    for n in range(numIter):
        Dg = leapct.TVgradient(g, delta, beta)
        Dg_sum = np.mean(Dg,axis=0)
        Dg[:] = Dg_sum[None,:,:]

        Dg[:] = g[:] - g_0[:] + Dg[:]
        
        num = leapct.innerProd(Dg,Dg)
        denom = leapct.TVquadForm(g, Dg, delta, beta)
        
        stepSize = num / (num+denom)
        
        g[:] = g[:] - stepSize*Dg[:]
    
    '''
    gainMap = g - g_0
    gainMap[gainMap>maxChange] = maxChange
    gainMap[gainMap<-maxChange] = -maxChange
    g = g_0 + gainMap
    #'''
    
    return g
    