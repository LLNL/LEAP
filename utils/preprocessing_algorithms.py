################################################################################
# Copyright 2022-2024 Lawrence Livermore National Security, LLC and other 
# LEAP project developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# LivermorE AI Projector for Computed Tomography (LEAP)
#
# This file contains preprocessing algorithms for CT projection data
# If one does not have enough CPU memory to process the entire data set, then we
# recommend splitting the data across projections for the outlierCorrection
# algorithms and splitting the data across detector rows for the ringRemoval
# algorithms.
# For beam hardening correction, see the demo_leapctype/test_beam_hardening.py
################################################################################

import sys
import os
import time
import numpy as np
from leapctype import *

def makeAttenuationRadiographs(g, air_scan=None, dark_scan=None, ROI=None):
    r"""Converts data to attenuation radiographs (flat fielding and negative log)

    
    .. math::
       \begin{eqnarray}
         transmission\_data &=& (raw - dark\_scan) / (air\_scan - dark\_scan) \\
         attenuation\_data &=& -log(transmission\_data)
       \end{eqnarray}
       
    Args:
        g (C contiguous float32 numpy array or torch tensor): radiograph data
        air_scan (C contiguous float32 numpy array or torch tensor): air scan radiograph data; if not given, assumes that the input data is transmission data
        dark_scan (C contiguous float32 numpy array or torch tensor): dark scan radiograph data; if not given assumes the inputs have already been dark subtracted
        ROI (4-element integer array): specifies a bounding box for which to estimate a mean value for flux correction
    
    """
    if dark_scan is not None:
        if air_scan is not None:
            g = (g - dark_scan) / (air_scan - dark_scan)
        else:
            g = g - dark_scan
    else:
        if air_scan is not None:
            g = g / air_scan
    if ROI is not None:
        if has_torch == True and type(g) is torch.Tensor:
            postageStamp = torch.mean(g[:,ROI[0]:ROI[1], ROI[2]:ROI[3]], axis=(1,2))
        else:
            postageStamp = np.mean(g[:,ROI[0]:ROI[1], ROI[2]:ROI[3]], axis=(1,2))
        g = g / postageStamp[:,None,None]
    g = leapct.negLog(g)
    return g

def outlierCorrection(leapct, g, threshold=0.03, windowSize=3, isAttenuationData=True):
    r"""Removes outliers (zingers) from CT projections
    
    Assumes the input data is in attenuation space.
    No LEAP parameters need to be set for this function to work 
    and can be applied to any CT geometry type.
    This algorithm processes each projection independently
    and removes outliers by a thresholded median filter
    
    Args:
        leapct (tomographicModels object): This is just needed to access LEAP algorithms
        g (contiguous float32 numpy array or torch tensor): attenuation or transmission projection data
        threshold (float): A pixel will be replaced by the median of its neighbors if \|g - median(g)\|/median(g) > threshold
        windowSize (int): The window size of the median filter applied is windowSize x windowSize
        isAttenuationData (bool): True if g is attenuation data, False otherwise
        
    Returns:
        The corrected data
    """
    # This algorithm processes each transmission
    if isAttenuationData:
        g = leapct.expNeg(g)
    leapct.MedianFilter2D(g, threshold, windowSize)
    if isAttenuationData:
        g = leapct.negLog(g)
    return g

def outlierCorrection_highEnergy(leapct, g, isAttenuationData=True):
    """Removes outliers (zingers) from CT projections
    
    Assumes the input data is in attenuation space.
    No LEAP parameters need to be set for this function to work 
    and can be applied to any CT geometry type.
    This algorithm processes each projection independently
    and removes outliers by a series of three thresholded median filters
    
    This outlier correction algorithm should most be used for MV CT or neutron CT
    where outliers effect a larger neighborhood of pixels.  This function uses a three-stage
    thresholded median filter to remove outliers.
    
    Args:
        leapct (tomographicModels object): This is just needed to access LEAP algorithms
        g (contiguous float32 numpy array or torch tensor): attenuation or transmission projection data
        isAttenuationData (bool): True if g is attenuation data, False otherwise
        
    Returns:
        The corrected data
    """
    
    if isAttenuationData:
        g = leapct.expNeg(g)
    leapct.MedianFilter2D(g, 0.08, 7)
    leapct.MedianFilter2D(g, 0.024, 5)
    leapct.MedianFilter2D(g, 0.0032, 3)
    if isAttenuationData:
        g = leapct.negLog(g)
    return g
    

def detectorDeblur_FourierDeconv(leapct, g, H, WienerParam=0.0, isAttenuationData=True):
    """Removes detector blur by fourier deconvolution
    
    Args:
        g (contiguous float32 numpy array or torch tensor): attenuation or transmission projection data
        H (2D contiguous float32 numpy array or torch tensor): Magnitude of the frequency response of blurring psf, DC is at [0,0]
        WienerParam (float): Parameter for Wiener deconvolution, number should be between 0.0 and 1.0
        isAttenuationData (bool): True if g is attenuation data, False otherwise
    
    """
    if has_torch == True and type(H) is torch.Tensor:
        H = H.cpu().detach().numpy()
    if np.min(np.abs(H)) < 1.0/100.0:
        WienerParam = max(2.5e-5, WienerParam)
    if 0 < WienerParam and WienerParam <= 1.0:
        H = (1.0+WienerParam)*H/(H*H+WienerParam)
    else:
        H = 1.0 / H
    H = H / H[0,0]
    return leapct.transmission_filter(g, H, isAttenuationData)
    
def detectorDeblur_RichardsonLucy(leapct, g, H, numIter=10, isAttenuationData=True):
    r"""Removes detector blur by Richardson-Lucy iterative deconvolution
    
    Richardson-Lucy iterative deconvolution is developed for Poisson-distributed data
    and inherently preserve the non-negativity of the input.  It uses the following update step,
    where t = transmission data
    
    .. math::
       \begin{eqnarray}
         t_{n+1} &=& t_n H^T\left[ \frac{t_0}{Ht_n} \right]
       \end{eqnarray}
    
    Args:
        g (contiguous float32 numpy array or torch tensor): attenuation or transmission projection data
        H (2D contiguous float32 numpy array or torch tensor): Magnitude of the frequency response of blurring psf, DC is at [0,0]
        numIter (int): Number of iterations
        isAttenuationData (bool): True if g is attenuation data, False otherwise
    
    """
    H = H / H[0,0]
    if isAttenuationData:
        t = leapct.expNeg(g)
    else:
        t = g
    t_0 = leapct.copyData(t)
    Ht = leapct.copyData(t)
    for n in range(numIter):
        Ht[:] = t[:]
        Ht = leapct.transmission_filter(Ht, H, False)
        Ht[:] = t_0[:] / Ht[:]
        Ht = leapct.transmission_filter(Ht, H, False)
        t[:] = t[:] * Ht[:]
    if isAttenuationData:
        g[:] = leapct.negLog(t)
    else:
        g[:] = t[:]
    return g

def ringRemoval_fast(leapct, g, delta, numIter, maxChange):
    r"""Removes detector pixel-to-pixel gain variations that cause ring artifacts in reconstructed images
    
    This algorithm estimates the rings by first averaging all projections.  Then denoises this
    signal by minimizing the TV norm.  Finally the gain correction map to correct the data
    is estimated by the difference of the TV-smoothed and averaged projection data (these are all 2D signals).
    This is summarized by the math equations below.
    
    .. math::
       \begin{eqnarray}
         \overline{g} &:=& \frac{1}{numAngles}\sum_{angles} g \\
         gain\_correction &:=&argmin \; TV(\overline{g}) - \overline{g} \\
         g\_corrected &:=& g + gain\_correction
       \end{eqnarray}
    
    Assumes the input data is in attenuation space.
    No LEAP parameters need to be set for this function to work 
    and can be applied to any CT geometry type.
    This algorithm is effective at removing ring artifacts and runs fast, but can
    sometimes create new rings of tangents of sharp transitions.
    
    Args:
        leapct (tomographicModels object): This is just needed to access LEAP algorithms
        g (contiguous float32 numpy array or torch tensor): attenuation projection data
        delta (float): The delta parameter of the Total Variation Functional
        numIter (int): Number of iterations
        maxChange (float): An upper limit on the maximum difference that can be applied to a detector pixels
    
    Returns:
        The corrected data
    """
    if has_torch == True and type(g) is torch.Tensor:
        g_sum = torch.zeros((1,g.shape[1],g.shape[2]), dtype=torch.float32)
        g_sum = g_sum.to(g.get_device())
        g_sum[0,:] = torch.mean(g,axis=0)
    else:
        g_sum = np.zeros((1,g.shape[1],g.shape[2]), dtype=np.float32)
        g_sum[0,:] = np.mean(g,axis=0)
    g_sum_save = leapct.copyData(g_sum)
    
    numNeighbors = leapct.get_numTVneighbors()
    leapct.set_numTVneighbors(6)
    leapct.diffuse(g_sum,delta,numIter)
    leapct.set_numTVneighbors(numNeighbors)
    
    gainMap = g_sum - g_sum_save
    
    gainMap[gainMap>maxChange] = maxChange
    gainMap[gainMap<-maxChange] = -maxChange
    g = g + gainMap[None,:,:]
    return g

def ringRemoval_median(leapct, g, threshold=0.0, windowSize=5, numIter=1):
    r"""Removes detector pixel-to-pixel gain variations that cause ring artifacts in reconstructed images
    
    .. math::
       \begin{eqnarray}
         gain\_correction &:=& \frac{1}{numAngles}\sum_{angles} [MedianFilter2D(g) - g] \\
         g\_corrected &:=& g + gain\_correction
       \end{eqnarray}
    
    Assumes the input data is in attenuation space.
    No LEAP parameters need to be set for this function to work 
    and can be applied to any CT geometry type.
    This algorithm is effective at removing ring artifacts without creating new ring artifacts.
    
    Args:
        leapct (tomographicModels object): This is just needed to access LEAP algorithms
        g (contiguous float32 numpy array or torch tensor): attenuation projection data
        threshold (float): The threshold for the thresholded median filter, where a pixel will be replaced by the median of its neighbors if \|g - median(g)\|/median(g) > threshold
        windowSize (int): The window size of the median filter applied is windowSize x windowSize
        numIter (int): Number of iterations
    
    Returns:
        The corrected data
    """
    
    Dg = leapct.copyData(g)
    for n in range(numIter):
        if n > 0:
            Dg[:] = g[:]
        leapct.MedianFilter2D(Dg, threshold, windowSize)
        Dg[:] = g[:] - Dg[:] # noise only
        if has_torch == True and type(g) is torch.Tensor:
            Dg_sum = torch.mean(Dg,axis=0)
        else:
            Dg_sum = np.mean(Dg,axis=0)

        g -= Dg_sum[None,:,:]
    return g

def ringRemoval(leapct, g, delta, beta, numIter):
    r"""Removes detector pixel-to-pixel gain variations that cause ring artifacts in reconstructed images
    
    This algorithm estimates the gain correction necessary to remove ring artifacts by solving denoising
    the data by minimizing the TV norm with the gradient step determined by averaging the gradients over
    all angles.
    
    Assumes the input data is in attenuation space.
    No LEAP parameters need to be set for this function to work 
    and can be applied to any CT geometry type.
    This algorithm is effective at removing ring artifacts without creating new ring artifacts, 
    but is computationally expensive.
    
    Args:
        leapct (tomographicModels object): This is just needed to access LEAP algorithms
        g (contiguous float32 numpy array or torch tensor): attenuation projection data
        delta (float): The delta parameter of the Total Variation Functional
        beta (float): The strength of the regularization
        numIter (int): Number of iterations
    
    Returns:
        The corrected data
    """
    numNeighbors = leapct.get_numTVneighbors()
    leapct.set_numTVneighbors(6)
    g_0 = leapct.copyData(g)
    for n in range(numIter):
        Dg = leapct.TVgradient(g, delta, beta)
        if has_torch == True and type(Dg) is torch.Tensor:
            Dg_sum = torch.mean(Dg,axis=0)
        else:
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
    leapct.set_numTVneighbors(numNeighbors)
    
    return g
    
def parameter_sweep(leapct, g, values, param='centerCol', iz=None, algorithmName='FBP'):
    r"""Performs single-slice reconstructions of several values of a given parameter
    
    The CT geometry parameters and the CT volume parameters must be set prior to running this function.
    The parameters to sweep are all standard LEAP CT geometry parameter names, except 'tilt'
    The 'tilt' parameter is swept by converting the geometry to modular-beam (see the convert_to_modularbeam function).
    Then the tilt is achieved by rotating the colVecs and rowVecs parameters (note that the data g is not rotated, 
    just the model of the detector orientation which is better because no interpolation is necessary).  This rotation
    is achieved with the rotate_detector function.
    
    Args:
        leapct (tomographicModels object): This is just needed to access LEAP algorithms
        g (contiguous float32 numpy array or torch tensor): attenuation projection data
        values (list of floats): the values to reconstruct with
        param (string): the name of the parameter to sweep; can be 'centerCol', 'centerRow', 'tau', 'sod', 'sdd', 'tilt'
        iz (integer): the z-slice index to perform the reconstruction; if not given, uses the central slice
        algorithmName (string): the name of the algorithm to use for reconstruction; can be 'FBP' or 'inconsistencyReconstruction'
        
    Returns:
        stack of single-slice reconstructions (i.e., 3D numpy array or torch tensor) for all parameter values
    """
    valid_params = ['centerCol', 'centerRow', 'tau', 'sod', 'sdd', 'tilt']
    if param == None:
        param = 'centerCol'
    if any(name in param for name in valid_params) == False:
        print('Error: Invalid parameter, must be one of: ' + str(valid_params))
        return None
    if iz is None:
        iz = int(leapct.get_numZ()//2)
    if iz < 0 or iz >= leapct.get_numZ():
        print('Error: Slice index is out of bounds for current volume specification.')
        return None
    if param == 'tilt' and leapct.get_geometry() == 'FAN':
        print('Error: Detector tilt can cannot be applied to fan-beam data.')
        return None
        
    centerCol_save = leapct.get_centerCol()
    centerRow_save = leapct.get_centerRow()
    tau_save = leapct.get_tau()
    sod_save = leapct.get_sod()
    sdd_save = leapct.get_sdd()
    offsetZ_save = leapct.get_offsetZ()
    numZ_save = leapct.get_numZ()
    
    z = leapct.z_samples()
    leapct.set_numZ(1)
    offsetZ = z[iz]
    leapct.set_offsetZ(offsetZ)
    
    f = leapct.allocate_volume()
    if has_torch == True and type(f) is torch.Tensor:
        f_stack = torch.zeros((len(values), leapct.get_numY(), leapct.get_numX()), dtype=torch.float32)
        f_stack = f_stack.to(f.get_device())
    else:
        f_stack = np.zeros((len(values), leapct.get_numY(), leapct.get_numX()), dtype=np.float32)

    if param == 'tilt':
        leapct_tilt = tomographicModels()
        leapct_tilt.copy_parameters(leapct)
        leapct_tilt.convert_to_modularbeam()
    else:
        leapct_tilt = None
    
    metrics = np.zeros(len(values))
    last_value = 0.0
    for n in range(len(values)):
        print(str(n) + ': ' + str(param) + ' = ' + str(values[n]))
        if param == 'centerCol':
            leapct.set_centerCol(values[n])
        elif param == 'centerRow':
            leapct.set_centerRow(values[n])
        elif param == 'tau':
            leapct.set_tau(values[n])
        elif param == 'sod':
            leapct.set_sod(values[n])
        elif param == 'sdd':
            leapct.set_sdd(values[n])
        if param == 'tilt':
            leapct_tilt.rotate_detector(values[n]-last_value)
            if algorithmName == 'inconsistencyReconstruction' or algorithmName == 'inconsistency':
                leapct_tilt.inconsistencyReconstruction(g,f)
                metrics[n] = leapct.sum(f**2)
                print('   inconsistency metric: ' + str(metrics[n]))
            else:
                leapct_tilt.FBP(g,f)
                metrics[n] = entropy(f)
                print('   entropy metric: ' + str(metrics[n]))
        else:
            if algorithmName == 'inconsistencyReconstruction' or algorithmName == 'inconsistency':
                leapct.inconsistencyReconstruction(g,f)
                metrics[n] = leapct.sum(f**2)
                print('   inconsistency metric: ' + str(metrics[n]))
            else:
                leapct.FBP(g,f)
                metrics[n] = entropy(f)
                print('   entropy metric: ' + str(metrics[n]))
                
        f_stack[n,:,:] = f[0,:,:]
        last_value = values[n]
    
    leapct.set_centerCol(centerCol_save)
    leapct.set_centerRow(centerRow_save)
    leapct.set_tau(tau_save)
    leapct.set_sod(sod_save)
    leapct.set_sdd(sdd_save)
    leapct.set_numZ(numZ_save)
    leapct.set_offsetZ(offsetZ_save)
    return f_stack
    
def entropy(x):
    marg = np.histogramdd(np.ravel(x), bins = int(np.sqrt(x.size)))[0]/x.size
    marg = list(filter(lambda p: p > 0, np.ravel(marg)))
    return -np.sum(np.multiply(marg, np.log2(marg)))
    
