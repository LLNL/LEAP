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

def makeAttenuationRadiographs(leapct, g, air_scan=None, dark_scan=None, ROI=None):
    r"""Converts data to attenuation radiographs (flat fielding and negative log)

    .. math::
       \begin{eqnarray}
         transmission\_data &=& (raw - dark\_scan) / (air\_scan - dark\_scan) \\
         attenuation\_data &=& -log(transmission\_data)
       \end{eqnarray}

    This function assumes the input data is never attenuation data.  See argument descriptions below for how
    the input data is treated.
       
    Args:
        leapct (tomographicModels object): This is just needed to access LEAP algorithms
        g (C contiguous float32 numpy array or torch tensor): radiograph data
        air_scan (C contiguous float32 numpy array or torch tensor): air scan radiograph data; if not given, assumes that the input data is transmission data
        dark_scan (C contiguous float32 numpy array or torch tensor): dark scan radiograph data; if not given assumes the inputs have already been dark subtracted
        ROI (4-element integer array): specifies a bounding box ([first row, last row, first column, last column]) for which to estimate a mean value for flux correction
    
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
            postageStamp = torch.mean(g[:,ROI[0]:ROI[1]+1, ROI[2]:ROI[3]+1], axis=(1,2))
        else:
            postageStamp = np.mean(g[:,ROI[0]:ROI[1]+1, ROI[2]:ROI[3]+1], axis=(1,2))
        print('ROI mean: ' + str(np.mean(postageStamp)))
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
    
    When sweeping the tau parameter, the centerCol parameter is also adjusted because these both apply to the location
    and the center of rotation of the scanner.  This additional adjustment of centerCol keeps the center of rotation
    in the same place and thus has the effect of rotating the detector around the axis pointing across the detector rows.
    
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
    if param == 'tau' and leapct.get_geometry() == 'PARALLEL':
        print('Error: tau does not apply to parallel-beam data.')
        return None
        
    if has_torch == True and type(g) is torch.Tensor:
        f_stack = torch.zeros((len(values), leapct.get_numY(), leapct.get_numX()), dtype=torch.float32)
        f_stack = f_stack.to(f.get_device())
    else:
        f_stack = np.zeros((len(values), leapct.get_numY(), leapct.get_numX()), dtype=np.float32)

    g_sweep = g
    leapct_sweep = tomographicModels()
    leapct_sweep.copy_parameters(leapct)

    if param == 'tilt':
        leapct_sweep.convert_to_modularbeam()
    elif leapct_sweep.get_geometry() == 'FAN' or leapct_sweep.get_geometry() == 'PARALLEL':
        leapct_sweep.set_numRows(1)
        if has_torch == True and type(g) is torch.Tensor:
            g_sweep = torch.zeros((leapct.get_numAngles(), 1, leapct.get_numCols()), dtype=torch.float32)
            g_sweep = g_sweep.to(g.get_device())
            g_sweep[:,0,:] = g[:,iz,:]
        else:
            g_sweep = np.zeros((leapct.get_numAngles(), 1, leapct.get_numCols()), dtype=np.float32)
            g_sweep[:,0,:] = g[:,iz,:]
    
    z = leapct_sweep.z_samples()
    leapct_sweep.set_numZ(1)
    offsetZ = z[iz]
    leapct_sweep.set_offsetZ(offsetZ)
    
    f = leapct_sweep.allocate_volume()
    
    metrics = np.zeros(len(values))
    last_value = 0.0
    for n in range(len(values)):
        print(str(n) + ': ' + str(param) + ' = ' + str(values[n]))
        if param == 'centerCol':
            leapct_sweep.set_centerCol(values[n])
        elif param == 'centerRow':
            if leapct_sweep.get_geometry() == 'CONE':
                z_shift = (leapct_sweep.get_centerRow() - values[n])*leapct_sweep.get_pixelHeight()*leapct_sweep.get_sod()/leapct_sweep.get_sdd()
                leapct_sweep.set_offsetZ(leapct_sweep.get_offsetZ() + z_shift)
            leapct_sweep.set_centerRow(values[n])
        elif param == 'tau':
            delta_tau = values[n] - leapct_sweep.get_tau()
            nativeVoxelSize = leapct_sweep.get_pixelWidth() * leapct_sweep.get_sod() / leapct_sweep.get_sdd()
            new_centerCol = leapct_sweep.get_centerCol()-delta_tau/nativeVoxelSize
            print('    and centerCol = ' + str(new_centerCol))
            leapct_sweep.set_centerCol(new_centerCol)
            leapct_sweep.set_tau(values[n])
        elif param == 'sod':
            leapct_sweep.set_sod(values[n])
        elif param == 'sdd':
            leapct_sweep.set_sdd(values[n])
        if param == 'tilt':
            leapct_sweep.rotate_detector(values[n]-last_value)
        
        if algorithmName == 'inconsistencyReconstruction' or algorithmName == 'inconsistency':
            leapct_sweep.inconsistencyReconstruction(g_sweep, f)
            metrics[n] = leapct_sweep.sum(f**2)
            print('   inconsistency metric: ' + str(metrics[n]))
        else:
            leapct_sweep.FBP(g_sweep, f)
            metrics[n] = entropy(f)
            print('   entropy metric: ' + str(metrics[n]))
                
        f_stack[n,:,:] = f[0,:,:]
        last_value = values[n]
    
    return f_stack
    
def entropy(x):
    marg = np.histogramdd(np.ravel(x), bins = int(np.sqrt(x.size)))[0]/x.size
    marg = list(filter(lambda p: p > 0, np.ravel(marg)))
    return -np.sum(np.multiply(marg, np.log2(marg)))
    
def geometric_calibration(leapct, g, numIter=10):

    numSubIter = 10
    #"""
    numVars = 3
    h = np.zeros((numVars,1))
    h[0] = 0.25
    h[1] = 0.25
    h[2] = 0.05
    
    indicators = []
    for n in range(numVars):
        ind = np.zeros((numVars,1))
        ind[n] = 1.0
        indicators.append(ind)
    
    x = np.zeros((numVars,1))
    for iter in range(numIter):
        print('iteration ' + str(iter+1) + ' of ' + str(numIter))
        grad = np.zeros((numVars,1))
        H = np.zeros((numVars,numVars))
        curCost = leapct.consistency_cost(g, x)
        
        # Calculate gradient and Hessian
        for n in range(numVars):
            dx_pos = x.copy()
            dx_pos[n] += h[n]
            dx_neg = x.copy()
            dx_neg[n] -= h[n]
            
            cost_d_pos = leapct.consistency_cost(g, dx_pos)
            cost_d_neg = leapct.consistency_cost(g, dx_neg)
            
            #grad[n] = (cost_d_pos - curCost) / hs[n]
            grad[n] = (cost_d_pos - cost_d_neg) / (2.0*h[n])
            H[n,n] = (cost_d_pos - 2.0*curCost + cost_d_neg) / h[n]**2
            for m in range(numVars):
                if m != n:
                    c_pp = leapct.consistency_cost(g, x+h*indicators[n]+h*indicators[m])
                    c_pm = leapct.consistency_cost(g, x+h*indicators[n]-h*indicators[m])
                    c_mp = leapct.consistency_cost(g, x-h*indicators[n]+h*indicators[m])
                    c_mm = leapct.consistency_cost(g, x-h*indicators[n]-h*indicators[m])
                    H[m,n] = (c_pp + c_mm - c_pm - c_mp) / (4.0*h[m]*h[n])

        # Perform Newton's method update
        detH = np.linalg.det(H)
        if detH == 0.0:
            print('Hessian is no positive definite ('+str(detH)+')')
            break
        elif detH < 0.0:
            maxVal = np.max(grad)
            d = np.min(h)*grad / maxVal
        else:
            d = 0.9*np.linalg.solve(H,grad)
        x -= d
        
        newCost = leapct.consistency_cost(g, x)
        for n in range(numSubIter):
            if newCost < curCost:
                break
            else:
                x += d
                d *= 0.1
                x -= d
                newCost = leapct.consistency_cost(g, x)
        if newCost > curCost:
            x += d
            break
    #if np.abs(x[2]) < h[2]:
    #    x[2] = 0.0
    #"""
    """
    Q = np.ones((4,1))
    Q[2] = 0.1
    nativeVoxelSize = leapct.get_pixelWidth() * leapct.get_sod() / leapct.get_sdd()
    numVars = 4
    h = np.zeros((numVars,1))
    h[0] = 0.25
    h[1] = 0.25
    h[2] = 0.25 #/ nativeVoxelSize
    h[3] = 0.05
    
    indicators = []
    for n in range(numVars):
        ind = np.zeros((numVars,1))
        ind[n] = 1.0
        indicators.append(ind)
    
    x = np.zeros((numVars,1))
    for iter in range(numIter):
        grad = np.zeros((numVars,1))
        H = np.zeros((numVars,numVars))
        curCost = leapct.consistency_cost(g, x)
        
        # Calculate gradient and Hessian
        for n in range(numVars):
            dx_pos = x.copy()
            dx_pos[n] += h[n]
            dx_neg = x.copy()
            dx_neg[n] -= h[n]
            
            cost_d_pos = leapct.consistency_cost(g, dx_pos)
            cost_d_neg = leapct.consistency_cost(g, dx_neg)
            
            #grad[n] = (cost_d_pos - curCost) / hs[n]
            grad[n] = (cost_d_pos - cost_d_neg) / (2.0*h[n])
            H[n,n] = (cost_d_pos - 2.0*curCost + cost_d_neg) / h[n]**2
            for m in range(numVars):
                if m != n:
                    c_pp = leapct.consistency_cost(g, x+h*indicators[n]+h*indicators[m])
                    c_pm = leapct.consistency_cost(g, x+h*indicators[n]-h*indicators[m])
                    c_mp = leapct.consistency_cost(g, x-h*indicators[n]+h*indicators[m])
                    c_mm = leapct.consistency_cost(g, x-h*indicators[n]-h*indicators[m])
                    H[m,n] = (c_pp + c_mm - c_pm - c_mp) / (4.0*h[m]*h[n])

        # Perform Newton's method update
        detH = np.linalg.det(H)
        if detH == 0.0:
            print('Hessian is no positive definite ('+str(detH)+')')
            break
        elif detH < 0.0:
            maxVal = np.max(grad)
            d = np.min(h)*grad / maxVal
        else:
            d = 0.9*np.linalg.solve(H,grad)
        d *= Q
        stepSize = 1.0
        x -= stepSize*d
        
        newCost = leapct.consistency_cost(g, x)
        for n in range(numSubIter):
            if newCost < curCost:
                break
            else:
                x += stepSize*d
                stepSize *= 0.1
                x -= stepSize*d
                newCost = leapct.consistency_cost(g, x)
        if newCost > curCost:
            x += stepSize*d
            break
    #"""
    """    
    nativeVoxelSize = leapct.get_pixelWidth() * leapct.get_sod() / leapct.get_sdd()
    h = np.zeros((4,1))
    h[0] = 1.0
    h[1] = 1.0
    h[2] = 1.0 / nativeVoxelSize
    h[3] = 0.1
    
    indicators = []
    for n in range(4):
        ind = np.zeros((4,1))
        ind[n] = 1.0
        indicators.append(ind)
    
    x = np.zeros((4,1))
    for iter in range(numIter):
        grad = np.zeros((4,1))
        H = np.zeros((4,4))
        curCost = leapct.consistency_cost(g, x)
        
        # Calculate gradient and Hessian
        for n in range(4):
            dx_pos = x.copy()
            dx_pos[n] += h[n]
            dx_neg = x.copy()
            dx_neg[n] -= h[n]
            
            cost_d_pos = leapct.consistency_cost(g, dx_pos)
            cost_d_neg = leapct.consistency_cost(g, dx_neg)
            
            #grad[n] = (cost_d_pos - curCost) / hs[n]
            grad[n] = (cost_d_pos - cost_d_neg) / (2.0*h[n])
            H[n,n] = (cost_d_pos - 2.0*curCost + cost_d_neg) / h[n]**2
            for m in range(4):
                if m != n:
                    c_pp = leapct.consistency_cost(g, x+h*indicators[n]+h*indicators[m])
                    c_pm = leapct.consistency_cost(g, x+h*indicators[n]-h*indicators[m])
                    c_mp = leapct.consistency_cost(g, x-h*indicators[n]+h*indicators[m])
                    c_mm = leapct.consistency_cost(g, x-h*indicators[n]-h*indicators[m])
                    H[m,n] = (c_pp + c_mm - c_pm - c_mp) / (4.0*h[m]*h[n])

        # Perform Newton's method update
        detH = np.linalg.det(H)
        if detH == 0.0:
            print('Hessian is no positive definite ('+str(detH)+')')
            break
        elif detH < 0.0:
            maxVal = np.max(grad)
            d = np.min(h)*grad / maxVal
            d[2] = 0.0
            x -= d
        else:
            d = np.linalg.solve(H,grad)
            d[2] = 0.0
            x = x - d
    """
    
    return x
    