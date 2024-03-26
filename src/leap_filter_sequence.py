################################################################################
# Copyright 2022-2024 Lawrence Livermore National Security, LLC and other 
# LEAP project developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# LivermorE AI Projector for Computed Tomography (LEAP)
# filterSequence class
################################################################################
import numpy as np
from leapctype import *

class denoisingFilter:
    """ Parent class for denoising filters
    
    All filters (i.e., regularizers) must be written as a Python class which inherets this class.
    The apply function must be defined for all filters.  This function performs denoising of the given input.
    If the filter is differentiable, you must define cost, gradient, and quadForm functions.
    Nondifferentiable filters (e.g., median filter, bilateral filter, etc.) may also be defined, but can
    only be used by ASDPOCS.  On the other hand all differentiable filters can be used in any of the LEAP
    iterative reconstruction algorithms.
    
    """
    def __init__(self, leapct):

        self.leapct = leapct
        self.weight = 1.0
        self.isDifferentiable = False
    
    def cost(self, f):
        pass
        
    def gradient(self, f):
        pass
        
    def quadForm(self, f, d):
        pass
        
    def apply(self, f):
        """If this function is not defined in the child class, perform one gradient descent iteration"""
        if self.isDifferentiable:
            d = self.gradient(f)
            num = self.leapct.sum(d**2)
            denom = self.quadForm(f, d)
            if denom <= 1.0e-16:
                return f
            stepSize = num / denom
            f -= stepSize * d
            return f
        else:
            return None

class BlurFilter(denoisingFilter):
    """This class defines a filter based on leapct.tomographicModels.BlurFilter"""
    def __init__(self, leapct, FWHM):
        super(BlurFilter, self).__init__(leapct)

        self.FWHM = FWHM
        self.isDifferentiable = False
        
    def apply(self, f):
        return self.leapct.BlurFilter(f, self.FWHM)

class BilateralFilter(denoisingFilter):
    """This class defines a filter based on leapct.tomographicModels.BilateralFilter"""
    def __init__(self, leapct, spatialFWHM, intensityFWHM, scale=1.0):
        super(BilateralFilter, self).__init__(leapct)

        self.spatialFWHM = spatialFWHM
        self.intensityFWHM = intensityFWHM
        self.scale = scale
        self.isDifferentiable = False
        
    def apply(self, f):
        return self.leapct.BilateralFilter(f, self.spatialFWHM, self.intensityFWHM, self.scale)
        
class MedianFilter(denoisingFilter):
    """This class defines a filter based on leapct.tomographicModels.MedianFilter"""
    def __init__(self, leapct, threshold=0.0, windowSize=3):
        super(MedianFilter, self).__init__(leapct)

        self.threshold = threshold
        self.windowSize = windowSize
        self.isDifferentiable = False
        
    def apply(self, f):
        return self.leapct.MedianFilter(f, self.threshold, self.windowSize)

class TV(denoisingFilter):
    """This class defines a filter based on leapct anisotropic Total Variation (TV) regularizer"""
    def __init__(self, leapct, delta=0.0, weight=1.0, f_0=None):
        super(TV, self).__init__(leapct)

        self.delta = delta
        self.weight = weight
        self.isDifferentiable = True
        self.f_0 = f_0
        
    def cost(self, f):
        if self.f_0 is not None:
            return self.leapct.TVcost(f-self.f_0, self.delta, self.weight)
        else:
            return self.leapct.TVcost(f, self.delta, self.weight)
        
    def gradient(self, f):
        if self.f_0 is not None:
            return self.leapct.TVgradient(f-self.f_0, self.delta, self.weight)
        else:
            return self.leapct.TVgradient(f, self.delta, self.weight)
        
    def quadForm(self, f, d):
        if self.f_0 is not None:
            return self.leapct.TVquadForm(f-self.f_0, d, self.delta, self.weight)
        else:
            return self.leapct.TVquadForm(f, d, self.delta, self.weight)
        
    def apply(self, f):
        if self.f_0 is not None:
            self.leapct.diffuse(f-self.f_0, self.delta, 1)
            f[:] += self.f_0[:]
            return f
        else:
            return self.leapct.diffuse(f, self.delta, 1)

class LpNorm(denoisingFilter):
    """This class defines a filter based on the L_p norm (raised to the p power) of the input"""
    def __init__(self, leapct, p=1.0, weight=1.0, f_0=None):
        super(LpNorm, self).__init__(leapct)

        self.f_0 = f_0
        self.p = p
        self.weight = weight
        self.isDifferentiable = True
        
    def cost(self, f):
        if self.f_0 is not None:
            return self.weight * self.leapct.sum(self.leapct.abs(f-self.f_0)**self.p)
        else:
            return self.weight * self.leapct.sum(self.leapct.abs(f)**self.p)
        
    def gradient(self, f):
        Df = self.leapct.copyData(f)
        if self.f_0 is not None:
            Df[:] -= self.f_0[:]
        #    return self.p * self.leapct.sign(f-self.f_0) * self.leapct.abs(f-self.f_0)**(self.p-1.0)
        #else:
        #    return self.p * self.leapct.sign(f) * self.leapct.abs(f)**(self.p-1.0)
        Df = self.weight * self.p * self.leapct.sign(Df) * self.leapct.abs(Df)**(self.p-1.0)
        return Df
        
    def quadForm(self, f, d):
        if self.f_0 is not None:
            f_copy = self.leapct.copyData(f)
            f_copy[:] = f_copy[:] - self.f_0[:]
            ind = f_copy == 0.0
            f_copy[ind] = 1.0
            f_copy = self.leapct.abs(f_copy)**(self.p-2.0)
            f_copy[ind] = 0.0
            return self.weight * self.p * self.leapct.innerProd(f_copy, d, d)
        else:
            ind = f == 0.0
            f_copy = self.leapct.copyData(f)
            f_copy[ind] = 1.0
            f_copy = self.leapct.abs(f_copy)**(self.p-2.0)
            f_copy[ind] = 0.0
            return self.weight * self.p * self.leapct.innerProd(f_copy, d, d)

    def apply(self, f):
        if self.f_0 is not None:
            f[:] = f[:] - self.f_0[:]
            f = super().apply(f)
            f[:] += self.f_0[:]
            return f
        else:
            return super().apply(f)
        
class histogramSparsity(denoisingFilter):
    """This class defines a filter that encourages sparisty in the histogram domain"""
    def __init__(self, leapct, mus=None, weight=1.0):
        super(histogramSparsity, self).__init__(leapct)

        self.weight = weight
        self.isDifferentiable = True
        
        if np.any(mus == 0.0) == False:
            mus = np.append(mus, 0.0)
        
        self.mus = np.sort(mus)
        
        if mus.size > 1:
            minDist = np.abs(self.mus[1] - self.mus[0])
            for l in range(1,self.mus.size):
                minDist = min(minDist, np.abs(self.mus[l] - self.mus[l-1]))
            self.GemanParam = 0.25*minDist*minDist
        else:
            self.GemanParam = 1.0e-5

    def Geman0(self, x):
        return x*x/(x*x + self.GemanParam)

    def Geman1(self, x):
        return 2.0*self.GemanParam*x/((x*x + self.GemanParam)*(x*x + self.GemanParam))

    def Geman1_over_x(self, x):
        return 2.0*self.GemanParam/((x*x + self.GemanParam)*(x*x + self.GemanParam))
        
    def cost(self, f):
        curTerm = self.leapct.copyData(f)
        curTerm[:] = 1.0
        for l in range(self.mus.size):
            curTerm *= self.Geman0(f-self.mus[l])
        return self.weight * self.leapct.sum(curTerm)
        
    def gradient(self, f):
        Sp = self.leapct.copyData(f)
        Sp[:] = 0.0
        for l_1 in range(self.mus.size):
            curTerm = self.leapct.copyData(f)
            curTerm[:] = 1.0
            for l_2 in range(self.mus.size):
                if l_1 == l_2:
                    curTerm *= self.Geman1(f-self.mus[l_1])
                else:
                    curTerm *= self.Geman0(f-self.mus[l_2])
            Sp += curTerm
            Sp *= self.weight
        return Sp
        
    def quadForm(self, f, d):
        minDiff = self.leapct.copyData(f)
        minDiff[:] = self.leapct.abs(minDiff[:] - self.mus[0])
        for l in range(1,self.mus.size):
            minDiff = np.minimum(minDiff, self.leapct.abs(minDiff[:] - self.mus[l]))
        return self.weight * self.leapct.innerProd(self.Geman1_over_x(minDiff), d, d)

class azimuthalFilter(denoisingFilter):
    """This class defines a filter based on leapct AzimuthalBlur filter"""
    def __init__(self, leapct, FWHM, p, weight=1.0):
        super(azimuthalFilter, self).__init__(leapct)

        self.FWHM = FWHM
        self.p = p
        self.weight = weight
        self.isDifferentiable = True

    def cost(self, f):
        Bf = self.leapct.copyData(f)
        self.leapct.AzimuthalBlur(Bf, self.FWHM)
        Bf[:] = f[:] - Bf[:]
        return self.weight * self.leapct.sum(self.leapct.abs(Bf)**p)
        
    def gradient(self, f):
        Bf = self.leapct.copyData(f)
        self.leapct.AzimuthalBlur(Bf, self.FWHM)
        Bf[:] = f[:] - Bf[:]
        Bf = self.p * self.leapct.sign(Bf) * self.leapct.abs(Bf)**(self.p-1.0)
        
        BBf = self.leapct.copyData(Bf)
        self.leapct.AzimuthalBlur(BBf, self.FWHM)
        Bf[:] = Bf[:] - BBf[:]
        Bf *= self.weight
        return Bf
        
    def quadForm(self, f, d):
        Bf = self.leapct.copyData(f)
        self.leapct.AzimuthalBlur(Bf, self.FWHM)
        Bf[:] = f[:] - Bf[:]
        Bd = self.leapct.copyData(d)
        self.leapct.AzimuthalBlur(Bd, self.FWHM)
        Bd[:] = d[:] - Bd[:]
        
        ind = Bf == 0.0
        Bf[ind] = 1.0
        Bf = self.leapct.abs(Bf)**(self.p-2.0)
        Bf[ind] = 0.0
        
        return self.weight * self.p * self.leapct.innerProd(Bf, Bd, Bd)
        
class filterSequence:
    """This class defines a weighted sum of filters (i.e., regularizers)"""
    def __init__(self, beta=1.0):
        self.filters = []
        self.beta = beta

    def append(self, newFilter):
        self.filters.append(newFilter)
        
    def clear(self):
        self.filters = []
        
    def cost(self, f):
        retVal = 0.0
        if self.beta > 0.0:
            for n in range(len(self.filters)):
                if self.filters[n].isDifferentiable == True and self.filters[n].weight > 0.0:
                    retVal += self.filters[n].cost(f)
            retVal *= self.beta
        return retVal
        
    def gradient(self, f):
        D = self.filters[0].leapct.copyData(f)
        D[:] = 0.0
        if self.beta > 0.0:
            for n in range(len(self.filters)):
                if self.filters[n].isDifferentiable == True and self.filters[n].weight > 0.0:
                    D += self.filters[n].gradient(f)
            D *= self.beta
        return D
        
    def quadForm(self, f, d):
        retVal = 0.0
        if self.beta > 0.0:
            for n in range(len(self.filters)):
                if self.filters[n].isDifferentiable == True and self.filters[n].weight > 0.0:
                    retVal += self.filters[n].quadForm(f,d)
            retVal *= self.beta
        return retVal
        
    def apply(self, f):
        for n in range(len(self.filters)):
            self.filters[n].apply(f)
        return f
