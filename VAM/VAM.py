'''*****************************************************************************************
   *  VAM.py - class for Volumetric Additive Manufacturing (VAM)
   *****************************************************************************************
   *  Copyright (C) 2023 Lawrence Livermore National Security.
   *  Produced at Lawrence Livermore National Laboratory (cf, DISCLAIMER).
   *  Written by Kyle Champley
   *  Algorithm based on: Numerical Optimization of the Light Intensity Fields used in
   *  Volumetric Additive Manufacturing by Kyle Champley, Erika Fong, and Maxim Shusteff
   *  which is referred to as the "VAM paper" in the comments below.
   *  This code is mostly a translation of the LTT code in the class: computedAxialLithography
   *  There are a few deviations from the VAM paper that readers should be aware of.
   *  First of all, there are a several scalars that set the proper units for the Light Intensity
   *  Field (LIF).  The code below omits the ratio alpha/Omega in the calculations until the
   *  very end when the LIF and delivered dose map are returned.  Secondly, the variable "c"
   *  is replaced by "T" in the code below.  In summary the VAM paper attempts to solve
   *  the cost function Phi(g) = 0.5|| u(alpha/Omega R_a*g - D_c) - f_T || and the code below
   *  solves the cost function Phi(g) = 0.5|| u(R_a*g - T) - f_T ||.  This does
   *  not effect the final result (as I said above) because the scaling coefficients are
   *  applied at the end of the algorithm.  These changes were made to simplify the code.
   *****************************************************************************************'''
import time
import numpy as np

# This implementation requires LivermorE AI Projector for Computed Tomography (LEAP)
# for the calculation of forward and backprojectors, and 2D ramp filter
# The LEAP routines are used because they are quantitatively accurate and fast,
# but this code could be easily modified to remove this dependency on LEAP.

from leapctype import *
leapct = tomographicModels()

class VAM:
    def __init__(self, alpha=0.0, D_c=1.0, Omega=0.0, resinDiameter=0.0, useDegPerSec=False):
        # This function sets default values for class member variables
    
        # Sigmoid parameter types
        self.LOGISTIC, self.POLYNOMIAL, self.ASYMMETRIC, self.IDEAL = [0, 1, 2, 3]

        # Initial guess types
        self.PROJECTION_METHOD, self.HYBRID, self.RAMP_FILTERED = [1, 2, 3]
        
        # These are fixed parameters, don't change them
        self.T = 0.5
        self.CGrestart = 50
        self.maxDoseFactor = 2.0
        
        # Initial guess parameters
        self.boostFactor = 0.0
        #self.seedType = self.HYBRID
        self.seedType = self.PROJECTION_METHOD
		
        # Numerical optimization parameters
        # These define the shape of the sigmoid function
        # Setting 1 <= minDoseFactor < 2, will use the ASYMMETRIC
        # sigmoid (Equation (7) on page 5 of the VAM paper)
        # Note that minDoseFactor is referred to as "m" in the VAM paper
        self.minDoseFactor = 1.0
        self.sigmoidType = self.POLYNOMIAL
        self.setSigmoidParameter(20.0/self.T)
        
        # Number of iterations to run
        self.N_iter = 20
        
        # Set this to a positive value if you want the maximum
        # LIF value constrained.  Warning: setting this value
        # to a value that is too small will prevent the algorithm
        # from converging and may give a bad result
        self.clipHigh = 0.0

        # Physics parameters
        # These parameters enable quantitatively accuracy of the LIF
        self.alpha = alpha # 0.1 cm^-1 = 0.01 mm^-1 is a typical value (0.2 is likely the largest value)
        self.D_c = D_c # criticalDose, J / mm^3 (typical value 50-100 miliJoules / cm^3)
        self.Omega = Omega # rotation rate, 1/sec
        self.resinDiameter = resinDiameter # mm
        self.useDegPerSec = useDegPerSec

        # Default projector and target volume parameters
        # These are just for demonstration, but the user
        # MUST set all of theses parameters themselves
        self.numberOfAngles = 360.0
        self.angularRange = 360.0 # degrees
        self.numberOfHorizontalRaySamples = 512 # should be == targetGeometry.shape[1] == targetGeometry.shape[2]
        self.numberOfVerticalRaySamples = 1 # should be == targetGeometry.shape[0]
        self.pixelSize = 1.0 # mm
        self.voxelSize = 1.0 # mm, should be equal to self.pixelSize
		
        # Rarely used parameters
        self.W = None
        self.beta = 0.0
        
    def execute(self, targetGeometry, numIter=None, pixelSize=1.0, numAngles=None, g=None, mu=None):
        # This is the main execution function.
        # Call this to provide an estimate of the light intensity field (LIF)
        # for a given target geometry
        # One may provide an initial estimate, g, of the LIF
        # This function returns the optimized LIF, g,
        # and the delivered dose map, Pstar_g
        
        if numIter is not None and numIter > 0:
            self.N_iter = numIter
        if numAngles is not None and numAngles > 0:
            self.numberOfAngles = numAngles
        if pixelSize > 0.0:
            self.pixelSize = pixelSize
        self.voxelSize = self.pixelSize
        
        # Set LEAP parameters (LEAP is only used in the P, Pstar, and ramp functions)        
        self.setLEAPparams(targetGeometry)
        
        if 1.0 <= self.minDoseFactor:
            self.minDoseFactor = max(1.0+1.0e-8, self.minDoseFactor)
            
        if 1.0 <= self.minDoseFactor and self.minDoseFactor < 2.0 and self.sigmoidType == self.POLYNOMIAL:
            self.minDoseFactor = max(self.minDoseFactor, 1.0/self.sigmoidExponent + 1.0)
            self.maxDoseFactor = max(self.minDoseFactor, self.maxDoseFactor)
            print("using ASYMMETRIC kernel (%f)!" % self.minDoseFactor)
            #minDoseFactor = max(1.1,minDoseFactor)
            self.setSigmoidParameter(self.sigmoidParameter, self.minDoseFactor)
            self.sigmoidType = self.ASYMMETRIC
        
        Omega_save = self.Omega
        if self.useDegPerSec == True and self.Omega > 0.0:
            # deg/sec ==> 1/sec
            self.Omega = self.Omega / self.angularStepSize

        ''' This code uses the LTT-VAM directly (just for debugging)
        LTT.setAllReconSlicesZ(targetGeometry)
        LTT.cmd('VAM {calculateDoseMap=true; seed=hybrid; minDoseFactor=1; N_iter=10}')
        g = LTT.getAllProjections()
        Pstar_g = LTT.getAllReconSlicesZ()
        return g, Pstar_g
        #'''

        targetGeometry_max = np.max(targetGeometry)
        targetGeometry = targetGeometry / targetGeometry_max
        
        if mu is not None:
            leapct.set_attenuationMap(mu)
            leapct.windowFOV(mu)
        
        if g is None:
            g, Pstar_g = self.setInitialGuess(targetGeometry)
        else:
            Pstar_g = self.Pstar(g)
            g, Pstar_g = self.scaleInitialGuess(g, targetGeometry, Pstar_g)
        g, Pstar_g = self.CGsolver(g, Pstar_g, targetGeometry)
        
        self.Omega = Omega_save
        targetGeometry *= targetGeometry_max
        
        g *= self.calculateScalingCoefficients()
        if self.D_c > 0.0:
            Pstar_g *= 2.0*self.D_c
        
        return g, Pstar_g

    def u(self, x, T=0.0):
        # Evaluates u(f), where u is a sigmoid function
        # See u_1(t) (LOGISTIC), u_2(t) (POLYNOMIAL), and u_3(t) (ASYMMETRIC) in the VAM paper
        x = x - T
        y = x.copy()
        
        if self.sigmoidType == self.LOGISTIC:
            y = 1.0 / (1.0 + np.exp(-self.sigmoidParameter*x))
        elif self.sigmoidType == self.POLYNOMIAL:
            y[x<=-0.5] = 0.0
            ind = np.logical_and(x<=0.0, x>-0.5)
            y[ind] = self.polyCoeff * (y[ind]+0.5)**self.sigmoidExponent
            ind = np.logical_and(x<0.5, x>0.0)
            y[ind] = 1.0 - self.polyCoeff*(0.5-y[ind])**self.sigmoidExponent
            y[x>=0.5] = 1.0
        elif self.sigmoidType == self.ASYMMETRIC:
            y[x<=-0.5] = 0.0
            ind = np.logical_and(x<=0.0, x>-0.5)
            y[ind] = self.polyCoeff*(y[ind]+0.5)**self.sigmoidExponent
            ind = np.logical_and(x<0.5*(self.minDoseFactor-1.0), x>0.0)
            y[ind] = 1.0 - self.polyCoeff_high*(0.5*(self.minDoseFactor-1.0)-y[ind])**self.sigmoidExponent_high
            y[x>=0.5*(self.minDoseFactor-1.0)] = 1.0
        elif self.sigmoidType == self.IDEAL:
            y += 0.5
        return y
               
    def up(self, x, T=0.0):
        # Evaluates the first derivative of u, i.e., u'(f), where u is a sigmoid function
        x = x - T
        y = x.copy()
        if self.sigmoidType == self.LOGISTIC:
            expTerm = y.copy()
            expTerm = np.exp(-self.sigmoidParameter*expTerm)
            y = self.sigmoidParameter*expTerm / (1.0 + expTerm)**2
        elif self.sigmoidType == self.POLYNOMIAL:
            y[x<=-0.5] = 0.0
            ind = np.logical_and(x<=0.0, x > -0.5)
            y[ind] = self.sigmoidExponent * self.polyCoeff * (y[ind]+0.5)**(self.sigmoidExponent-1)
            ind = np.logical_and(x < 0.5, x > 0.0)
            y[ind] = self.sigmoidExponent * self.polyCoeff * (0.5-y[ind])**(self.sigmoidExponent-1.0)
            y[x>=0.5] = 0.0
        elif self.sigmoidType == self.ASYMMETRIC:
            y[x<=-0.5] = 0.0
            ind = np.logical_and(x<=0.0, x > -0.5)
            y[ind] = self.sigmoidExponent * self.polyCoeff * (y[ind]+0.5)**(self.sigmoidExponent-1.0)
            ind = np.logical_and(x < 0.5*(self.minDoseFactor-1.0), x > 0.0)
            y[ind] = self.sigmoidExponent_high * self.polyCoeff_high * (0.5*(self.minDoseFactor-1.0)-y[ind])**(self.sigmoidExponent_high-1.0)
            y[x >= 0.5*(self.minDoseFactor-1.0)] = 0.0
        elif self.sigmoidType == self.IDEAL:
            y = 1.0
        else:
            y = 1.0
        return y
        
    def upp(self, x, T=0.0):
        # Evaluates the second derivative of u, i.e., u''(f), where u is a sigmoid function
        x = x - T
        y = x.copy()
        if self.sigmoidType == self.LOGISTIC:
            expTerm = y.copy()
            expTerm = np.exp(-self.sigmoidParameter*expTerm)
            y = 2.0*(self.sigmoidParameter*expTerm)**2 / (1.0+expTerm)**3 - self.sigmoidParameter**2*expTerm / (1.0+expTerm)**2
        elif self.sigmoidType == self.POLYNOMIAL:
            y[x<= -0.5] = 0.0
            ind = np.logical_and(x<=0.0, x>-0.5)
            y[ind] = (self.sigmoidExponent-1.0) * self.sigmoidExponent * self.polyCoeff * (y[ind]+0.5)**(self.sigmoidExponent-2.0)
            ind = np.logical_and(x<0.5, x>0.0)
            y[ind] = -(self.sigmoidExponent-1.0) * self.sigmoidExponent * self.polyCoeff * (0.5-y[ind])**(self.sigmoidExponent-2.0)
            y[x>=0.5] = 0.0
        elif self.sigmoidType == self.ASYMMETRIC:
            y[x<=-0.5] = 0.0
            ind = np.logical_and(x<=0.0, x>-0.5)
            y[ind] = (self.sigmoidExponent-1.0) * self.sigmoidExponent * self.polyCoeff * (y[ind]+0.5)**(self.sigmoidExponent-2.0)
            
            if np.abs(self.sigmoidExponent_high-1.0) > 1.0e-8:
                ind = np.logical_and(x < 0.5*(self.minDoseFactor-1.0), x > 0.0)
                y[ind] = -(self.sigmoidExponent_high-1.0) * self.sigmoidExponent_high * self.polyCoeff_high * (0.5*(self.minDoseFactor-1.0)-y[ind])**(self.sigmoidExponent_high-2.0)
                y[x >= 0.5*(self.minDoseFactor-1.0)] = 0.0
            else:
                y[x>0.0] = 0.0
            
        elif self.sigmoidType == self.IDEAL:
            y = 0.0
        else:
            y = 0.0
        return y

    def setSigmoidParameter(self, x, y = 0.0):
        # This function sets the sigmoid parameters as a function of the user inputs
        self.sigmoidParameter = x
        self.sigmoidExponent = max(2.0, self.sigmoidParameter / 4.0)
        self.polyCoeff = 2**(self.sigmoidExponent-1)
    
        self.sigmoidExponent_high = self.sigmoidExponent;
        self.polyCoeff_high = self.polyCoeff;
        if 1.0 < y and y < 2.0:
            # standard should be y = 2, so (2-1)/2 = 0.5
            self.sigmoidExponent_high = 2.0*((y-1.0)/2.0)*self.sigmoidExponent
            self.polyCoeff_high = 0.5*((y-1.0)/2.0)**(-self.sigmoidExponent_high)

    def calculateCostFunction(self, Pstar_g, f, g=None):
        # This function evaluates the cost function,
        # which is used to either quantify the accuracy of the estimate
        # or helps find the optimal step size
        
        costFunctionScalar = 100.0 * 2.0 / np.sum(f**2)
        return 0.5*costFunctionScalar*np.sum((self.u(Pstar_g - self.T)-f)**2)
            
    def calculateStepSize(self, g, Pstar_g, d, f, maxStep, curCost = None):
        # This function calculates the step size of the gradient or conjugate gradient step
        regTerm = 0.0
        if curCost is None:
            curCost = self.calculateCostFunction(Pstar_g, f)
        curStep = maxStep

        # An upper bound of the step size was calculated by the Hessian of the cost function
        # But since we use a non-quadratic cost function this may not be optimal
        # Thus we use the backtracking method to find the optimal step size
        for n in range(16):
            # Evaluate the result if the step size is given by curStep
            # newProjs = clip(g + curStep * d)
            newProjs = g.copy()
            newProjs += curStep * d
            self.clipLowAndHigh(newProjs)
            
            Pstar_update = self.Pstar(newProjs)
            newCost = self.calculateCostFunction(Pstar_update, f, g)
            print("cost with step " + str(curStep) + " is " + str(newCost))
            if newCost < curCost:
                if regTerm > 0.0:
                    print("cost: " + str(curCost) + " => " + str(newCost) + " (" + str(newCost-regTerm) + " + " + str(regTerm) + ")")
                else:
                    print("cost: " + str(curCost) + " => " + str(newCost))
                newFinalCost = newCost
                return curStep
            else:
                curStep *= 0.5
        
        # An improved solution cannot be found so just return a zero step size
        return 0.0
        
    def Pstar(self, g):
        # Computes the backprojection which is the adjoint (i.e., transpose) of the forward projection
        # of the X-ray/Attenuated X-ray Transform
        # The "star" in the function name is for the adjoint
        # This is essentially the "forward" model for VAM
        # One can add more linear operators to this function to improve the accuracy
        # of the LIF to account for new physics.  Just remember that these must be added to
        # both Pstar and P
        # A description for how to do this is in the Discussion and Conclusions section of the VAM paper
        #startTime = time.time()
        f = leapct.allocateVolume()
        leapct.backproject(g,f)
        #print('backproject time: ' + str(time.time()-startTime))
        return f
        
    def P(self, f):
        # Computes the forward projection of the X-ray/Attenuated X-ray Transform
        # which for VAM is actually the adjoint of the forward model
        #startTime = time.time()
        g = leapct.allocateProjections()
        leapct.project(g,f)
        #print('project time: ' + str(time.time()-startTime))
        return g
        
    def ramp(self, f):
        # Applies the 2D ramp filter to the input volume
        # The frequency response of the 2D ramp filter is given by
        # H_2D(X,Y) = sqrt(H^2(X) + H^2(Y) - H^2(X)*H^2(Y) / H^2(1/2)), where
        # H(X) = 2*|sin(pi*X)|
        # See LEAP user's manual
        leapct.rampFilterVolume(f)
        return f
    
    def calculateScalingCoefficients(self):
        # Need to get rid of scaling factor introduced by the LEAP backprojectors
        sampleConstant = (self.voxelSize*self.voxelSize) / self.pixelSize # mm
        if self.alpha != 0.0:
            scalingFactors = sampleConstant / self.alpha
            if self.Omega > 0.0:
                scalingFactors *=  self.Omega
            if self.D_c > 0.0:
                scalingFactors *= (2.0 * self.D_c)
            return scalingFactors
        else:
            scalingFactors = sampleConstant;
            if self.Omega > 0.0:
                scalingFactors *= self.Omega
            if self.D_c > 0.0:
                scalingFactors *= (2.0 * self.D_c)
            return scalingFactors
    
    def highFreqAtten(self, f):
        # Blurs the volume along each of the three axes by this filter: [0.25, 0.5, 0.25]
        leapct.BlurFilter(f,2.0)
        return f
        
    def clipOutsideOfSupport(self, f):
        leapct.windowFOV(f)
        return f
        
    def clipLowAndHigh(self, x):
        if self.clipHigh > 0.0:
            x = np.clip(x, 0.0, clipHigh)
        else:
            x[x<0.0] = 0.0
        return x

    def setLEAPparams(self, targetGeometry):
        self.numberOfVerticalRaySamples = targetGeometry.shape[0]
        self.numberOfHorizontalRaySamples = max(targetGeometry.shape[1], targetGeometry.shape[2])
    
        numAngles = self.numberOfAngles
        numRows = self.numberOfVerticalRaySamples
        numCols = self.numberOfHorizontalRaySamples
        pixelSize = self.pixelSize
        leapct.set_parallelbeam(numAngles, numRows, numCols, pixelSize, pixelSize, 0.5*(numRows-1), 0.5*(numCols-1), leapct.setAngleArray(numAngles, self.angularRange))
        leapct.set_volume(targetGeometry.shape[2], targetGeometry.shape[1], targetGeometry.shape[0], pixelSize, pixelSize)
    
        rFOV = 0.5*(numCols-1)*pixelSize
        dxfov = 2.0*rFOV
        if self.resinDiameter > 0.0:
            dxfov = min(dxfov, self.resinDiameter)
        leapct.set_diameterFOV(dxfov)
        
        if self.alpha != 0.0:
            leapct.set_cylindircalAttenuationMap(self.alpha, 0.5*dxfov)
		        
    def setInitialGuess(self, targetGeometry):
        # This function calculates the initial guess of the LIF
        # and its resulting dose distribution
        # This numerical solver in this class will ONLY succeed if this
        # initial guess is close enough to the true solution

        # Calculate SART-based initial guess
        # See equations (11-13) in Section 3.3 of the VAM paper
        f = targetGeometry.copy()
        if leapct.muSpecified() == False:
            # In the case of no attenuation and parallel-beam, most of those
            # terms in equation (12) don't matter
            #g = self.P(f)
            
            g = leapct.allocateProjections()
            leapct.project(g, f)
            
        else:
            leapct.flipAttenuationMapSign()
            
            '''
            OnesVolume = targetGeometry.copy()
            OnesVolume[:] = 1.0
            POne = self.P(OnesVolume)
            
            PstarOne = leapct.sensitivity()
            
            PstarOne[PstarOne <= 0.0] = 1.0
            g = self.P(f / PstarOne)
            g[POne>0.0] = g[POne>0.0] / POne[POne>0.0]
            #'''
            
            #'''
            PstarOne = leapct.sensitivity()
            PstarOne[PstarOne <= 0.0] = 1.0
            g = self.P(f / PstarOne)
            #'''
            
            leapct.flipAttenuationMapSign()
        
        # Calculate scaled dose map
        Pstar_g = leapct.allocateVolume()
        leapct.backproject(g, Pstar_g)
        
        # Apply the proper slice-by-slice scaling (equation 13 in the VAM paper)
        # to have a LIF that delivers the correct does
        g, Pstar_g = self.scaleInitialGuess(g, targetGeometry, Pstar_g)
        
        if self.boostFactor > 0.0:
            # The SART-based initial guess leads to a blurry dose map
            # This "boosting" routine boosts the high frequencies
            # which may lead to a better guess
            
            # Calculate the ramp filtered forward projection, PRf
            # g_2 = clip(P(R(targetGeometry)))
            Rf = self.ramp(targetGeometry)
            Rf *= 1.0/(2.0*self.numberOfAngles*self.pixelSize)
            g_2 = self.P(Rf)
            g_2[g_2<0.0] = 0.0
            
            Pstar_g_2 = self.Pstar(g_2)
            g_2, Pstar_g_2 = self.scaleInitialGuess(g_2, targetGeometry, Pstar_g_2)

            g *= 1.0-boostFactor
            g = g + boostFactor*g_2
            Pstar_g = self.Pstar(g)
            
            # Need to rescale now that two terms were added
            g, Pstar_g = self.scaleInitialGuess(g, targetGeometry, Pstar_g)
        elif self.seedType == self.HYBRID or self.seedType == self.RAMP_FILTERED:
            # In the OSMO paper the authors start with a LIF that is the forward projection
            # of the 2D ramp filter of the target geometry.  This is a good guess because,
            # without the clipping step, the backprojection of this would lead to a perfect
            # dose distribution.
            
            # This initial guess strategy is based on the observation that
            # the optimal dose distribution is often a the FBP reconstruction
            # of the forward projection of the target geometry shifted by a constant,
            # i.e., FBP( P( targetGeometry) )
            # Thus we preempt this by adding in this shift at the beginning
            
            # First set a dose distribution of 0.45 everywhere inside the vial
            # we smooth the edges of this volume to avoid ringing artifacts
            Rf = targetGeometry.copy()
            Rf[:] = 0.45
            self.clipOutsideOfSupport(Rf)
            self.highFreqAtten(Rf)
            
            # Now add 0.8 * targetGeometry and perform the ramp filter
            Rf = Rf + 0.8*targetGeometry
            Rf = self.ramp(Rf)
            Rf *= 1.0/(2.0*self.numberOfAngles*self.pixelSize)
            
            # Then the initial guess is a forward projection of this, and clip off the zeros
            g_2 = self.P(Rf)
            g_2[g_2<0.0] = 0.0

            # Now calculate the dose delivered by this initial guess
            Pstar_g_2 = self.Pstar(g_2)
            g_2, Pstar_g_2 = self.scaleInitialGuess(g_2, targetGeometry, Pstar_g_2)
            
            if self.seedType == self.RAMP_FILTERED:
                # Use the shift ramp filter initial guess
                Pstar_g[:] = Pstar_g_2[:]
                g[:] = g_2[:]
            else:
                # Use a combination of the SART-based and Ramp filter based initial guesses
                g, Pstar_g = self.scaleInitialGuesses(g, targetGeometry, Pstar_g, g_2, Pstar_g_2)
    
        return g, Pstar_g

    def scaleInitialGuesses(self, g_1, targetGeometry, Pstar_g_1, g_2, Pstar_g_2):
        # This function calculates the optimal weighted combination of two
        # initial guesses (see end of Section 3.3 in the VAM paper)
        f = targetGeometry # just a shortcut
        g = g_1;
        Pstar_g = Pstar_g_1
        
        # Calculate some inner products
        g_one_dot_f = np.zeros(f.shape[0])
        g_two_dot_f = np.zeros(f.shape[0])
        g_one_dot_g_two = np.zeros(f.shape[0])
        g_one_dot_g_one = np.zeros(f.shape[0])
        g_two_dot_g_two = np.zeros(f.shape[0])
        for k in range(f.shape[0]):
        
            f_slice = np.squeeze(f[k,:,:])
            Pstar_g_1_slice = np.squeeze(Pstar_g_1[k,:,:])
            Pstar_g_2_slice = np.squeeze(Pstar_g_2[k,:,:])
            
            ind = np.logical_or(f_slice != np.roll(f_slice, 1, axis=0), f_slice != np.roll(f_slice, 1, axis=1))
            g_one_dot_f[k] = self.T*np.sum(Pstar_g_1_slice[ind])
            g_two_dot_f[k] = self.T*np.sum(Pstar_g_2_slice[ind])
            g_one_dot_g_two[k] = np.sum(Pstar_g_1_slice[ind] * Pstar_g_2_slice[ind])
            g_one_dot_g_one[k] = np.sum(Pstar_g_1_slice[ind] * Pstar_g_1_slice[ind])
            g_two_dot_g_two[k] = np.sum(Pstar_g_2_slice[ind] * Pstar_g_2_slice[ind])

        # Set the scaling weight between 1 and 2
        scalars_1 = np.zeros(f.shape[0])
        scalars_2 = np.zeros(f.shape[0])        
        denom = g_two_dot_g_two*g_one_dot_g_one - g_one_dot_g_two*g_one_dot_g_two
        ind = np.abs(denom) > 1.0e-16
        scalars_2[ind] = (g_two_dot_f[ind]*g_one_dot_g_one[ind] - g_one_dot_f[ind]*g_one_dot_g_two[ind]) / denom[ind];

        ind = g_one_dot_g_one > 0.0
        scalars_2[ind] = np.clip(scalars_2[ind], 0.0, np.max(scalars_2));
        scalars_1[ind] = (g_one_dot_f[ind] - scalars_2[ind]*g_one_dot_g_two[ind]) / g_one_dot_g_one[ind]

        scalars_1[scalars_1 < 0.0] = 0.0
        scalars_2[scalars_1 < 0.0] = 1.0
        
        scalars_2[~ind] = 0.0
        scalars_1[~ind] = 0.0
        
        scalars_1[np.abs(denom) <= 1.0e-16] = 1.0;
        scalars_2[np.abs(denom) <= 1.0e-16] = 0.0;
        
        scalars_1 = scalars_1.astype('float32')
        scalars_2 = scalars_2.astype('float32')
        
        # Compute the weighted combination of 1 and 2
        g = scalars_1.reshape((1, len(scalars_1), 1))*g_1 + scalars_2.reshape((1, len(scalars_2), 1))*g_2
        Pstar_g = scalars_1.reshape((len(scalars_1), 1, 1))*Pstar_g_1 + scalars_2.reshape((len(scalars_2), 1, 1))*Pstar_g_2
        return g, Pstar_g
        
    def scaleInitialGuess(self, g, targetGeometry, Pstar_g):
        # This function calculates the optimal scaling of the
        # initial guess as described in eqauation (13) in the VAM paper
        f = targetGeometry # just a shortcut
        
        # Scale initial guess
        # This method works by determining the value to scale the target volume
        # so that its values on the boundary are closest to the critical dose
        for k in range(f.shape[0]):
            
            f_slice = np.squeeze(f[k,:,:])
            Pstar_g_slice = np.squeeze(Pstar_g[k,:,:])
            
            # Mark the boundary pixels in the target geometry
            ind = np.logical_or(f_slice != np.roll(f_slice, 1, axis=0), f_slice != np.roll(f_slice, 1, axis=1))
            scalingFactors_denom = np.sum(Pstar_g_slice[ind]**2)
            
            if scalingFactors_denom > 0.0:
                scalingFactors = self.T*np.sum(Pstar_g_slice[ind]) / scalingFactors_denom
                g[:,k,:] = g[:,k,:] * scalingFactors
                Pstar_g[k,:,:] = Pstar_g[k,:,:] * scalingFactors
            #print('scaling[' + str(k) + '] = ' + str(scalingFactors_denom))
        
        return g, Pstar_g
        
    def CGsolver(self, g, Pstar_g, targetGeometry):
        # Called "execute_full" in LTT
        # See Section 3.2 in the VAM paper
        
        # I'm not completely sure why I need this line
        # I just copied it from LTT
        f_T = self.u(targetGeometry.copy(), self.T)
        
        print('initial cost = ' + str(self.calculateCostFunction(Pstar_g, f_T)))
        #print('Jaccard Index = ' + str(self.JaccardIndex(Pstar_g, f_T)))
        print('Pixel Error Rate = ' + str(self.pixelErrorRate(Pstar_g, f_T)))
        print(' ')
        
        doCG = True
        doCG_save = doCG
        
        # Perform iterations
        grad_dot_grad_old = 0.0
        grad_old_dot_grad_old = 0.0
        grad_dot_grad = 0.0
        d = g.copy()
        d_old = None
        if doCG == True:
            d_old = g.copy()
            grad_old = g.copy()
        for n in range(1,self.N_iter+1):
            print('Iteration ' + str(n) + ' of ' + str(self.N_iter))
            ### Calculate gradient
            grad = self.costGradient(Pstar_g, f_T)
            
            ### Calculate CG descent direction
            grad_dot_grad = np.sum(grad**2)
            d[:] = -grad[:]
            if n != 1 and n % self.CGrestart != 0 and doCG == True:
                if grad_old is not None:
                    grad_dot_grad_old = np.sum(grad * grad_old)
                else:
                    grad_dot_grad_old = 0.0
                beta_local = (grad_dot_grad - grad_dot_grad_old) / grad_old_dot_grad_old
                if grad_old is not None:
                    beta_local = max(0.0, beta_local)
                d += beta_local*d_old
            
            # Check if the CG step is a descent direction
            # If not, revert to gradient descent
            num = -np.sum(d*grad)
            if num < 0.0:
                # revert to gradient descent
                print('CG step diverges; reverting to gradient descent')
                d[:] = -grad[:]
            if grad_old is not None:
                grad_old[:] = grad[:]
            
            ### Calculate initial guess of the step size
            denomA = self.costHessianQuadraticForm(Pstar_g, f_T, d)
            if self.beta > 0.0:
                denomB = 2.0*beta*np.sum(d*d)
            else:
                denomB = 0.0
            alpha = num / (denomA + denomB)
            if alpha < 0.0:
                print("Encountered negative step size, quitting!")
                break
            
            # Refine step size for non-negativity constraint
            alpha = self.calculateStepSize(g, Pstar_g, d, f_T, alpha)
            if alpha <= 0.0:
                if doCG and n>1:
                    # revert to gradient descent
                    doCG = False
                    continue
                else:
                    print("Cost function cannot be reduced further, quitting!")
                    break
            
            # Update solution
            g += alpha*d
            self.clipLowAndHigh(g)
            leapct.backproject(g, Pstar_g)
            #Pstar_g = self.Pstar(g)
            
            # Save old descent direction and gradient for CG algorithm
            grad_old_dot_grad_old = grad_dot_grad
            if d_old is not None:
                d_old[:] = d[:]
            
            doCG = doCG_save
            print('Pixel Error Rate = ' + str(self.pixelErrorRate(Pstar_g, f_T)))
            #print('Jaccard Index = ' + str(self.JaccardIndex(Pstar_g, f_T)))
            print(' ')
        
        return g, Pstar_g

    def costGradient(self, Pstar_g, f_T):
        # Set Phi'(g), the gradient of the cost function
        # See page 6 of the VAM paper
        tempVolume = f_T.copy()
        tempVolume = (self.u(Pstar_g - self.T) - f_T) * self.up(Pstar_g - self.T)
        if self.W is not None:
            tempVolume *= self.W
        grad = self.P(tempVolume)
        if self.beta > 0.0:
            grad += 2.0*beta*g
        return grad

    def costHessianQuadraticForm(self, Pstar_g, f_T, d):
    
        # Calculate the quadratic form of the Hessian of the cost function
        # This function calculate the denominator of equation (10) in the VAM paper
        # See also Phi''(g) on page 6 of the VAM paper
        centerOfQuad = f_T.copy()
        centerOfQuad = (self.u(Pstar_g - self.T) - f_T) * self.upp(Pstar_g - self.T) + self.up(Pstar_g - self.T)**2
        if self.W is not None:
            centerOfQuad *= self.W
        innerProdTerm = self.Pstar(d)
        return np.sum(centerOfQuad * innerProdTerm**2)
        
    def pixelErrorRate(self, f_est, f_true):
        retVal = 0.0
        retVal += f_est[np.logical_and(f_est > self.T, f_true <= self.T)].size
        retVal += f_est[np.logical_and(f_est <= self.T, f_true > self.T)].size
        return retVal/f_est.size

    def JaccardIndex(self, f_est, f_true):
        num = 0.0
        denom = 0.0
        ind_est = f_est > self.T
        ind_true = f_true > self.T
        num += f_est[np.logical_and(ind_est, ind_true)].size
        denom += f_est[np.logical_or(ind_est, ind_true)].size
        return num/denom
        