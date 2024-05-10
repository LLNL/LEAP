################################################################################
# This script generates some sample dictionaries and
# will certainly change in upcoming releases
################################################################################
import numpy as np
from leapctype import *

def GramSchmidt(basisFcns):
    numDims = len(basisFcns.shape)-1
    if numDims < 1 or numDims > 3:
        return None
    
    for n in range(basisFcns.shape[0]):
        if numDims == 1:
            u = basisFcns[n,:]
        elif numDims == 2:
            u = basisFcns[n,:,:]
        elif numDims == 3:
            u = basisFcns[n,:,:]
        u = u / np.sqrt(np.sum(u**2))
        for m in range(0,n):
            if numDims == 1:
                v = basisFcns[m,:]
            elif numDims == 2:
                v = basisFcns[m,:,:]
            elif numDims == 3:
                v = basisFcns[m,:,:]
            u = u - np.sum(u*v)*v
        u = u / np.sqrt(np.sum(u**2))
        
        if numDims == 1:
            basisFcns[n,:] = u
        elif numDims == 2:
            basisFcns[n,:,:] = u
        elif numDims == 3:
            basisFcns[n,:,:] = u
    
    return basisFcns

def LegendrePolynomialBasis(order, x=None):
    from scipy.special import legendre

    if x is None:
        x = (np.array(range(order))-float(order-1)/2.0)*2.0/float(order)
        
    poly = np.zeros((order,x.size))
    for N in range(order):
        coeff = np.flip(np.array(legendre(N)))
        p = x.copy()
        p[:] = 0.0
        
        for n in range(len(coeff)):
            p = p + coeff[n]*x**n
        p = p / np.sqrt(np.sum(p**2))
        poly[N,:] = p[:]
    return poly

def BumpBasis(N,scale=2.0):
    basisFcns = np.zeros((N,N))
    for k in range(0,N):
        n = k
        basisFcns[k,(n-1)%N] = 0.25
        basisFcns[k,n] = 0.5
        basisFcns[k,(n+1)%N] = 0.25
    return basisFcns

def DCTBasis(N,type=2):

    ns = np.array(range(N))

    basisFcns = np.zeros((N,N))
    for k in range(0,N):
        if type == 2:
            basisFcns[k,:] = np.cos(np.pi/float(N)*(ns+0.5)*float(k)) # DCT-II
        else:
            basisFcns[k,:] = np.cos(np.pi/float(N)*ns*float(k)) # DCT-I
            
    return basisFcns

def generateSeparableBasis(basisFcns,do3D=False):
    
    N_max = basisFcns.shape[0]
    if do3D == False:
        numTerms = N_max*N_max
        basisFunctions = np.zeros((numTerms, 1, N_max, N_max), dtype=np.float32)
        count = 0
        x = np.zeros((N_max,1))
        y = np.zeros((N_max,1))
        for i in range(N_max):
            x[:,0] = basisFcns[i,:]
            for j in range(N_max):
                y[:,0] = basisFcns[j,:]
                atom = np.matmul(x,y.T)
                
                if count > 0:
                    atom = atom - np.mean(atom)
                atom = atom / np.sqrt(np.sum(atom**2))
                
                basisFunctions[count,0,:,:] = atom
                count += 1
    else:
        numTerms = N_max*N_max*N_max
        basisFunctions = np.zeros((numTerms, N_max, N_max, N_max), dtype=np.float32)
        atom = np.zeros((N_max, N_max, N_max), dtype=np.float32)
        count = 0
        x = np.zeros((N_max,1))
        y = np.zeros((N_max,1))
        z = np.zeros((N_max,1))
        for i in range(N_max):
            x[:,0] = basisFcns[i,:]
            for j in range(N_max):
                y[:,0] = basisFcns[j,:]
                for k in range(N_max):
                    z[:,0] = basisFcns[k,:]

                    x_3d,y_3d,z_3d = np.meshgrid(x,y,z,indexing='ij')
                    atom = x_3d*y_3d*z_3d
                
                    if count > 0:
                        atom = atom - np.mean(atom)
                    atom = atom / np.sqrt(np.sum(atom**2))
                    
                    basisFunctions[count,:,:,:] = atom
                    count += 1
        
    return basisFunctions


def generateLegendreDictionary(N_max=8,do3D=False):
    
    basisFcns = LegendrePolynomialBasis(N_max)
    basisFcns = GramSchmidt(basisFcns)
    return generateSeparableBasis(basisFcns, do3D)


def generateDCTDictionary(N_max=8,do3D=False):

    basisFcns = DCTBasis(N_max)
    basisFcns = GramSchmidt(basisFcns)
    return generateSeparableBasis(basisFcns, do3D)


def generateRidgeletDictionary(N_max=8):
    numTermsPerAngle = N_max-1
    numAngles = N_max+1
    numTerms = numAngles * numTermsPerAngle

    basisFcns = BumpBasis(N_max)
    #print(basisFcns.shape)

    basisFcns = DCTBasis(N_max)
    basisFunctions = np.zeros((numTerms+1, 1, N_max, N_max), dtype=np.float32)
    #print(basisFcns.shape)

    leapct = tomographicModels()
    for iphi in range(numAngles):
        phi = iphi * 180.0/float(numAngles)
        
        numRays = N_max
        leapct.set_parallelbeam(1,numTermsPerAngle,numRays,1.0,1.0,0.0,0.5*float(numRays-1),leapct.setAngleArray(1,180.0)+phi)
        leapct.set_default_volume(float(numRays)/float(N_max))
        leapct.set_diameterFOV(100.0)
        g = leapct.allocate_projections()
        f = leapct.allocate_volume()

        ns = np.array(range(numRays),dtype=np.float32)
        ys = []
        count = 0
        for N in range(N_max,N_max+1):
            for k in range(1,N):
                g[0,count,:] = basisFcns[k,:]
                count += 1
        leapct.backproject(g,f)
        
        for k in range(numTermsPerAngle):
            atom = np.zeros((N_max, N_max), dtype=np.float32)
            atom[:,:] = f[k,:,:]
            atom = atom - np.mean(atom)
            #print(np.sqrt(np.sum(atom**2)))
            atom = atom / np.sqrt(np.sum(atom**2))
            #print(str(np.sum(atom)) + ' ' + str(np.sum(atom**2)))
            basisFunctions[iphi*numTermsPerAngle+k,0,:,:] = atom[:,:]
    
    atom = np.zeros((N_max, N_max), dtype=np.float32)
    atom[:,:] = 1.0
    atom = atom / np.sqrt(np.sum(atom**2))
    #print(str(np.sum(atom)) + ' ' + str(np.sum(atom**2)))
    basisFunctions[numTerms,0,:,:] = atom[:,:]
    
    if basisFunctions.shape[0] <= basisFunctions.shape[1]*basisFunctions.shape[2]*basisFunctions.shape[3]:
        print('Performing orthogonalization...')
        basisFunctions = GramSchmidt(basisFunctions)
        
    return basisFunctions
