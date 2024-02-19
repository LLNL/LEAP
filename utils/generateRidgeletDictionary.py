import numpy as np
from leapctype import *

def generateRidgeletDictionary(N_max=8):
    numTermsPerAngle = N_max*(N_max-1)//2
    # 4 ==> 4*3/2 = 6
    # 6 ==> 6*5/2 = 15
    # 8 ==> 8*7/2 = 28

    numAngles = N_max

    numTerms = numAngles * numTermsPerAngle

    basisFunctions = np.zeros((numTerms+1, 1, N_max, N_max), dtype=np.float32)

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
        for N in range(1,N_max+1):
            for k in range(1,N):
                y = np.cos(np.pi/float(N)*ns*float(k)) # DCT-I
                #y = np.cos(np.pi/float(N)*(ns+0.5)*float(k)) # DCT-II
                #y = y - np.mean(y)
                #y = y / np.sqrt(np.sum(y**2))
                g[:,count,:] = y
                count += 1
        leapct.backproject(g,f)
        
        for k in range(numTermsPerAngle):
            atom = np.zeros((N_max, N_max), dtype=np.float32)
            atom[:,:] = f[k,:,:]
            atom = atom - np.mean(atom)
            atom = atom / np.sqrt(np.sum(atom**2))
            #print(str(np.sum(atom)) + ' ' + str(np.sum(atom**2)))
            basisFunctions[iphi*numTermsPerAngle+k,0,:,:] = atom[:,:]
    
    atom = np.zeros((N_max, N_max), dtype=np.float32)
    atom[:,:] = 1.0
    #atom = atom - np.mean(atom)
    atom = atom / np.sqrt(np.sum(atom**2))
    #print(str(np.sum(atom)) + ' ' + str(np.sum(atom**2)))
    basisFunctions[numTerms,:,:,:] = atom[:,:]
    #leapct.display(np.squeeze(basisFunctions))
    return basisFunctions
