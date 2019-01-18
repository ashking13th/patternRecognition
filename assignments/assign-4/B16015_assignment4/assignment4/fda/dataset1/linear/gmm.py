import numpy as np
# import kmeans
import matplotlib

noOfClusters = 0
noOfPoints=0
nFeature=0
dimensions=0
X=[]

meanVect = np.zeros((noOfClusters,nFeature),float)
gammaVect = np.zeros((noOfClusters,noOfPoints),float)
piVect = np.zeros((noOfClusters),float)
covMatVect = np.zeros((noOfClusters,nFeature,nFeature),float)

def applyGMM(ncluster,ipData,gamma,meanV):
    global noOfClusters
    global noOfPoints
    global nFeature
    global dimensions
    global X
    X=ipData
    nFeature=len(ipData[0])
    dimensions=nFeature
    noOfPoints=len(ipData)
    noOfClusters=ncluster

    initialize(gamma,meanV)
    algorithmEM()
    # print(gammaVect)
    return piVect,meanVect,covMatVect

def gaussian(covMat, x, mean):
    # print(mean)
    numFeature = np.size(mean)
    gaussian = -(1/2)*np.sum((np.transpose(x-mean)*(np.linalg.inv(covMat)))*(x-mean))
    # gaussian = (-1/2)*(np.dot(np.transpose(x-mean), np.dot(np.linalg.inv(covMat), x-mean)))
    gaussian = np.exp(gaussian)
    deter = np.linalg.det(covMat)
    gaussian *= deter**(-1./2)
    gaussian *= (2*np.pi)**(-numFeature/2.)
    # print("Gaussian: ", gaussian)
    return gaussian

def updateCovMatVector():
    # global iterationCount
    # dimensions=nFeature
    for k in range(noOfClusters):
        sigma = np.zeros((dimensions, dimensions),float)
        gammaSum = 0.0
        for n in range(noOfPoints):
            deviation = np.copy(X[n]-meanVect[k])
            deviation = deviation.reshape(dimensions, 1)
            deviation = np.matmul(deviation, np.transpose(deviation))

            sigma = sigma + (gammaVect[k, n] * deviation)
            # print("gammValur = ", gammaVect[k, n])
            # print("Gasmmsmmm = ", gammaSum)
            gammaSum += gammaVect[k,n]
        if(gammaSum!=0.0):
            sigma /= gammaSum
        for i in range(dimensions):
            for j in range(dimensions):
                if i != j:
                    sigma[i,j] = 0
                elif sigma[i, j] == 0:
                    sigma[i, j] += 1e-6
        covMatVect[k] = sigma

def updateMeanVect():
    # meanVect = []
    for k in range(noOfClusters):
        mean = np.zeros(shape=(dimensions), dtype=np.float64)
        gammaSum = 0
        for n in range(noOfPoints):
            mean += gammaVect[k, n]*X[n]
            gammaSum += gammaVect[k, n]
        mean = mean/gammaSum
        meanVect[k] = mean
        # print("k = ", k, " : MEan = ", meanVect[k])
 

def updatePiVect():
    # piVect = []
    for k in range(noOfClusters):
        piK = 0.0
        for n in range(noOfPoints):
            piK += gammaVect[k,n]
        piK /= noOfPoints
        piVect[k] = piK
        # print()
    # print(piVect)


def updateGammaVect():
    # print("Pi Vector: ",piVect)
    for n in range(noOfPoints):
        gamma = 0.0
        for ind in range(noOfClusters):
            gamma += piVect[ind]*gaussian(covMatVect[ind], X[n], meanVect[ind])

        for k in range(noOfClusters):
            if(gamma == 0.0):
                gammaVect[k, n] = piVect[k]    
            else:
                gammaVect[k, n] = (piVect[k]*gaussian(covMatVect[k], X[n], meanVect[k]))/gamma

def logLikelihood():
    likelihood = 0
    for n in range(noOfPoints):
        l = 0
        for k in range(noOfClusters):
            # print("covMatrix = ", covMatVect[k])
            temp = gaussian(covMatVect[k], X[n], meanVect[k])
            if(temp == 0.0):
                l += piVect[k]
            else:
                l += piVect[k]*gaussian(covMatVect[k], X[n], meanVect[k])
            # if l == 0:
                # print("MYGOD!!!!!!!!!!!\n")
        likelihood += np.log(l)
    return likelihood


def algorithmEM():
    print("GMM start_tr")
    iterationCount = 0    # print("GMM Start")
    lPrev = 0
    lCurrent = -1

    
    # loopTime = datetime.now()

    while iterationCount <= 40:
        # print(iterationCount)

        iterationCount += 1
        updateGammaVect()
        updateMeanVect()
        updatePiVect()
        updateCovMatVector()

        # lPrev = lCurrent
        # lCurrent = logLikelihood()
        # print(lCurrent)
        # apniList.append(lCurrent)
        # # print("iteration = ", iterationCount, "lCurrent = ", lCurrent)

        # if lPrev != -1 and abs(lPrev-lCurrent) < threshold: 
            # return

def initialize(gamma,meanV):
    global piVect
    global covMatVect
    global gammaVect
    global meanVect
    meanVect = np.zeros((noOfClusters,nFeature),float)
    gammaVect = np.zeros((noOfClusters,noOfPoints),float)
    piVect = np.zeros((noOfClusters),float)
    covMatVect = np.zeros((noOfClusters,nFeature,nFeature),float)

    meanVect=meanV
    gammaVect=np.transpose(gamma)
    # print(gnVect)
    # print("Initializing ")

    # print("Initializing gamma")
    for n in range(noOfPoints):
        # print("Assign pt: ",pointsAssignCluster[n])
        # gammaVect[int(pointsAssignCluster[n]),n] = 1
        for i in range(noOfClusters):
            piVect[i] += gammaVect[i][n]

    # print("Initializing Pi Vector")
    piVect /= noOfPoints
  
    # print("updating cov mat")
    updateCovMatVector()
    # print(covMatVect)
    # print("updated cov mat")

# def assignCluster():
#     clusterAssignment = []
#     for i in range(noOfPoints):
#         clusterAssignment.append(np.argmax(gammaVect[:,i]))
# return clusterAssignment