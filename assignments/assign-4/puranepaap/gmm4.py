import numpy as np
import os
import matplotlib
from random import randint
from datetime import datetime
# import grapher as gp
import errno

start_time = datetime.now()
X = []
meanVect = np.zeros(2)
gammaVect = np.zeros(shape=(2, 2), dtype=np.float64)
piVect = np.zeros(2, dtype=np.float64)
covMatVect = []
dimensions = 2
noOfClusters = 2
threshold = 0.001
noOfPoints = 0


apniList = []

#   Calculate gaussian function value for some x, mean and covMat
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


#   Updating the entire covariance matrix vector using updateCovMatK() on K elements
def updateCovMatVector():
    # global iterationCount
    for k in range(noOfClusters):
        sigma = np.zeros(shape=(dimensions, dimensions))
        gammaSum = 0.0
        for n in range(noOfPoints):
            deviation = np.copy(X[n]-meanVect[k])
            deviation = deviation.reshape(dimensions, 1)
            deviation = np.matmul(deviation, np.transpose(deviation))

            sigma = sigma + (gammaVect[k, n] * deviation)
            # print("gammValur = ", gammaVect[k, n])
            # print("Gasmmsmmm = ", gammaSum)
            gammaSum += gammaVect[k,n]

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

    
    loopTime = datetime.now()

    while iterationCount <= 40:

        iterationCount += 1
        updateGammaVect()
        updateMeanVect()
        updatePiVect()
        updateCovMatVector()

        lPrev = lCurrent
        lCurrent = logLikelihood()
        print(lCurrent)
        apniList.append(lCurrent)
        # print("iteration = ", iterationCount, "lCurrent = ", lCurrent)

        if lPrev != -1 and abs(lPrev-lCurrent) < threshold: 
            return

def initialize(pointsAssignCluster):
    global piVect
    global gammaVect
    # print("Initializing ")

    # print("Initializing gamma")
    for n in range(noOfPoints):
        # print("Assign pt: ",pointsAssignCluster[n])
        gammaVect[int(pointsAssignCluster[n]),n] = 1
        piVect[int(pointsAssignCluster[n])] += 1

    # print("Initializing Pi Vector")
    piVect /= noOfPoints
  
    # print("updating cov mat")
    for i in range(noOfClusters):
        covMatVect.append(0)

    updateCovMatVector()
    for i in range(noOfClusters):
        for j in range(dimensions):
            covMatVect[i][j][j]
    # print(covMatVect)
    # print("updated cov mat")

def assignCluster():
    clusterAssignment = []
    for i in range(noOfPoints):
        clusterAssignment.append(np.argmax(gammaVect[:,i]))
    return clusterAssignment


def master(threshold, noOfPoints, X, dimensions, noOfClusters, meanVect, pointsAssignCluster, outPath):
    # print("In gmm ")
    globals()['threshold'] = threshold
    globals()['noOfPoints'] = noOfPoints
    globals()['X'] = X
    globals()['dimensions'] = dimensions
    globals()['noOfClusters'] = noOfClusters
    globals()['meanVect'] = meanVect

    # print("Threshold: ", threshold)
    # print("Clusters: ", noOfClusters)
    # print("Dimensions: ", dimensions)
    # print("Mean vector: ",meanVect)

    globals()['gammaVect'] = np.zeros(shape=(noOfClusters, noOfPoints),dtype=np.float64)
    globals()['piVect'] = np.zeros(noOfClusters, dtype=np.float64)
    initialize(pointsAssignCluster)
    algorithmEM()
    return apniList, meanVect, piVect, covMatVect
    # print(meanVect)
