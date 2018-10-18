import numpy as np
import os
import matplotlib
from random import randint

X = []
meanVect = np.zeros(2)
gammaVect = np.zeros(shape=(2,2))
piVect = np.zeros(2)
covMatVect = []
dimensions = 2
clusters = 2
threshold = 0.001
noOfPoints = 0

#   Calculate gaussian function value for some x, mean and covMat
def gaussian(covMat, x, mean):
    numFeature = np.size(mean,1)
    gaussian = -(1/2)*((np.transpose(x-mean)*(np.linalg.inv(covMat)))*(x-mean))
    gaussian = np.exp(gaussian)
    deter = np.linalg.det(covMat)
    gaussian *= deter**(-1./2)
    gaussian *= (2*np.pi)**(-numFeature/2.)
    return gaussian


#   Updating the entire covariance matrix vector using updateCovMatK() on K elements
def updateCovMatVector():
    # covMatVect = []
    for k in range(clusters):
        # covMatVect.append(updateCovMatK(gammaVect[k], X, meanVect[k], noOfPoints, dimensions))
        sigma = np.zeros(shape=(dimensions, dimensions))
        gammaSum = 0
        for n in range(noOfPoints):
            sigma += gammaVect[k,n]*(X[n]-meanVect[k])*np.transpose(X[n]-meanVect[k])
            gammaSum += gammaVect[k,n]
        sigma /= gammaSum
        for i in range(dimensions):
            for j in range(dimensions):
                if i != j:
                    sigma[i][j] = 0
    covMatVect[k] = sigma


def updateMeanVect():
    # meanVect = []
    for k in range(clusters):
        # meanVect.append(updateMeanK(gammaVect[k], X, noOfPoints, dimensions))
        mean = np.zeros(shape=(dimensions))
        gammaSum = 0
        for n in range(noOfPoints):
            mean += gammaVect[k, n]*X[n]
            gammaSum += gammaVect[k, n]
        mean = mean/gammaSum
        meanVect[k] = mean
 

def updatePiVect():
    # piVect = []
    for k in range(clusters):
        # piVect.append(updatePiK(gammaVect[k], noOfPoints))
        piK = 0
        for n in range(noOfPoints):
            piK += gammaVect[k,n]
        piK /= noOfPoints
        piVect[k] = piK


def updateGammaVect():
    # gammaVect = []
    for k in range(clusters):
        # gammaRow = []
        for n in range(noOfPoints):
        #     gammaRow.append(updateGammaNK(k, piVect[k], X[n], covMatVect[k], meanVect[k], clusters))
        # gammaVect.append(gammaRow)
            gamma = 0
            for ind in range(clusters):
                gamma += piVect[k]*gaussian(covMatVect[ind], X[n], meanVect[ind])
            gamma /= gaussian(covMatVect[k], X[n], meanVect[k])
        gammaVect[k,n] = gamma

def logLikelihood():
    l = 0
    for n in range(noOfPoints):
        for k in range(clusters):
            l += piVect[k]*gaussian(covMatVect[k], X[n], meanVect[k])
    return np.log(l)

def algorithmEM():
    # global X;global meanVect;global gammaVect;global piVect;global covMatVect
    # global dimensions;global clusters;global threshold;global noOfPoints
    
    lPrev = 0
    lCurrent = -1

    iterationCount = 0
    while True:
        print("Iteration No. : ", iterationCount)
        iterationCount += 1
        updateGammaVect()
        updateMeanVect()
        updatePiVect()
        updateCovMatVector()

        lPrev = lCurrent
        lCurrent = logLikelihood()

        if lPrev != -1 and (lPrev-lCurrent) < threshold:
            break

def initialize():
    for k in range(clusters):
        piVect[k] = 1/clusters
    
    for n in range(noOfPoints):
        gammaVect[randint(0,clusters-1),n] = 1

    updateCovMatVector()


def master(threshold, noOfPoints, X, dimensions, clusters, covMatVect, meanVect):
    globals()['threshold'] = threshold
    globals()['noOfPoints'] = noOfPoints
    globals()['X'] = X
    globals()['dimensions'] = dimensions
    globals()['clusters'] = clusters
    globals()['meanVect'] = meanVect

    globals()['gammaVect'] = np.zeros(shape=(clusters, noOfPoints))
    globals()['piVect'] = np.zeros(clusters)
