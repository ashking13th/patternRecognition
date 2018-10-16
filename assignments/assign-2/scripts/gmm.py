import numpy as np
import os
import matplotlib


#   Calcualte gaussian function value for some x,mean and covMat
def gaussian(covMat, x, mean):
    gaussian = -(1/2)*(x-mean)*(np.linalg.inv(covMat))*(x-mean)
    gaussian = np.exp(gaussian)
    gaussian /= 1/(np.sqrt((2*np.pi)*covMat))
    return gaussian


# Finding covariance matrix for Kth cluster
def updateCovMatK(gammaK, X, meanK, noOfPoints, dimensions):
    sigma = np.zeros(shape=(dimensions, dimensions))
    gammaSum = 0
    for n in range (noOfPoints):
        sigma += gammaK[n]*(X[n]-meanK)*np.transpose(X[n]-meanK)
        gammaSum = gammaK[n]
    sigma = sigma/gammaSum
    for i in range(dimensions):
        for j in range(dimensions):
            if i != j:
                sigma[i][j] = 0
    return sigma

#   Updating the entire covariance matrix vector using updateCovMatK() on K elements
def updateCovMatVector(noOfPoints, X, meanVect, gammaVect, dimensions, clusters):
    covMatVect = []
    for k in range(clusters):
        covMatVect.append(updateCovMatK(gammaVect[k], X, meanVect[k], noOfPoints, dimensions))
    return covMatVect


def updateMeanK(gammaK, X, noOfPoints, dimensions):
    mean = np.zeros(shape=(dimensions))
    gammaSum = 0
    for n in range(noOfPoints):
        mean += gammaK[n]*X[n]
        gammaSum += gammaK[n]
    mean = mean/gammaSum
    return mean  


def updateMeanVect(noOfPoints, X, gammaVect, dimensions, clusters):
    meanVect = []
    for k in range(clusters):
        meanVect.append(updateMeanK(gammaVect[k], X, noOfPoints, dimensions))
    return meanVect


def updatePiK(gammaK, noOfPoints):
    piK = 0
    for n in range(noOfPoints):
        piK += gammaK[n]
    piK /= noOfPoints
    return piK


def updatePiVect(noOfPoints, gammaVect, clusters):
    piVect = []
    for k in range(clusters):
        piVect.append(updatePiK(gammaVect[k], noOfPoints))
    return piVect


def updateGammaNK(k, piK, x, covMatVect, meanVect, clusters):
    gamma = 0
    for ik in range(clusters):
        gaussian(covMatVect[ik], x, meanVect[ik])
    gamma /= gaussian(covMatVect[k], x, meanVect[k])
    return gamma


def updateGammaVect(piVect, noOfPoints, X, covMatVect, meanVect, clusters):
    gammaVect = []
    
    for k in range(clusters):
        gammaRow = []
        for n in range(noOfPoints):
            gammaRow.append(updateGammaNK(k, piVect[k], X[n], covMatVect[k], meanVect[k], clusters))
        gammaVect.append(gammaRow)
    return gammaVect

def updateParameterVectors():
    pass
