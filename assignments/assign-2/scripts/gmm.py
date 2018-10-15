import numpy as np
import os
import matplotlib


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


def updateCovMatVector(noOfPoints, X, meanVect, gammaVect, dimensions, clusters):
    covMatVect = []
    for k in range(clusters):
        covMatVect.append(updateCovMatK(gammaVect[k], X, meanVect[k], noOfPoints, dimensions))
    return covMatVect


def updateMuK(covMat):
    pass


def updateMuVect():
    pass


def updatePiK():
    pass


def updatePiVect():
    pass


def updateGammaNK():
    pass


def updateGammaVect():
    pass


def updateParameterVectors():
    pass
