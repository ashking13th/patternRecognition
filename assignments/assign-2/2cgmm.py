import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import math
import random
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as gm
import gmm
import gmm2
import grapher as gp
import errno


ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="Raw data set location")
ap.add_argument("-m1", "--mean1", required=True, help="Mean 1 location")
ap.add_argument("-p1", "--pi1", required=True, help="Mean 1 location")
ap.add_argument("-c1", "--cov1", required=True, help="Mean 1 location")
ap.add_argument("-o", "--output", required=True, help="Output mean location")
args = vars(ap.parse_args())

clusters = 3

wholeData = []

def fileHandle(fileName):
    inpData = []
    file = open(fileName)
    for line in file:
        teLine = line.rstrip('\n ').split(' ')
        nLine = [float(i) for i in teLine]
        nLine = np.array(nLine)
        inpData.append(nLine)
    file.close()
    return inpData


def fileHandle2(fileName):
    # inpData = []
    file = open(fileName)
    for line in file:
        teLine = line.rstrip('\n ').split(' ')
        nLine = [float(i) for i in teLine]
        nLine = np.array(nLine)
        wholeData.append(nLine)
    file.close()
    # return inpData


def fileHandle3(fileName):
    inpData1 = []
    file = open(fileName)
    for line in file:
        teLine = line.rstrip('\n ').split(' ')
        nLine = [float(i) for i in teLine]
        nLine = np.array(nLine)
        inpData1 = nLine
    file.close()
    return inpData1

# def fileHandle(fileName):
#     inpData = []
#     file = open(fileName)
#     for line in file:
#         teLine = line.rstrip('\n ').split(' ')
#         nLine = [float(i) for i in teLine]
#         nLine = np.array(nLine)
#         inpData.append(nLine)
#     file.close()
#     return inpData


print("Process start")
for root, dirs, files in os.walk(args["source"]):
    for f in files:
        path = os.path.relpath(os.path.join(root, f), ".")
        # print("read: ",path)
        fileHandle2(path)
        # lengthOfFile.append(len(wholeData)-cntForFile)
		# cntForFile = len(wholeData)

# print("whole = ", )

def gaussian(covMat, x, mean):
    numFeature = np.size(mean)
    gaussian = -(1/2)*np.sum((np.transpose(x-mean)
                              * (np.linalg.inv(covMat)))*(x-mean))
    gaussian = np.exp(gaussian)
    deter = np.linalg.det(covMat)
    gaussian *= deter**(-1./2)
    gaussian *= (2*np.pi)**(-numFeature/2.)
    return gaussian


def covaMat(mat, clusters, dimensions):
    print("MAT = ", mat)
    vect = np.zeros(shape=(clusters, dimensions, dimensions))
    for i in range(clusters):
        for j in range(dimensions):
            vect[i, j, j] = mat[i][j]
    return vect

def gammaAllot(x, covMatVect, meanVector, piVect, clusters):
    gammaVect = np.zeros((clusters))
    sum = 0.0
    gaussians = np.zeros((clusters))
    # print("Gamma length: ", len(gammaVect))
    # print("Pi length: ", len(piVect))
    # print("Mean length: ", len(meanVector))

    # print("pi = ", len(piVect))
    # print("")

    for k in range(clusters):
        gaussians[k] = gaussian(covMatVect[k], x, meanVector[k])
    for k in range(clusters):
        sum += piVect[k]*gaussians[k]
    for k in range(clusters):
        gammaVect[k] = (piVect[k]*gaussians[k])/sum
    ans = np.argmax(gammaVect)
    # if ans != 0:
    # print("gamma vect: ",gammaVect)
    # print("Allotment : ", ans)
    return ans


meanVect = []
piVect = []
covMatVect = []
# print(args['mean1'])
meanVect = fileHandle(args['mean1'])
piVect = fileHandle3(args['pi1'])
wholeData = np.array(wholeData)
pointsAssignCluster = np.zeros((np.size(wholeData, axis=0)))

# print(meanVect)
# print("PI = ", piVect)
dimensions = len(meanVect[0])
covMatVect = covaMat(fileHandle(args['cov1']), clusters, dimensions)
# print("COV = ", covMatVect)

# print(wholeData)
cnt = 0
for xx in wholeData:
    # print("xx = ", xx)
    # print("\n")
    pointsAssignCluster[cnt] = gammaAllot(xx, covMatVect, meanVect, piVect, clusters)
    cnt += 1

print(pointsAssignCluster)
gp.plotClustersAndMean(args['output']+"1"+"_", wholeData, clusters, pointsAssignCluster, np.array(meanVect), "GMM")


