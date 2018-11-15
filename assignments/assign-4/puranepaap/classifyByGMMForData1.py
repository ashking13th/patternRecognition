import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import math
import random
from datetime import datetime
import matplotlib.patches as mpatches

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="Raw data set location")
ap.add_argument("-d", "--dest", required=False, help="destination location")

ap.add_argument("-m1", "--mean1", required=True, help="Mean 1 location")
ap.add_argument("-m2", "--mean2", required=True, help="mean 2 location")
ap.add_argument("-m3", "--mean3", required=True, help="mean 3 location")

ap.add_argument("-p1", "--pi1", required=True, help="Pi 1 location")
ap.add_argument("-p2", "--pi2", required=True, help="Pi 2 location")
ap.add_argument("-p3", "--pi3", required=True, help="Pi 3 location")

ap.add_argument("-c1", "--cov1", required=True, help="covMat 1 location")
ap.add_argument("-c2", "--cov2", required=True, help="covMat 2 location")
ap.add_argument("-c3", "--cov3", required=True, help="covMat 3 location")

meanPath1 = "../output/prep/2b/gmm/"

args = vars(ap.parse_args())


def fileHandle(fileName):
    wholeData = []
    file = open(fileName)
    for line in file:
        teLine = line.rstrip('\n ').split(' ')
        nLine = [float(i) for i in teLine]
        nLine = np.array(nLine)
        wholeData.append(nLine)
    file.close()
    return wholeData


def covaMat(mat, clusters, dimensions):
    vect = np.zeros(shape=(clusters, dimensions, dimensions))
    for i in range(clusters):
        for j in range(dimensions):
            vect[i, j, j] = mat[i][j]
    return vect


meanVect = []
piVect = []
covMatVect = []

meanVect.append(fileHandle(args['mean1']))
piVect.append(fileHandle(args['pi1'])[0])
clusters = len(meanVect[0])
dimensions = len(meanVect[0][0])
covMatVect.append(covaMat(fileHandle(args['cov1']), clusters, dimensions))

meanVect.append(fileHandle(args['mean2']))
piVect.append(fileHandle(args['pi2'])[0])
covMatVect.append(covaMat(fileHandle(args['cov2']), clusters, dimensions))

meanVect.append(fileHandle(args['mean3']))
piVect.append(fileHandle(args['pi3'])[0])
covMatVect.append(covaMat(fileHandle(args['cov3']), clusters, dimensions))

mainList = []


def fileHandle2(fileName):
    file = open(fileName)
    tempList = []
    for line in file:
        teLine = line.rstrip('\n ').split(' ')
        nLine = [float(i) for i in teLine]
        tempList.append(teLine)
    file.close()
    x = np.array(tempList,float)
    return x
    
nClass = 3

def gaussian(covMat, x, mean):
    numFeature = np.size(mean)
    gaussian = -(1/2)*np.sum((np.transpose(x-mean) * (np.linalg.inv(covMat)))*(x-mean))
    gaussian = np.exp(gaussian)
    deter = np.linalg.det(covMat)
    gaussian *= deter**(-1./2)
    gaussian *= (2*np.pi)**(-numFeature/2.)
    return gaussian

def allotClass(x, nClass, clusters, covMatVect, meanVect, piVect):
    likelihood = np.zeros((nClass))
    for numC in range(nClass):
        for k in range(clusters):
            likelihood[numC] += piVect[numC][k] * gaussian(covMatVect[numC][k], x, meanVect[numC][k])
    ans = np.argmax(likelihood)
    # print("likelihood: ",likelihood)
    # print("Ans: ",ans)
    # print(ans)
    # return likelihood # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return ans

def gammaAllot(x, covMatVect, meanVector, piVect, clusters):
    gammaVect = np.zeros((clusters))
    sum = 0
    gaussians = np.zeros((clusters))
    # print("Gamma length: ", len(gammaVect))
    # print("Pi length: ", len(piVect))
    # print("Mean length: ", len(meanVector))

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

# ############################################### #
# ############################################### #
# ############################################### #

# def classify():
#     pass


data = fileHandle(args["source"])
data = np.array(data)
print(len(data))
result = np.zeros(3)
for point in data:
    # print(point)
    result[allotClass(point, 3, clusters, covMatVect, meanVect, piVect)] += 1

print(result)


# plot(covMatVect, meanVect, args['dest'], piVect)
# classify()
