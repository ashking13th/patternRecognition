import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import math
import random
from datetime import datetime
import matplotlib.patches as mpatches

ap = argparse.ArgumentParser()
# ap.add_argument("-s", "--source", required=True, help="Raw data set location")
# ap.add_argument("-d", "--dest", required=False, help="destination location")

# ap.add_argument("-m1", "--mean1", required=True, help="Mean 1 location")
# ap.add_argument("-m2", "--mean2", required=True, help="mean 2 location")
# ap.add_argument("-m3", "--mean3", required=True, help="mean 3 location")

# ap.add_argument("-p1", "--pi1", required=True, help="Pi 1 location")
# ap.add_argument("-p2", "--pi2", required=True, help="Pi 2 location")
# ap.add_argument("-p3", "--pi3", required=True, help="Pi 3 location")

# ap.add_argument("-c1", "--cov1", required=True, help="covMat 1 location")
# ap.add_argument("-c2", "--cov2", required=True, help="covMat 2 location")
# ap.add_argument("-c3", "--cov3", required=True, help="covMat 3 location")

# ap.add_argument("-l", "--lval", required=True, help="l value")
ap.add_argument("-c", "--clust", required=True, help="clusters value")
# ap.add_argument("-n", "--cnum", required=True, help="class number value")

args = vars(ap.parse_args())

testPath = "../dataset/prep/2b/bovw/test_"
classNames = ["bayou", "chalet", "creek"]

inputPath = "../output/prep/2b/gmm/train_"
# meanPath1 = inputPath + classNames[0] + "_L" + args["lval"] + "_C" + args["clust"] + "_GMM.means"
# meanPath2 = inputPath + classNames[1] + "_L" + args["lval"] + "_C" + args["clust"] + "_GMM.means"
# meanPath3 = inputPath + classNames[2] + "_L" + args["lval"] + "_C" + args["clust"] + "_GMM.means"

# piPath1 = inputPath + classNames[0] + "_L" + args["lval"] + "_C" + args["clust"] + "_GMM.piVect"
# piPath2 = inputPath + classNames[1] + "_L" + args["lval"] + "_C" + args["clust"] + "_GMM.piVect"
# piPath3 = inputPath + classNames[2] + "_L" + args["lval"] + "_C" + args["clust"] + "_GMM.piVect"

# covPath1 = inputPath + classNames[0] + "_L" + args["lval"] + "_C" + args["clust"] + "_GMM.cov"
# covPath2 = inputPath + classNames[1] + "_L" + args["lval"] + "_C" + args["clust"] + "_GMM.cov"
# covPath3 = inputPath + classNames[2] + "_L" + args["lval"] + "_C" + args["clust"] + "_GMM.cov"

meanPath1 = "../output/prep/2b/old/train_" + classNames[0] + "_C" + args["clust"] + "_GMM.means" 
meanPath2 = "../output/prep/2b/old/train_" + classNames[1] + "_C" + args["clust"] + "_GMM.means" 
meanPath3 = "../output/prep/2b/old/train_" + classNames[2] + "_C" + args["clust"] + "_GMM.means" 

piPath1 = "../output/prep/2b/old/train_" + classNames[0] + "_C" + args["clust"] + "_GMM.piVect"
piPath2 = "../output/prep/2b/old/train_" + classNames[1] + "_C" + args["clust"] + "_GMM.piVect"
piPath3 = "../output/prep/2b/old/train_" + classNames[2] + "_C" + args["clust"] + "_GMM.piVect"

covPath1 = "../output/prep/2b/old/train_" + classNames[0] + "_C" + args["clust"] + "_GMM.cov"
covPath2 = "../output/prep/2b/old/train_" + classNames[1] + "_C" + args["clust"] + "_GMM.cov"
covPath3 = "../output/prep/2b/old/train_" + classNames[2] + "_C" + args["clust"] + "_GMM.cov"

meanPath1

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

meanVect.append(fileHandle(meanPath1))
piVect.append(fileHandle(piPath1)[0])
clusters = len(meanVect[0])
dimensions = len(meanVect[0][0])
covMatVect.append(covaMat(fileHandle(covPath1), clusters, dimensions))

meanVect.append(fileHandle(meanPath2))
piVect.append(fileHandle(piPath2)[0])
covMatVect.append(covaMat(fileHandle(covPath2), clusters, dimensions))

meanVect.append(fileHandle(meanPath3))
piVect.append(fileHandle(piPath3)[0])
covMatVect.append(covaMat(fileHandle(covPath3), clusters, dimensions))

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


# print(len(data))


for className in classNames:
    data = fileHandle(testPath + className + ".bovw")
    data = np.array(data)
    result = np.zeros(3)
    for point in data:
        # print(point)
        result[allotClass(point, 3, clusters, covMatVect, meanVect, piVect)] += 1
    print(result)



# plot(covMatVect, meanVect, args['dest'], piVect)
# classify()
