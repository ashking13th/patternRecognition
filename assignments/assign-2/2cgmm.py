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
	file = open(fileName)
	for line in file:
		teLine = line.rstrip('\n ').split(' ')
		nLine = [float(i) for i in teLine]
		nLine = np.array(nLine)
		wholeData.append(nLine)

	file.close()
	return


print("Process start")
for root, dirs, files in os.walk(args["source"]):
	for f in files:
		path = os.path.relpath(os.path.join(root, f), ".")
		# print("read: ",path)
		fileHandle(path)
		# lengthOfFile.append(len(wholeData)-cntForFile)
		# cntForFile = len(wholeData)


def covaMat(mat, clusters, dimensions):
    vect = np.zeros(shape=(clusters, dimensions, dimensions))
    for i in range(clusters):
        for j in range(dimensions):
            vect[i, j, j] = mat[i][j]
    return vect

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


meanVect = []
piVect = []
covMatVect = []



meanVect.append(fileHandle(args['mean1']))
piVect.append(fileHandle(args['pi1']))
wholeData = np.array(wholeData)
pointsAssignCluster = np.zeros((np.size(wholeData, axis=0)))




dimensions = len(meanVect[0][0])
covMatVect.append(covaMat(fileHandle(args['cov1']), clusters, dimensions))

for xx in wholeData:
    pointsAssignCluster[i] = gammaAllot(xx, covMatVect, meanVect, piVect, clusters)

gp.plotClustersAndMean(args['output']+"1"+"_", wholeData, clusters, pointsAssignCluster, meanVect, "GMM")


