import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import math
import random
from datetime import datetime
from sklearn.cluster import KMeans
# from sklearn.mixture import GaussianMixture as gm

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="Raw data set location")

ap.add_argument("-m1", "--mean1", required=True, help="Mean 1 location")
ap.add_argument("-m2", "--mean2", required=True, help="mean 2 location")
ap.add_argument("-m3", "--mean3", required=True, help="mean 3 location")

ap.add_argument("-p1", "--pi1", required=True, help="Mean 1 location")
ap.add_argument("-p2", "--pi2", required=True, help="mean 2 location")
ap.add_argument("-p3", "--pi3", required=True, help="mean 3 location")

ap.add_argument("-c1", "--cov1", required=True, help="Mean 1 location")
ap.add_argument("-c2", "--cov2", required=True, help="mean 2 location")
ap.add_argument("-c3", "--cov3", required=True, help="mean 3 location")

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
# clusters = len(meanVector[0])
# dimensions = len(meanVector[0][0])
covMatVect.append(covaMat(fileHandle(args['cov2']), clusters, dimensions))

meanVect.append(fileHandle(args['mean3']))
piVect.append(fileHandle(args['pi3'])[0])
# clusters = len(meanVector[0])
# dimensions = len(meanVector[0][0])
covMatVect.append(covaMat(fileHandle(args['cov3']), clusters, dimensions))

inpData = []


def fileHandle2(fileName):
    # wholeData = []
    file = open(fileName)
    for line in file:
        teLine = line.rstrip('\n ').split(' ')
        nLine = [float(i) for i in teLine]
        nLine = np.array(nLine)
        inpData.append(nLine)
    file.close()
    # return wholeData


nClass = 3


# def rpa(matr):
# 	trueclass = np.zeros(nClass)
# 	predictedclass = np.zeros(nClass)
# 	correctpredicted = np.zeros(nClass)
# 	totalexample = 0
# 	for i in range(nClass):
# 		for j in range(nClass):
# 			trueclass[i] += matr[i][j]
# 			predictedclass[i] += matr[j][i]
# 			totalexample += matr[i][j]
# 			if i == j:
# 				correctpredicted[i]=matr[i][j]
# 	accuracy=np.sum(correctpredicted)/totalexample
# 	recall=correctpredicted/trueclass
# 	precision=correctpredicted/predictedclass
# 	print('Accuracy = ',accuracy)
# 	print('Recall = ')
# 	for i in recall:
# 		print(i)
# 	print('Precision = ')
# 	for i in precision:
# 		print(i)


def gaussian(covMat, x, mean):
    numFeature = np.size(mean)
    gaussian = -(1/2)*np.sum((np.transpose(x-mean)
                              * (np.linalg.inv(covMat)))*(x-mean))
    gaussian = np.exp(gaussian)
    deter = np.linalg.det(covMat)
    gaussian *= deter**(-1./2)
    gaussian *= (2*np.pi)**(-numFeature/2.)
    return gaussian

# def rpa(matr):
# 	trueclass = np.zeros(nClass)
# 	predictedclass = np.zeros(nClass)
# 	correctpredicted = np.zeros(nClass)
# 	totalexample = 0
# 	for i in range(nClass):
# 		for j in range(nClass):
# 			trueclass[i] += matr[i][j]
# 			predictedclass[i] += matr[j][i]
# 			totalexample += matr[i][j]
# 			if i == j:
# 				correctpredicted[i]=matr[i][j]
# 	accuracy=np.sum(correctpredicted)/totalexample
# 	recall=correctpredicted/trueclass
# 	precision=correctpredicted/predictedclass
# 	print('Accuracy = ',accuracy)
# 	print('Recall = ')
# 	for i in recall:
# 		print(i)
# 	print('Precision = ')
# 	for i in precision:
# 		print(i)

numFeature = 2
imgAssign = np.zeros((nClass))

minMax = np.zeros((numFeature, 2))
colors = ['#136906', '#fcbdfc', '#e5ff00', '#ff0000', '#3700ff', '#000000']

# Resolution affects the time required to process.
res = 100

# for root, dirs, files in os.walk(args["source"]):
#     for f in files:
#         path = os.path.relpath(os.path.join(root, f), ".")
#         print("read = ", path)
#         fileHandle2(path)
#         inpData = np.array(inpData)
        

    #     count = 0
    #     for i in range(nClass):
    #         for j in range(numFeature):
    #             if count == 0:
    #                 minMax[j, 0] = np.ceil(np.amin(mainList[i][:, j]))
    #                 minMax[j, 1] = np.ceil(np.amax(mainList[i][:, j]))
    #             else:				
    #                 minMax[j, 0] = min(minMax[j,0], np.ceil(np.amin(mainList[i][:, j])))
    #                 minMax[j, 1] = max(minMax[j,1], np.ceil(np.amax(mainList[i][:, j])))
    #             count = 1

    # print("Found MinMax")
    # print(minMax)

    # dataRange = np.zeros((numFeature))
    # for i in range(numFeature):
    #     dataRange[i] = 0.1*(minMax[i, 1] - minMax[i, 0])

x = np.linspace(-6, 6, res)
y = np.linspace(-6, 6, res)

xf = []
yf = []

for i in range(3):
    xf.append([])
    yf.append([])

for i in x:
    for j in y:
        # ptAssigned = np.zeros((nClass))
        # for ind in range(len(inpData)):

        likelihood = np.zeros((nClass))
        for numC in range(nClass):
            for k in range(clusters):
                likelihood[numC] += piVect[numC][k] * gaussian(covMatVect[numC][k], [i,j], meanVect[numC][k])
        xf[np.argmax(likelihood)].append(i)
        yf[np.argmax(likelihood)].append(j)
        # ptAssigned[np.argmax(likelihood)] += 1
    # inpData = []

plt.plot(xf[0], yf[0], c='blue',  linestyle="None", marker=".")
plt.plot(xf[1], yf[1], c='red',  linestyle="None", marker=".")
plt.plot(xf[2], yf[2], c='green',  linestyle="None", marker=".")

plt.show()
print("!!!!!!!!!!!!!!!!!!!!!!! Done !!!!!!!!!!!!!!!!!!!!!!!!!!!")
