import numpy as np
import matplotlib.pyplot as plt 
import random

numOfClusters =0
def kmeanIntialize(ipData,ncluster,nIteration):
    global numOfClusters
    numOfClusters=ncluster
    meanVector=initMean(ipData)
    
    for i in range(nIteration-1):
        z,meanVector=assignDataPt(ipData,meanVector)
    z,meanVector=assignDataPt(ipData,meanVector)
    return z,meanVector


def euclidDist(centroid, dataPt):
    ldist = centroid-dataPt
    return np.sum(ldist*ldist)

#calculating distance matrix for a point
def distArray(dataPt,meanVector):
    distVector = np.zeros((numOfClusters), dtype=np.float64)
    for ind in range(numOfClusters):
        distVector[ind] = euclidDist(meanVector[ind], dataPt)
    return distVector

#assignment of clusters
def assignDataPt(wholeData,meanVector):
    cnt = 0
    clusterSize=np.zeros((numOfClusters),int)
    clusters = np.zeros((numOfClusters, np.size(wholeData, axis=1)), float)
    z=np.zeros((len(wholeData),numOfClusters), float)
    for dataPt in wholeData:
        distVector = distArray(dataPt,meanVector)
        noCl = np.argmin(distVector)    
        # pointsAssignCluster[cnt] = noCl
        # print("\n noCl = ", noCl, "\ncluster = ", clusters[noCl], "\n")
        clusters[noCl] += dataPt
        # print("data = ", dataPt)
        # print("\n noCl = ", noCl, "\ncluster = ", clusters[noCl], "\n")
        clusterSize[noCl] += 1
        z[cnt][noCl]=1
        # costFunc += distVector[noCl]
        cnt += 1
    meanV=meanVector
    for i in range(numOfClusters):
        if(clusterSize[i]!=0):
            meanV[i]=clusters[i]/clusterSize[i]
    return z,meanV


def initMean(wholeData):
    # global meanVector
    meanVector = []
    temp = random.sample(range(len(wholeData)-1), numOfClusters)
    for assign in temp:
        meanVector.append(wholeData[assign])
    meanVector = np.array(meanVector, dtype=np.float64)
    return meanVector