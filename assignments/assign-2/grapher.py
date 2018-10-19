import numpy as np
import matplotlib.pyplot as plt

def plotHistogram(values, bins, type, name):
    graphName = "Bag of Visual Words representation for "+name

    plt.hist(values, bins, facecolor='green')
    plt.xlabel("Visual Words")
    plt.ylabel("Number of patches")
    plt.title(graphName)
    plt.show()


def plotClustersAndMean(wholeData, numOfClusters, pointsAssignCluster, meanVector, plotName="My beautiful  Cat",flag=False):
    xFeature = []
    yFeature = []

    print("Grapher:")
    print("No of points: ",len(wholeData))
    print("no of clusters: ", numOfClusters)
    print("points assigned1: ", len(pointsAssignCluster))
    print("mean vector: ", meanVector)
    plt.ion()
    # if flag:
    #     plt.figure()

    for i in range(numOfClusters):
        xFeature.append([])
        yFeature.append([])
    # plt.figure()
    for i in range(len(wholeData)):
        # print("index val: ", int(pointsAssignCluster[i]), " : size: ",len(xFeature))
        xFeature[int(pointsAssignCluster[i])].append(wholeData[i, 0])
        yFeature[int(pointsAssignCluster[i])].append(wholeData[i, 1])

    for k in range(numOfClusters):
        plt.scatter(xFeature[k], yFeature[k], marker='.', s=1)
        plt.scatter([meanVector[k, 0]], [meanVector[k, 1]], marker='*')
        plt.title(plotName)
    plt.draw()
    plt.pause(0.5)
    plt.clf()
    return
