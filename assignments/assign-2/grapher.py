import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plotHistogram(values, bins, type, name):
    graphName = "Bag of Visual Words representation for "+name

    plt.hist(values, bins, facecolor='green')
    plt.xlabel("Visual Words")
    plt.ylabel("Number of patches")
    plt.title(graphName)
    plt.show()


def likelihoodPlotter(values, graphName, title):
    y = range(1,len(values)+1)
    plt.plot(values, y)
    plt.xlabel("No. of")
    plt.ylabel("")
    plt.title(title)
    plt.savefig(graphName+".jpg")
    


def plotClustersAndMean(outputPath, wholeData, numOfClusters, pointsAssignCluster, meanVector, plotName="My beautiful  Cat",flag=False):
    xFeature = []
    yFeature = []
    colors = ['#136906', '#fcbdfc', '#e5ff00', '#ff0000', '#3700ff', '#000000']
    plt.xlabel('Mean')
    plt.ylabel('Dispersion')

    class_colours = ['green', 'red', 'blue']
    class_colours1 = ['green', 'yellow', 'blue']
    classes = ["Cluster1", "Cluster2", "Cluster3"]

    recs = []
    for i in range(0, len(class_colours)):
        recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=class_colours[i]))
    plt.legend(recs, classes, loc='upper right')

    print("Grapher:")
    # print("No of points: ",len(wholeData))
    # print("no of clusters: ", numOfClusters)
    # print("points assigned1: ", len(pointsAssignCluster))
    # print("mean vector: ", meanVector)
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
        plt.scatter(xFeature[k], yFeature[k], marker='.', s=1, c=class_colours[k])
        plt.scatter([meanVector[k, 0]], [meanVector[k, 1]], marker='*', c=class_colours1[3-k-1])
        plt.title(plotName)
    plt.draw()
    plt.savefig(outputPath+".jpg")
    plt.pause(0.5)
    plt.clf()
    return
