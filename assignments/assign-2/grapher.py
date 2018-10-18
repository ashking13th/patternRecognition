import numpy as np
import matplotlib.pyplot as plt

def plotHistogram(values, bins, type, name):
    graphName = "Bag of Visual Words representation for "+name

    plt.hist(values, bins, facecolor='green')
    plt.xlabel("Visual Words")
    plt.ylabel("Number of patches")
    plt.title(graphName)
    plt.show()


def plotClustersAndMean(clusterFeatures, meanVect):
    
