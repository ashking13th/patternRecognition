import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import math
import random
from datetime import datetime
from sklearn.cluster import KMeans
import matplotlib.patches as mpatches
# from sklearn.mixture import GaussianMixture as gm

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="Raw data set location")
ap.add_argument("-d", "--dest", required=True, help="destination location")

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
    # print(ans)
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


def plot(covMatVect, meanVector, name, piVect, classname="1"):
    # mainList = mainList
    nClass = 3
    numFeature = 2
    minMax = np.zeros((numFeature,2))
    colors = ['#136906', '#fcbdfc', '#e5ff00', '#ff0000', '#3700ff', '#000000']

	# Resolution affects the time required to process.
    res = 100

    print("mainList size: ",len(mainList))
    # print("mainList[0].shape(): ",mainList[0].shape())

    count = 0
    for i in range(nClass):
        for j in range(numFeature):
            if count == 0:
                minMax[j, 0] = np.ceil(np.amin(mainList[i][:, j]))
                minMax[j, 1] = np.ceil(np.amax(mainList[i][:, j]))
            else:				
                minMax[j, 0] = min(minMax[j,0], np.ceil(np.amin(mainList[i][:, j])))
                minMax[j, 1] = max(minMax[j,1], np.ceil(np.amax(mainList[i][:, j])))
            count = 1

    print("Found MinMax")
    print(minMax)

    dataRange = np.zeros((numFeature))
    for i in range(numFeature):
        dataRange[i] = 0.1*(minMax[i, 1] - minMax[i, 0])

    x = np.linspace(minMax[0, 0] - dataRange[0], minMax[0, 1] + dataRange[0], res)
    y = np.linspace(minMax[1,0] - dataRange[1], minMax[1,1] + dataRange[1], res)

    tellClassNum = np.zeros((np.size(x,0)*np.size(y,0), nClass))

    allotment = np.zeros(3)
    count = 0
    for j in y:
        for i in x:
            for k in range(nClass):
                dataPt = np.array([i,j])
                # a = gammaAllot(dataPt, covMatVect[k], meanVect[k], piVect[k], clusters)
                a = allotClass(dataPt, nClass, clusters, covMatVect, meanVect, piVect)
                tellClassNum[count, k] = a
                allotment[a] += 1
            count += 1
    print("Allotment: ", allotment)
    lenX = np.size(x,0)
    Z = np.zeros((nClass, clusters, lenX, lenX))

    for k in range(nClass):
        for cl in range(clusters):
            count = 0
            for j in y:
                for i in x:
                    dataPt = np.array([i,j])
                    fi = int(count/lenX)
                    sec = count%lenX
                    Z[k, cl, fi, sec] = gaussian(covMatVect[k][cl], dataPt, meanVect[k][cl])
                    count += 1	

    count = 0
    for idx in range(nClass+1):
        fig1 = plt.figure(1)
        ax = fig1.gca()

        plotClass = []
        for cl in range(nClass):
            temp1 = []
            for fe in range(numFeature):
                temp1.append([])
            plotClass.append(temp1)

        class_colours = []
        classes = []
        count = 0
        for j in y:
            for i in x:	
                tempArr = np.argsort(tellClassNum[count, :])
                count += 1
                classNum = tempArr[-2] if tempArr[-1] == idx else tempArr[-1]
                plotClass[classNum][0].append(i)
                plotClass[classNum][1].append(j)

        plotname = "plot.png"
		
        if idx==0:
            ax.plot(plotClass[1][0], plotClass[1][1], c=colors[1],marker=".",  linestyle="None", label="Class 2 Prediction")
            ax.plot(plotClass[2][0], plotClass[2][1], c=colors[2],marker=".",  linestyle="None", label="Class 3 Prediction")
            ax.plot(mainList[1][:,0], mainList[1][:,1], c=colors[4], marker=".",  linestyle="None", label="Class 2 Data", ms='2')
            ax.plot(mainList[2][:,0], mainList[2][:,1], c=colors[5], marker=".",  linestyle="None", label="Class 3 Data", ms='2')
            class_colours = [colors[4], colors[5], colors[1], colors[2]]
            classes = ["Class 2 Data", "Class 3 Data", "Class 2 Prediction", "Class 3 Prediction"]
            plotname = "23"+plotname
        elif idx==1:
            ax.plot(plotClass[0][0], plotClass[0][1], c=colors[0],marker=".", linestyle="None", label="Class 1 Prediction")
            ax.plot(plotClass[2][0], plotClass[2][1], c=colors[2],marker=".", linestyle="None", label="Class 3 Prediction")
            ax.plot(mainList[0][:,0], mainList[0][:, 1], c=colors[3], marker=".", linestyle="None", label="Class 1 Data", ms='2')
            ax.plot(mainList[2][:,0],mainList[2][:,1], c=colors[5], marker=".", linestyle="None", label="Class 3 Data", ms='2')
            class_colours = [colors[5], colors[3], colors[0], colors[2]]
            classes = ["Class 1 Data", "Class 3 Data", "Class 1 Prediction", "Class 3 Prediction"]
            plotname = "13" + plotname
        elif idx==2:
            ax.plot(plotClass[0][0], plotClass[0][1], c = colors[0],marker=".",  linestyle="None", label="Class 1 Prediction")
            ax.plot(plotClass[1][0], plotClass[1][1], c = colors[1],marker=".",  linestyle="None", label="Class 2 Prediction")
            ax.plot(mainList[0][:,0],mainList[0][:,1], c=colors[3], marker=".",  linestyle="None", label="Class 1 Data", ms='2')
            ax.plot(mainList[1][:,0],mainList[1][:,1], c = colors[4], marker=".",  linestyle="None", label="Class 2 Data", ms='2')
            class_colours = [colors[3], colors[4], colors[0], colors[1]]
            classes = ["Class 1 Data", "Class 2 Data", "Class 1 Prediction", "Class 2 Prediction"]
            plotname = "12" + plotname
        else:
            ax.plot(plotClass[0][0], plotClass[0][1], c = colors[0],marker=".",  linestyle="None", label="Class 1 Prediction")
            ax.plot(plotClass[1][0], plotClass[1][1], c = colors[1],marker=".",  linestyle="None", label="Class 2 Prediction")
            ax.plot(plotClass[2][0], plotClass[2][1], c=colors[2],marker=".",  linestyle="None", label="Class 3 Prediction")
            ax.plot(mainList[0][:,0],mainList[0][:,1], c = colors[3], marker=".",  linestyle="None", label="Class 1 Data", ms='2')
            ax.plot(mainList[1][:,0],mainList[1][:,1], c = colors[4], marker=".",  linestyle="None", label="Class 2 Data", ms='2')
            ax.plot(mainList[2][:,0],mainList[2][:,1], c = colors[5], marker=".",  linestyle="None", label="Class 3 Data", ms='2')
            class_colours = [colors[3], colors[4], colors[5], colors[0], colors[1], colors[2]]
            classes = ["Class 1 Data", "Class 2 Data", "Class 3 Data", "Class 1 Prediction", "Class 2 Prediction", "Class 3 Prediction"]
            plotname = "123" + plotname
        ax.patch.set_visible(False)
        if idx == 0:
            plt.title("Class 2 vs Class 3")
        elif idx == 1:
            plt.title("Class 1 vs Class 3")
        elif idx == 2:
            plt.title("Class 1 vs Class 2")
        else:
            plt.title("Class 1 - Class 2 - Class 3")
        plotname = classname + plotname
        plt.xlabel('X')
        plt.ylabel('Y')
        recs = []
        for i in range(0,len(class_colours)):
            recs.append(mpatches.Rectangle((0,0),1,1,fc=class_colours[i]))
        plt.legend(recs, classes, loc='upper right')
        plotname = name + plotname
        plt.savefig(plotname)
        # plt.show()
        if idx == 3:
            fig2 = plt.figure(2)
            bx = fig2.gca()

            X,Y = np.meshgrid(x,y)

            bx.plot(mainList[0][:,0],mainList[0][:,1], c = colors[3], marker=".",  linestyle="None", label="Class 1 Data", ms='2')
            bx.plot(mainList[1][:,0],mainList[1][:,1], c = colors[4], marker=".",  linestyle="None", label="Class 2 Data", ms='2')
            bx.plot(mainList[2][:,0],mainList[2][:,1], c = colors[5], marker=".",  linestyle="None", label="Class 3 Data", ms='2')
            class_colours = [colors[3], colors[4], colors[5], colors[0], "#824003", "#cc00ff"]
            classes = ["Class 1 Data", "Class 2 Data", "Class 3 Data", "Class 1 Contours", "Class 2 Contours", "Class 3 Contours"]

            bx.contour(X, Y, Z[0][0], alpha=1, linewidth=10, colors=colors[0], label="Class 1 Contour")
            bx.contour(X, Y, Z[1][0], alpha=1, linewidth=10, colors="#824003", label="Class 2 Contour")
            bx.contour(X, Y, Z[2][0], alpha=1, linewidth=10, colors="#cc00ff", label="Class 3 Contour")

            recs = []
            for i in range(0, len(class_colours)):
                recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=class_colours[i]))
            plotname = classname+"_contours.png"

            plt.legend(recs, classes, loc='upper right')
            plt.title("Contours with training data")
            plt.xlabel('X')
            plt.ylabel('Y')
            plotname = name + plotname
            plt.savefig(plotname)
            # plt.show()


# imgAssign = np.zeros((nClass))

for root, dirs, files in os.walk(args["source"]):
    for f in files:
        path = os.path.relpath(os.path.join(root, f), ".")
        print("read = ", path)
        mainList.append(fileHandle2(path))
        # mainList = np.array(mainList)
# mainList = np.array(mainList)
# print(mainList.shape())
plot(covMatVect, meanVect, args['dest'], piVect)

# print(ptAssigned)
#
