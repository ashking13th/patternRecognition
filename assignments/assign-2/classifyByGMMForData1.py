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


def plot(covMat, classname):
	minMax = np.zeros((numFeature,2))
	colors = ['#136906', '#fcbdfc', '#e5ff00', '#ff0000', '#3700ff', '#000000']

	# Resolution affects the time required to process.
	res = 100

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

	dataRange = np.zeros((numFeature))
	for i in range(numFeature):
		dataRange[i] = 0.1*(minMax[i, 1] - minMax[i, 0])

	x = np.linspace(minMax[0, 0] - dataRange[0], minMax[0, 1] + dataRange[0], res)
	y = np.linspace(minMax[1,0] - dataRange[1], minMax[1,1] + dataRange[1], res)

	tellClassNum = np.zeros((np.size(x,0)*np.size(y,0), nClass))

	count = 0
	for j in y:
		for i in x:
			for k in range(nClass):
				dataPt = np.array([i,j])
				tellClassNum[count, k] = discriminant(dataPt, meanVector[k], covMat[k])
			count += 1

	lenX = np.size(x,0)
	Z = np.zeros((nClass, lenX, lenX))

	for k in range(nClass):
		count = 0
		for j in y:
			for i in x:
				dataPt = np.array([i,j])
				fi = int(count/lenX)
				sec = count%lenX
				Z[k, fi, sec] = gaussianDensity(dataPt, meanVector[k], covMat[k])
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
		#plt.savefig(plotname)
		plt.show()
		if idx == 3:
			fig2 = plt.figure(2)
			bx = fig2.gca()

			X,Y = np.meshgrid(x,y)

			bx.plot(mainList[0][:,0],mainList[0][:,1], c = colors[3], marker=".",  linestyle="None", label="Class 1 Data", ms='2')
			bx.plot(mainList[1][:,0],mainList[1][:,1], c = colors[4], marker=".",  linestyle="None", label="Class 2 Data", ms='2')
			bx.plot(mainList[2][:,0],mainList[2][:,1], c = colors[5], marker=".",  linestyle="None", label="Class 3 Data", ms='2')
			class_colours = [colors[3], colors[4], colors[5], colors[0], "#824003", "#cc00ff"]
			classes = ["Class 1 Data", "Class 2 Data", "Class 3 Data", "Class 1 Contours", "Class 2 Contours", "Class 3 Contours"]

			bx.contour(X, Y, Z[0], alpha=1, linewidth=10, colors=colors[0], label="Class 1 Contour")
			bx.contour(X, Y, Z[1], alpha=1, linewidth=10, colors="#824003", label="Class 2 Contour")
			bx.contour(X, Y, Z[2], alpha=1, linewidth=10, colors="#cc00ff", label="Class 3 Contour")

			recs = []
			for i in range(0, len(class_colours)):
				recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=class_colours[i]))
			plotname = classname+"_contours.png"

			plt.legend(recs, classes, loc='upper right')
			plt.title("Contours with training data")
			plt.xlabel('X')
			plt.ylabel('Y')
			#plt.savefig(plotname)
			plt.show()


imgAssign = np.zeros((nClass))

for root, dirs, files in os.walk(args["source"]):
    for f in files:
        path = os.path.relpath(os.path.join(root, f), ".")
        print("read = ", path)
        fileHandle2(path)
        inpData = np.array(inpData)

        ptAssigned = np.zeros((nClass))
        for ind in range(len(inpData)):
            
            likelihood = np.zeros((nClass))
            for numC in range(nClass):
                for k in range(clusters):
                    likelihood[numC] += piVect[numC][k] * gaussian(covMatVect[numC][k], inpData[ind], meanVect[numC][k])
            ptAssigned[np.argmax(likelihood)] += 1
        inpData = []
        

print(ptAssigned)
