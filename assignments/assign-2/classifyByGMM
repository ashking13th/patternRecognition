import numpy as np
import matplotlib.pyplot as plt
import os, argparse, math, random
from datetime import datetime
from sklearn.cluster import KMeans
# from sklearn.mixture import GaussianMixture as gm

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="Raw data set location")
args = vars(ap.parse_args())


nClass = 0
inpData = []

def gaussian(covMat, x, mean):
    numFeature = np.size(mean)
    gaussian = -(1/2)*np.sum((np.transpose(x-mean)*(np.linalg.inv(covMat)))*(x-mean))
    gaussian = np.exp(gaussian)
    deter = np.linalg.det(covMat)
    gaussian *= deter**(-1./2)
    gaussian *= (2*np.pi)**(-numFeature/2.)
    return gaussian

def rpa(matr):
	trueclass=np.zeros(nClass)
	predictedclass=np.zeros(nClass)
	correctpredicted=np.zeros(nClass)
	totalexample=0;
	for i in range(nClass):
		for j in range(nClass):
			trueclass[i]+=matr[i][j]
			predictedclass[i]+=matr[j][i]
			totalexample+=matr[i][j]
			if i==j:
				correctpredicted[i]=matr[i][j];
	accuracy=np.sum(correctpredicted)/totalexample;
	recall=correctpredicted/trueclass;
	precision=correctpredicted/predictedclass;
	print('Accuracy = ',accuracy)
	print('Recall = ')
	for i in recall:
		print(i)
	print('Precision = ')
	for i in precision:
		print(i)

for root, dirs, files in os.walk(args["source"]):
	for f in files:
		path = os.path.relpath(os.path.join(root, f), ".")
		fileHandle(path)
		lengthOfFile.append(len(inpData)-cntForFile)
		cntForFile = len(inpData)

inpData = np.array(inpData)

#Passing the calculated pi-k, mean-k, covariance-k
#here initialise our nClass, inpData, ptsAssign

ptsAssign = np.zeros((nClass))

for ind in len(inpData):
	likelihood = np.zeros((nClass))
	for numC in range(nClass):
		likelihood[numC] = pik[numC] * gaussian(covMatVect[numC], inpData[ind], meanVect[numC])
	ptsAssign[np.argmin(likelihood)] += 1

rpa(ptsAssign)
