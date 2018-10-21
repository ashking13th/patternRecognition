import numpy as np
import matplotlib.pyplot as plt
import os, argparse, math, random
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.interpolate import spline
import matplotlib.pyplot as plt
import gmm

# import grapher as gp
import errno

''' 
	Select k distinct random vectors from the features dataset for each (class(combining all images data))
	Now, for each image we want to have a k(32) dimensional bag of visual words
'''
start_time = datetime.now()

numOfClusters = 1

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="Raw data set location")
ap.add_argument("-n", "--name", required=True, help="name data set location")

args = vars(ap.parse_args())

cntForFile = 0
wholeData = []
lengthOfFile = []

clusterSize = np.zeros((numOfClusters))

#handling of files
def fileHandle(fileName):
	file = open(fileName)
	for line in file:
		teLine = line.rstrip('\n ').split(' ')
		nLine = [float(i) for i in teLine]
		nLine = np.array(nLine)
		wholeData.append(nLine)

	file.close()
	return

def makeGraph(values, graphName, title):
	y = range(1,len(values)+1)
	# xnew = np.linspace(1,len(values),300) #300 represents number of points to make between T.min and T.max
	# power_smooth = spline(y,values,xnew)

	# plt.plot(xnew,power_smooth)
	# plt.show()
	plt.plot(y, values)
	plt.xlabel("No. of iterations")
	plt.ylabel("loglikelihood")
	plt.title(title)
	plt.savefig(graphName+".jpg")
	plt.close()

#euclidean distance calculation
def euclidDist(centroid, dataPt):
	ldist = centroid-dataPt
	return np.sum(np.transpose(ldist)*ldist)

#calculating distance matrix for a point
def distArray(dataPt):
	distVector = np.zeros((numOfClusters), dtype=np.float64)
	for ind in range(numOfClusters):
		distVector[ind] = euclidDist(meanVector[ind], dataPt)
	return distVector

#assignment of clusters
def assignDataPt():
	costFunc = 0.0
	cnt = 0
	for dataPt in wholeData:
		distVector = distArray(dataPt)
		noCl = np.argmin(distVector)	
		pointsAssignCluster[cnt] = noCl
		clusters[noCl] += dataPt
		clusterSize[noCl] += 1
		costFunc += distVector[noCl]
		cnt += 1
	return costFunc

def reCalcMean():
	for i in range(numOfClusters):
		meanVector[i] = clusters[i]/clusterSize[i]
		clusters[i] = 0
		clusterSize[i] = 0


# def allUnique(x):
# 	seen = list()
# 	return not any(i in seen or seen.append(i) for i in x)


print("Process start")
for root, dirs, files in os.walk(args["source"]):
	for f in files:
		path = os.path.relpath(os.path.join(root, f), ".")
		fileHandle(path)
		lengthOfFile.append(len(wholeData)-cntForFile)
		cntForFile = len(wholeData)

wholeData = np.array(wholeData)
meanVector = []
clusters = np.zeros((numOfClusters, np.size(wholeData, axis=1)),dtype=np.float64)

def initMean():
	global meanVector
	temp = random.sample(range(len(wholeData)-1), numOfClusters)
	for assign in temp:
		meanVector.append(wholeData[assign])
	meanVector = np.array(meanVector, dtype=np.float64)

	
#Now, k-means clustering
J = 0.0					#present Cost function
Jprev = -1.0			#previous Cost function
threshold = 1e-3

# kmeans = KMeans(n_clusters=numOfClusters, init='random', n_init=10, max_iter=300, tol=0.001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm='full').fit(wholeData)
# print("number of iterations = ", kmeans.n_iter_)
# print("cost of tour = ", kmeans.inertia_)




initMean()
pointsAssignCluster = np.zeros((np.size(wholeData,axis=0)))
# print(wholeData)

counter = 0
while True:
	a = datetime.now()
	J = assignDataPt()
	# print(counter," : J: ", J, "\t : ",(Jprev-J)," : ",(datetime.now()-loopStarttime))
	# meanVector, " : ", 
	# print(counter, " : Cost = ", (Jprev-J))
	counter += 1
	if Jprev != -1 and Jprev - J < threshold:
		reCalcMean()
		break
	Jprev = J
	reCalcMean()
	b = datetime.now()
	# print(b-a)


# apniList = []

GMM = gmm.master(threshold, len(wholeData), wholeData, len(wholeData[0]), numOfClusters, meanVector, pointsAssignCluster)
	# apniList.append(GMM)
makeGraph(GMM, args["name"] + "_"+ str(numOfClusters), "Iterations vs loglikelihood")
# kabTak += 1

# lengthOfFile = np.array(lengthOfFile)
# BOVW = np.zeros((len(lengthOfFile), numOfClusters))

# cnt = 0
# for i, lenFile in enumerate(lengthOfFile):
# 	for ind in range(lenFile):
# 	 	j = pointsAssignCluster[cnt + ind]
# 	 	BOVW[i, int(j)] += 1 
# 	cnt += lenFile

# # print("Bag of visual words")

# print(BOVW)
# print("Poiutsv in cleusrf = ", pointsAssignCluster)
# print("Final mean Vector = ", meanVector)
# # print("cluster centers = ", kmeans.cluster_centers_)
# GMM = GaussianMixture(n_components=numOfClusters, covariance_type='diag', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans', weights_init=None, means_init=None, precisions_init=None, random_state=None, warm_start=False, verbose=0, verbose_interval=10).fit(wholeData)
# gmm.master(threshold, len(wholeData), wholeData, len(wholeData[0]), numOfClusters, meanVector, pointsAssignCluster)
# print("lower bound on GMM = ", GMM.lower_bound_)