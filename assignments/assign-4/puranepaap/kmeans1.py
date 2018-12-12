import numpy as np
import matplotlib.pyplot as plt
import os, argparse, math, random
# import gmm
# import gmm2
# import grapher as gp
import gmm4
import errno

''' 
	Select k distinct random vectors from the features dataset for each (class(combining all images data))
	Now, for each image we want to have a k(32) dimensional bag of visual words
'''
start_time = datetime.now()

sourcePath = "../dataset/prep/2b/pca/train"
targetP = "../output/prep/2b/gmm/train"
classNames = ["bayou", "chalet", "creek"]

numOfClusters = 64

ap = argparse.ArgumentParser()
# ap.add_argument("-s", "--source", required=True, help="Raw data set location")
# ap.add_argument("-o", "--output", required=True, help="Output mean location")
ap.add_argument("-c", "--clusters",required=True,help="No of clusters")
ap.add_argument("-n", "--classNum", required=True, help="No of clusters")
ap.add_argument("-l", "--lval", required=True, help="No of clusters")
args = vars(ap.parse_args())

numOfClusters = int(args["clusters"])

sourceFile = sourcePath + "_" + classNames[int(args["classNum"])] + "_" + args["lval"] + ".pca"
targetFile = targetP + "_" + classNames[int(args["classNum"])] + "_L" + args["lval"] + "_C"

cntForFile = 0
wholeData = []
lengthOfFile = []

clusterSize = np.zeros((numOfClusters))
print("Starting up")

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
		# print("\n noCl = ", noCl, "\ncluster = ", clusters[noCl], "\n")
		clusters[noCl] += dataPt
		# print("data = ", dataPt)
		# print("\n noCl = ", noCl, "\ncluster = ", clusters[noCl], "\n")
		clusterSize[noCl] += 1
		costFunc += distVector[noCl]
		cnt += 1
	return costFunc

def reCalcMean():
	for i in range(numOfClusters):
		# print("Cluster Size[i]: ",clusterSize[i])
		# if(clusterSize[i]==0):
		# 	print("Reinitializing mean :( ")
		# 	initMean()
		# 	return
		meanVector[i] = clusters[i]/clusterSize[i]
		clusters[i] = 0
		clusterSize[i] = 0


# def allUnique(x):
# 	seen = list()
# 	return not any(i in seen or seen.append(i) for i in x)

#	@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@	#
# print("Process start")
# for root, dirs, files in os.walk(args["source"]):
# 	for f in files:
# 		path = os.path.relpath(os.path.join(root, f), ".")
# 		# print("read: ",path)
# 		fileHandle(path)
# 		lengthOfFile.append(len(wholeData)-cntForFile)
# 		cntForFile = len(wholeData)
#	@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@	#


fileHandle(sourceFile)

wholeData = np.array(wholeData)
meanVector = []
clusters = np.zeros((numOfClusters, np.size(wholeData, axis=1)),dtype=np.float64)

def initMean():
	global meanVector
	meanVector = []
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
# # print(wholeData)
loopStarttime = datetime.now()
counter = 0
while True:
	a = datetime.now()
	J = assignDataPt()
	print(counter," : J: ", J, "\t : ",(Jprev-J)," : ",(datetime.now()-loopStarttime))
	# meanVector, " : ", 
	# print(counter, " : Cost = ", (Jprev-J))
	counter += 1
	# if len(wholeData[0]) < 3:
	# 	gp.plotClustersAndMean(args['output']+str(counter)+"_",wholeData, numOfClusters, pointsAssignCluster, meanVector,"K-Means")
	if Jprev != -1 and Jprev - J < threshold:
		reCalcMean()
		break
	Jprev = J
	reCalcMean()
	b = datetime.now()
	# print(b-a)


# lengthOfFile = np.array(lengthOfFile)
# BOVW = np.zeros((len(lengthOfFile), numOfClusters))

# cnt = 0
# for i, lenFile in enumerate(lengthOfFile):
# 	for ind in range(lenFile):
# 	 	j = pointsAssignCluster[cnt + ind]
# 	 	BOVW[i, int(j)] += 1 
# 	cnt += lenFile

# print("Bag of visual words")
# if numOfClusters> 0:
# 	print(BOVW)
# 	targetPath = args['output']+str(numOfClusters)+".BOW"
# 	if not os.path.exists(os.path.dirname(args['output'])):
# 			try:
# 				os.makedirs(os.path.dirname(args['output']))
# 			except OSError as exc:  # Guard against race condition
# 				if exc.errno != errno.EEXIST:
# 					raise
# 	try:
# 		print("target File: ", targetPath)
# 		outfile = open(targetPath, "w")
# 	except IOError:
# 		print("File not created !!!!!!!!!!!!!!!!!!!!!!!!!")

# 	for bag in BOVW:
# 		for a in bag:
# 			outfile.write(str(a)+" ")
# 		outfile.write("\n")
# 	outfile.close()
# # print("Final mean Vector = ", meanVector)

# targetPath = args['output']+".kmeans"
# if not os.path.exists(os.path.dirname(args['output'])):
#         try:
#             os.makedirs(os.path.dirname(args['output']))
#         except OSError as exc:  # Guard against race condition
#             if exc.errno != errno.EEXIST:
#                 raise
# try:
# 	print("target File: ", targetPath)
# 	outfile = open(targetPath, "w")
# except IOError:
# 	print("File not created !!!!!!!!!!!!!!!!!!!!!!!!!")

# for mean in meanVector:
# 	for feature in mean:
# 		outfile.write(str(feature)+" ")
# 	outfile.write("\n")
# outfile.close()

# if len(wholeData[0]) < 3:
# 	gp.plotClustersAndMean(wholeData, numOfClusters, pointsAssignCluster, meanVector)
# print("Cluster centers = ", kmeans.cluster_centers_)

# print("Initializer: ",pointsAssignCluster)






apniList, gmmMeans, piVect, covMatVect = gmm4.master(threshold, len(wholeData), wholeData, len(wholeData[0]), numOfClusters, meanVector, pointsAssignCluster, targetP)

targetPath = targetFile
if not os.path.exists(os.path.dirname(targetFile)):
        try:
            os.makedirs(os.path.dirname(targetFile))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
try:
	targetPath = targetFile+ str(numOfClusters) + "_GMM.means"
	meanFile = open(targetPath, "w")
	print("target File: ", targetPath)

	for mean in meanVector:
		for feature in mean:
			meanFile.write(str(feature)+" ")
		meanFile.write("\n")
	meanFile.close()

	targetPath = targetFile+ str(numOfClusters) + "_GMM.piVect"
	piFile = open(targetPath,"w")
	print("target File: ", targetPath)

	for pi in piVect:
		piFile.write(str(pi)+" ")
	piFile.write("\n")
	piFile.close()

	targetPath = targetFile+ str(numOfClusters) + "_GMM.cov"
	covFile = open(targetPath,"w")
	print("target File: ", targetPath)

	for covMat in covMatVect:
		# for cov in covMat:
		for index in range(len(covMat)):
			covFile.write(str(covMat[index][index])+" ")
		covFile.write("\n")

	covFile.close()

except IOError:
	("File not created !!!!!!!!!!!!!!!!!!!!!!!!!")