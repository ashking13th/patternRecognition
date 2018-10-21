import numpy as np 
import os, argparse
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from scipy.interpolate import spline

kabTak = 0
wholeData = []

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=True, help="Raw data set location")
ap.add_argument("-n", "--name", required=True, help="name data set location")
args = vars(ap.parse_args())

def makeGraph(values, graphName, title):
	y = range(1,len(values)+1)
	# xnew = np.linspace(1,len(values),300) #300 represents number of points to make between T.min and T.max
	# power_smooth = spline(y,values,xnew)

	plt.plot(y,values)
	# plt.plot(xnew,power_smooth)
	plt.xlabel("No. of iterations")
	plt.ylabel("loglikelihood")
	plt.title(title)
	plt.savefig(graphName+".jpg")
	plt.close()

def fileHandle(fileName):
	file = open(fileName)
	for line in file:
		teLine = line.rstrip('\n ').split(' ')
		nLine = [float(i) for i in teLine]
		nLine = np.array(nLine)
		wholeData.append(nLine)

	file.close()
	return

for root, dirs, files in os.walk(args["source"]):
	for f in files:
		path = os.path.relpath(os.path.join(root, f), ".")
		fileHandle(path)
		# lengthOfFile.append(len(wholeData)-cntForFile)
		# cntForFile = len(wholeData)

wholeData = np.array(wholeData)
# print("Len = ", len(wholeData))

# while (2**kabTak <= 32):
apniList = []
for ind in range(8):
	GMM = GaussianMixture(n_components = 16, covariance_type = 'diag', tol = 0.001, reg_covar = 1e-6, max_iter = ind+1, n_init = 1, init_params = 'kmeans', weights_init = None, means_init = None, precisions_init = None, random_state = None, warm_start = False, verbose = 0, verbose_interval = 10).fit(wholeData)
	apniList.append(GMM.lower_bound_)
makeGraph(apniList, args["name"] + "_B"+ str(16), "Iterations vs loglikelihood")
	# kabTak += 1