import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from operator import itemgetter
import math
import random
import sys
import os
from sklearn import mixture

k = int(sys.argv[1])
dim = 1
w_vec = []

#====================================================================================================================================

def input_classwise(folder):
	data_points = [[],[],[]]

	for c in range(1,4):
		f = open(folder+"/train_class"+str(c)+".txt","r")
		for i in f.readlines():
			x = [float(j) for j in i.split()]
			data_points[c-1].append(x)

		# f = open(folder+"/test_class"+str(c)+".txt","r")
		# for i in f.readlines():
		# 	x = [float(j) for j in i.split()]
		# 	data_points[c-1].append(x)

	data_points = np.array(data_points)
	return np.array(data_points)


#====================================================================================================================================

def fda(folder, c_p, c_n, data_plus, data_minus):
	mean_plus = np.mean(data_plus, axis=0)
	mean_minus = np.mean(data_minus, axis=0)

	covar_plus = np.cov(np.transpose(data_plus))
	covar_minus = np.cov(np.transpose(data_minus))

	scatter_matrix_plus = len(data_plus)*covar_plus
	scatter_matrix_minus = len(data_minus)*covar_minus

	s_w = scatter_matrix_plus + scatter_matrix_minus
	s_w_inv = inv(s_w)

	w = 1000*np.matmul(s_w_inv, mean_plus-mean_minus)

	# print "SW :"
	# print s_w
	# print "WWWWW : ",w

	train_A = []
	train_B = []

	f = open(folder+"/train_class"+str(c_p)+".txt","r")
	for i in f.readlines():
		x = [float(j) for j in i.split()]
		x=np.array(x)
		r_x=np.matmul(np.transpose(w),x)
		train_A.append(r_x)
	train_A=np.array(train_A)

	f = open(folder+"/train_class"+str(c_n)+".txt","r")
	for i in f.readlines():
		x = [float(j) for j in i.split()]
		x=np.array(x)
		r_x=np.matmul(np.transpose(w),x)
		train_B.append(r_x)
	train_B=np.array(train_B)

	# print "TRAIN A : ",train_A.shape
	# print "TRAIN B : ", train_B.shape

	w_vec.append(w)

	# print "PREDICTED DATA A AND B :"
	# print train_A
	# print train_B
	return w,train_A,train_B


#=================================== KMEANS ===================================================================
def kMeans(data_points):
	mu = np.zeros((k,dim))

	for i in range(k):
		# ind = random.randint(0,len(data_points)-1)
		mu[i] = data_points[i]
	
	flag = True
	# plt.ion()
	while(flag):
		flag = False
		cluster = []
		old_mu = np.zeros((k,dim))
		old_mu = np.copy(mu)

		for i in range(k):
			cluster.append([])
			# plt.plot(mu[i][0],mu[i][1],'bo',markersize=6,marker='v',color='black')
	
		for i in range(len(data_points)):
			minimum = np.linalg.norm(data_points[i]-mu[0])
			index = 0
			for j in range(k):
				dist = np.linalg.norm(data_points[i]-mu[j])
				if minimum > dist:
					minimum = dist
					index = j
			cluster[index].append(data_points[i])

		for i in range(k):
			# x,y = zip(*cluster[i])
			# plt.scatter(x,y,s=1,color = (Colour[i][0],Colour[i][1],Colour[i][2]))
			mu[i] = np.mean(cluster[i],axis=0)
		
		if np.linalg.norm(old_mu-mu)!=0:
			flag =True
		# plt.draw()
		# plt.pause(0.8)
		# plt.clf()	

	covar = np.zeros((k,dim,dim))

	for i in range(k):
		print "			Print Length ",len(cluster[i])
		if len(cluster[i])==1:
			print "				Vector : " ,cluster[i]
		covar[i] = np.cov(np.transpose(cluster[i]))
		for j in range(len(covar[i])):
			for x in range(len(covar[i])):
				if j!=x:
					covar[i][j][x] = 0.0


	pi = np.zeros(k)
	for i in range(k):
		pi[i] = float(len(cluster[i]))/float(len(data_points))

	return mu,covar,pi,data_points

#====================================================================================================================================

def find_det(matrix):
	val = 1.0
	for i in range(len(matrix)):
		val*= matrix[i][i]
	return val	

#====================================================================================================================================

def findInverse(matrix):
	x = np.zeros((len(matrix),len(matrix)))

	for i in range(len(matrix)):
		x[i][i] = (1/matrix[i][i])
	return x	

#====================================================================================================================================

def findN(x,mu,sigma):
	det = find_det(sigma)
	if det==0.0:
		det=0.00000001
	invSigma = findInverse(sigma)

	N = (1/math.sqrt(2*math.pi*det))   ##check here
	N *= math.exp((-1.0/2.0)*np.matmul(np.matmul(np.transpose(x-mu),invSigma),x-mu))

	return N

#====================================== GMM MODEL =========================================================================================
def GMM(mean,covariance,pi,data_points):
	l_theta_old = 1000000000000
	l_theta = 0.0
	itr = 1
	iterationsX,loglikely = [],[]
	# plt.ion()
	while abs(l_theta - l_theta_old) > 0.01:
		l_theta_old = l_theta
		l_theta = 0.0
		cluster = []

		for i in range(k):
			cluster.append([])
			# plt.plot(mean[i][0],mean[i][1],'bo',markersize=10,marker='v',color='black')

		# for i in range(k):
		# 	x,y = zip(*cluster[i])
		# 	plt.scatter(x,y,s=10,color = (Colour[i][0],Colour[i][1],Colour[i][2]))	

		for i in range(len(data_points)):
			sumK = 0.0
			for j in range(k):
				sumK += pi[j]*findN(data_points[i],mean[j],covariance[j]) 
			l_theta += math.log(sumK)	

		gammaZNK = np.zeros((len(data_points),k))
		
		for i in range(len(data_points)):
			prob=0.0
			for j in range(k):
				gammaZNK[i][j] = (pi[j]*findN(data_points[i],mean[j],covariance[j]))
				prob+=gammaZNK[i][j]
			# l_theta += np.log(prob)
			gammaZNK[i]/=prob

		NK = np.zeros(k)

		for j in range(k):
			sumN = 0.0
			for i in range(len(data_points)):
				sumN += gammaZNK[i][j]
			NK[j] = sumN
			
		new_mean = np.zeros((k,dim))

		for j in range(k):
			sumN = np.zeros(dim)
			for i in range(len(data_points)):
				sumN += (gammaZNK[i][j]*data_points[i])
			sumN = sumN/NK[j]
			new_mean[j] = np.copy(sumN)

		new_covariance = np.zeros((k,dim,dim))

		for j in range(k):
			sumN = np.zeros((dim,dim))
			for i in range(len(data_points)):
				mat = np.matmul(np.array(data_points[i]-mean[j]).reshape((dim,1)),np.array(data_points[i]-mean[j]).reshape((1,dim)))
				sumN += (gammaZNK[i][j]*mat)
			sumN = sumN/NK[j]
			for x in range(dim):
				for y in range(dim):
					if x!=y :
						sumN[x][y] = 0.0
			new_covariance[j] = np.copy(sumN)	
		
		new_pi = np.zeros(k)

		for i in range(k):
			new_pi[i] = float(NK[i]/len(data_points))

		covariance = np.copy(new_covariance)
		mean = np.copy(new_mean)
		pi = np.copy(new_pi)

		loglikely.append(l_theta)
		iterationsX.append(itr)
		itr += 1

		# plt.xlabel("X1 -->")
		# plt.ylabel("X2 -->")
		# plt.savefig("my"+str(cl)+".png",bbox_inches="tight", pad_inches=0.5)	
		# plt.draw()
		# plt.pause(0.0001)
		# plt.clf()
		print "			GMM DIFF :",abs(l_theta - l_theta_old)
		
	return mean,covariance,pi,data_points

#============================================================= BULING MODELS for 2 CLASSES ==============================================

parameters=[]
def GMM_two_class(trainA, trainB):
	param_turn = []
	MEAN,COVARIANCE, PI = [],[],[]
	print "BUILDING MODEL btw 2 Classes:"
	print "		Kmeans Running :"
	mean, cov, pi, data_points=kMeans(trainA)
	print "		GMM Running :"
	mean_gmm, cov_gmm, pi_gmm, data_points_gmm = GMM(mean, cov, pi, data_points)
	MEAN.append(mean_gmm)
	COVARIANCE.append(cov_gmm)
	PI.append(pi_gmm)

	mean, cov, pi, data_points=kMeans(trainB)
	mean_gmm, cov_gmm, pi_gmm, data_points_gmm = GMM(mean, cov, pi, data_points)
	MEAN.append(mean_gmm)
	COVARIANCE.append(cov_gmm)
	PI.append(pi_gmm)

	param_turn.append(MEAN)
	param_turn.append(COVARIANCE)
	param_turn.append(PI)
	parameters.append(param_turn)


#============================================================ FIND CLASS =======================================
def findClass(data_x):
	# data = [x, y]
	# print(data)
	c1, c2, c3=0,0,0

	for t in range(3):
		data=np.matmul(np.array(w_vec[t]),data_x)

		# print "		Reduced Test Data : ",data
		MEAN, COVARIANCE, PI = parameters[t][0], parameters[t][1], parameters[t][2]
		ll = np.zeros(2)
		for c in range(2):
			sumOverAllClusters = 0.0
			for clusterNum in range(k):
				sumOverAllClusters += (PI[c][clusterNum])*findN(data, MEAN[c][clusterNum], COVARIANCE[c][clusterNum])
			if(sumOverAllClusters < 0.00000000001):
				sumOverAllClusters = 0.00000000001
			ll[c] = math.log(sumOverAllClusters)

		if(ll[0] > ll[1]):
			if(t==0):
				c1+=1
				# print "			Predicted Class in turn 1 : 1"
			elif(t==1):
				c1+=1
				# print "			Predicted Class in turn 2 : 1"
			elif(t==2):
				c2+=1
				# print "			Predicted Class in turn 3 : 2"
		elif(ll[0]<ll[1]):
			if(t==0):
				c2+=1
				# print "			Predicted Class in turn 1 : 2"
			elif(t==1):
				c3+=1
				# print "			Predicted Class in turn 2 : 3"
			elif(t==2):
				c3+=1
				# print "			Predicted Class in turn 3 : 3"
		else:
			if(t==0):
				c1+=1
			elif(t==1):
				c3+=1
			elif(t==2):
				c2+=1
	if(c1>c2 and c1>c3):
		# print "				FINAL predicted : 1"
		return 0
	elif(c2>c1 and c2>c3):
		# print "				FINAL predicted : 2"
		return 1
	elif(c3>c1 and c3>c2):
		# print "				FINAL predicted : 3"
		return 2
	else:
		# print "MISSSCLASSIFIED :",data_x
		return 0

####################################################################################

def decision_boundary_plot():
	test_data=[[],[],[]]
	back_data1,back_data2,back_data3=[],[],[]

	for i in range(3):
		f = open("old_data/test_class"+str(i+1)+".txt","r")
		for j in f.readlines():
			x = [float(jj) for jj in j.split()]
			# print x
			test_data[i].append(x)
	test_data=np.array(test_data)

	i = -5
	while(i<=25):
		j = -20
		while(j<=10):
			point=[i,j]
			# point=np.array(point)
			# print "Point : ",point
			c = findClass(point)
			if(c==0):
				back_data1.append(point)
			elif(c==1):
				back_data2.append(point)
			elif(c==2):
				back_data3.append(point)
			j += 0.05
		i += 0.05
	# back_data=np.array(back_data)
	back_data1=np.array(back_data1)
	back_data2=np.array(back_data2)
	back_data3=np.array(back_data3)
	
	plt.xlabel("X Coordinate")
	plt.ylabel("Y Coordinate")
	plt.scatter(back_data1[:,0], back_data1[:,1],c="pink", edgecolor="", alpha=0.3)
	plt.scatter(back_data2[:,0], back_data2[:,1],c="lightgreen", edgecolor="", alpha=0.3)
	plt.scatter(back_data3[:,0], back_data3[:,1], c="lightblue", edgecolor="", alpha=0.3)
	plt.scatter(test_data[0][:,0], test_data[0][:,1],label="Class1", c="red", edgecolor="black", alpha=1)
	plt.scatter(test_data[1][:,0], test_data[1][:,1],label="Class2", c="green", edgecolor="black", alpha=1)
	plt.scatter(test_data[2][:,0], test_data[2][:,1],label="Class3", c="blue", edgecolor="black", alpha=1)
	plt.legend(loc="upper right")
	plt.show()


data = input_classwise("old_data")
w12,train12A,train12B =fda("old_data",1,2,data[0],data[1])
w13,train13A,train13B =fda("old_data",1,3,data[0],data[2])
w23,train23A,train23B =fda("old_data",2,3,data[1],data[2])

GMM_two_class(train12A,train12B)
GMM_two_class(train13A,train13B)
GMM_two_class(train23A,train23B)

decision_boundary_plot()

ConfusionMatrix = np.zeros([3,3])

l = [0, 0 , 0]

for i in range(3):
	f = open("old_data/test_class"+str(i+1)+".txt","r")
	for j in f.readlines():
		l[i]+=1
		x = [float(jj) for jj in j.split()]
		# print "Test Data : ",x
		# print "		Actual Class : ",i+1
		c = findClass(x)
		if(c!=3):
			ConfusionMatrix[i][c] += 1

print "Confusion Matrix :\n",ConfusionMatrix, "\n"

accuracy = float((ConfusionMatrix[0][0]+ConfusionMatrix[1][1]+ConfusionMatrix[2][2])/(l[0]+l[1]+l[2]))
print "Accuracy : ", accuracy*100, "%\n"

recallC1 = float(ConfusionMatrix[0][0]/l[0])
recallC2 = float(ConfusionMatrix[1][1]/l[1])
recallC3 = float(ConfusionMatrix[2][2]/l[2])

print "Recall :"
print "Class 1 :   ",recallC1
print "Class 2 :   ",recallC2
print "Class 3 :   ",recallC3
print "Mean Recall : ", (recallC1+recallC2+recallC3)/3
print "\n"


precisionC1=float((ConfusionMatrix[0][0])/(ConfusionMatrix[0][0]+ConfusionMatrix[1][0]+ConfusionMatrix[2][0]))
precisionC2=float((ConfusionMatrix[1][1])/(ConfusionMatrix[0][1]+ConfusionMatrix[1][1]+ConfusionMatrix[2][1]))
precisionC3=float((ConfusionMatrix[2][2])/(ConfusionMatrix[0][2]+ConfusionMatrix[1][2]+ConfusionMatrix[2][2]))

print "Precision :"
print "Class 1 :   ",precisionC1
print "Class 2 :   ",precisionC2
print "Class 3 :   ",precisionC3
print "Mean Precision : ", (precisionC1+precisionC2+precisionC3)/3
print "\n"

if (precisionC1+recallC1)==0:
	fmeasureC1=0
else: fmeasureC1 = float((2*(precisionC1*recallC1))/(precisionC1+recallC1))
if (precisionC2+recallC2)==0:
	fmeasureC2=0
else: fmeasureC2 = float((2*(precisionC2*recallC2))/(precisionC2+recallC2))
if (precisionC3+recallC3)==0:
	fmeasureC3=0
else: fmeasureC3 = float((2*(precisionC3*recallC3))/(precisionC3+recallC3))

print "F-Measure :"
print "Class 1 :   ",fmeasureC1
print "Class 2 :   ",fmeasureC2
print "Class 3 :   ",fmeasureC3
print "Mean F-Measure : ", (fmeasureC1+fmeasureC2+fmeasureC3)/3
print "\n"