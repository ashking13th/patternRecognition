import numpy as np 
import sklearn.svm as svm 

nClass=3

def fileHandle(fileName):
	file = open(fileName)
	tempList = []

	for line in file:
		teLine = line.rstrip('\n ').split(' ')
		nLine = [float(i) for i in teLine]
		tempList.append(np.array(teLine,float))

	file.close()
	x = np.array(tempList,float)
	return x

mainList = []
mainList.append(fileHandle("bayou_train.txt"))
mainList.append(fileHandle("chalet_train.txt"))
mainList.append(fileHandle("creek_train.txt"))

testList = []
testList.append(fileHandle("bayou_test.txt"))
testList.append(fileHandle("chalet_test.txt"))
testList.append(fileHandle("creek_test.txt"))

nFeature=len(mainList[0][0])

X=mainList[0]
for i in range(1,nClass):
	X=np.concatenate((X,mainList[i]),axis=0)
# print(X.shape)
Y=np.zeros(len(mainList[0]),int)
for i in range(1,nClass):
	b=(np.zeros(len(mainList[i]),int)+i)
	Y=np.concatenate((Y,b))
# print(Y.shape)
# testData=testList[0]
# for i in range(1,nClass):
# 	tes=np.concatenate((X,mainList[i]),axis=1)

clf=svm.SVC(kernel='poly',gamma='auto',coef0=3,degree=3)
clf.fit(X,Y)
# a=np.array([[1,1.2],[2,4]],float)
confusionMatrix=np.zeros((nClass,nClass),int)

for i in range(nClass):
	pr=clf.predict(testList[i])
	for j in pr:
		confusionMatrix[i][j]+=1

print(confusionMatrix)
