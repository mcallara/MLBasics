import numpy as np

class data(object):
	@staticmethod
	def featureNorm(X):
		mu=np.mean(X,axis=0)
		std=np.std(X,axis=0)
		Xnorm=(X-mu)/std
		return Xnorm,mu,std

	def __init__(self,X,y):
		self.X=X
		self.y=y
		self.samples,self.features=np.shape(X) #X.shape
		self.Xaug=np.concatenate((np.ones([self.samples,1]),X),axis=1)
		self.Xnorm,self.mu,self.std=self.featureNorm(X)
		self.Xaugnorm=np.concatenate((np.ones([self.samples,1]),self.Xnorm),axis=1)

	def __repr__(self):
		return "Data Matrix"

	def shape(self):
		return self.X.shape

def cost(Theta,X,y):
	#Theta tiene que ser one-dimensional o columna.
	return sum((np.dot(X,Theta.transpose()).reshape(-1,)-y)**2)/X.shape[0]

def deriv(Theta,X,y):
	return (np.dot(X.transpose(),(np.dot(X,Theta)-y)))/X.shape[0]

def sto(Theta,X,y,alpha,maxite,tol):
	ite=0
	delta=np.inf
	aux=np.zeros([maxite,Theta.shape[0]])
	while delta>tol and ite<maxite:
		J=cost(Theta,X,y)
		aux[ite,]=Theta-alpha*deriv(Theta,X,y)
		Theta=aux[ite,]
		delta=J-cost(Theta,X,y)
		ite+=1
	print("Max ite: {0}, Delta: {1}".format(ite,delta))
	return aux[:ite,]

def normaleq(X,y):
	return np.dot(np.linalg.inv(np.dot(X.transpose(),X)),np.dot(X.transpose(),y))

from cardata import *
X=cars.ndata
y=cars.nlabels

data=data(X,y)

#Initialize Theta
#Theta=np.zeros([data.features+1])
Theta=np.random.uniform(0,1,data.features+1)

#Stochastic Gradient Parameters
alpha=0.01
maxite=10000
tol=0.000001

#Results
ThetaHist=sto(Theta,data.Xaugnorm,y,alpha,maxite,tol)
Theta=ThetaHist[-1,]
print('Theta GD:',Theta,'Cost GD:',cost(Theta,data.Xaugnorm,y))

#Normal Equation Result
ThetaNorm=normaleq(data.Xaugnorm,y)
print('Theta Normal Eq for Xnorm ',ThetaNorm,'Cost Normal Equation Norm:',cost(ThetaNorm,data.Xaugnorm,y))

print('---------------------------------------')
print('Without Normalizing')
print('---------------------------------------')

#Initialize Theta
#Theta=np.zeros([data.features+1])
Theta=np.random.uniform(0,1,data.features+1)

#Stochastic Gradient Parameters
alpha=0.00001
maxite=100000
tol=0.0000001
ThetaHist=sto(Theta,data.Xaug,y,alpha,maxite,tol)
Theta=ThetaHist[-1,]
print('Theta GD:',Theta,'Cost GD:',cost(Theta,data.Xaug,y))

#Normal Equation Result
ThetaNorm=normaleq(data.Xaug,y)
print('Theta Normal Eq',ThetaNorm,'Cost Normal Equation:',cost(ThetaNorm,data.Xaug,y))
#import matplotlib.pyplot as plt
#plt.plot(histJ)
#plt.show()