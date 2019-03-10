import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas
import math

#-----------------------------------------------------------------------------------------------------------

#Loading data------------------------------------------------------------

a = pandas.read_csv("nba.csv",sep=",",skiprows=1)
data = a.values
#data = np.loadtxt('nba.csv',delimiter=',',skiprows=1) 
data = data[:,1:data.shape[1]] 
X_train = np.array(data[:,0:data.shape[1]-1],dtype=np.float)
Y_train = np.array(data[:,-1],dtype=np.float)
Y_train = Y_train.reshape((1,Y_train.shape[0]))

X_train = X_train.T
maxs = np.amax(X_train,axis=1)
mins = np.amin(X_train,axis=1)
ranges = (maxs-mins).reshape(maxs.shape[0],1)
X_train = X_train/ranges


print("Shape of the feature matrix: " + str(X_train.shape) + " Shape of the labels vector: " + str(Y_train.shape))
print(X_train)
print(Y_train)

#-----------------------------------------------------------------------------------------------------------

#Functions---------------------------------------------------------------

def sigmoid(z):
	#Input: Any scalar, numpy vector or numpy matrix
	#Return value: Elementwise sigmoid applied on input
	return 1/(1 + np.exp(-z))
		
def layer_sizes(X,Y):
	#Input: Feature mat X and label vec Y
	#Return values: Input layer size, hidden layer size and output layer size
	n_x = X.shape[0]
	n_h = 10
	n_y = Y.shape[0]
	return n_x,n_h,n_y

def initParams(n_x,n_h,n_y):
	#Input: Sizes of input layer, hidden layer and output layer
	#Return values: Parameters W1,b1,W2 and b2 in a dictionary
	W1 = (np.random.randn(n_h,n_x))*0.01
	b1 = (np.random.randn(n_h,1))*0.01
	W2 = (np.random.randn(n_y,n_h))*0.01
	b2 = (np.random.randn(n_y,1))*0.01
	parameters = {"W1" : W1,
			      "b1" : b1,
		     	  "W2" : W2,
			      "b2" : b2}
	return parameters

def forward(X,parameters):
	#Input: Training feature matrix and parameters dictionary
	#Return values: A2 and dictionary containing activation matrices and 'Z' matrices
	W1 = parameters['W1']
	b1 = parameters['b1']
	W2 = parameters['W2']
	b2 = parameters['b2']
	
	Z1 = np.dot(W1,X) + b1
	A1 = np.tanh(Z1)
	Z2 = np.dot(W2,A1) + b2
	A2 = sigmoid(Z2)
	
	cache = {"Z1" : Z1, "A1" : A1, "Z2" : Z2, "A2" : A2}
	return A2,cache

def cost(A2, Y, parameters):
	#Input: Activation mat A2, labels vector Y and parameters dictionary
	#Output: Cross entpropy cost
	m = Y.shape[1]
	cost_val = -(1/m)*np.sum(Y*np.log(A2) + (1-Y)*np.log(1-A2)) 
	return cost_val

def backward(parameters, cache, X, Y):
	#Input: Parameters dictionary, cache dictionary, feature mat X and labels vector Y
	#Output: grads dictionary
	W1 = parameters['W1']
	b1 = parameters['b1']
	W2 = parameters['W2']
	b2 = parameters['b2']
	
	m = Y.shape[1]

	A1 = cache['A1']
	A2 = cache['A2']
	
	dZ2 = A2 - Y
	dW2 = (1/m)*np.dot(dZ2,A1.T)
	db2 = (1/m)*np.sum(dZ2,axis=1,keepdims=True)
	dZ1 = np.dot(W2.T, dZ2)*(1 - np.power(A1,2))  
	dW1 = (1/m)*np.dot(dZ1,X.T)
	db1 = (1/m)*np.sum(dZ1,axis=1,keepdims=True)

	grads = {"dW1" : dW1,
			 "db1" : db1,
			 "dW2" : dW2,
			 "db2" : db2}
	return grads

def gradDescent(parameters,grads,alpha=0.01):
	#Input: parameters dictionary, grads dictionary and learning rate 'alpha'
	#Return values: Updated parameter values

	W1 = parameters['W1']
	b1 = parameters['b1']
	W2 = parameters['W2']
	b2 = parameters['b2']
	
	dW1 = grads["dW1"]
	db1 = grads["db1"]
	dW2 = grads["dW2"]
	db2 = grads["db2"]

	W1 = W1 - alpha*dW1
	b1 = b1 - alpha*db1
	W2 = W2 - alpha*dW2
	b2 = b2 - alpha*db2

	updated_param =	{"W1" : W1,
			     	 "b1" : b1,
		     	   	 "W2" : W2,
			      	 "b2" : b2}
	return updated_param

def predict(parameters, X):
	#Input: parameters dictionary and feature matrix
	#Output: 1 if Success and 0 if Failure
	W1 = parameters['W1']
	b1 = parameters['b1']
	W2 = parameters['W2']
	b2 = parameters['b2']

	Z1 = np.dot(W1,X) + b1
	A1 = np.tanh(Z1)
	Z2 = np.dot(W2,A1) + b2
	A2 = sigmoid(Z2)
	return (A2 > 0.5)

#-----------------------------------------------------------------------------------------------------------

#Model--------------------------------------------------------------------------

def model(X, Y, n_h, iterations=200,print_cost=True):
	n_x = layer_sizes(X,Y)[0]
	n_y = layer_sizes(X,Y)[2]

	parameters = initParams(n_x,n_h,n_y) 
	for i in range (iterations):
		A2,cache = forward(X,parameters)
		cost_val = cost(A2, Y, parameters)	
		if i % 100 == 0 and i != 0 and print_cost == True:
			print("The cost after %i iterations is %f" %(i,cost_val))
		grads = backward(parameters, cache, X, Y)
		parameters = gradDescent(parameters,grads,alpha=0.01)
	

	return parameters

#-----------------------------------------------------------------------------------------------------------

#Main code---------------------------------------------------------------------

parameters = model(X_train,Y_train,18,8000,True)
predictions = predict(parameters, X_train)

correct_train = (np.sum(predictions==Y_train))/Y_train.shape[1]
percentage_train = correct_train*100
print("Training accuracy = " + str(percentage_train) + "%")

