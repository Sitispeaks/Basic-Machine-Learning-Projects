import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image

#-------------------------------------------------------------------------

#Loading the Training dataset----------------

filelist = glob.glob('CarDataset/*')
X_train1 = np.array([np.array(Image.open(fname)) for fname in filelist])
Y_train1 = np.zeros((1,X_train1.shape[0]))+1

files = glob.glob('NonCarDataset/*')
X_train2 = np.array([np.array(Image.open(fname)) for fname in files])
Y_train2 = np.zeros((1,X_train2.shape[0])) 

X_train_orig = np.concatenate((X_train1,X_train2),axis=0)
X_train = X_train_orig.reshape((X_train_orig.shape[0],-1))
Y_train = np.concatenate((Y_train1,Y_train2),axis=1)
print("Shape of the label vector is " + str(Y_train.shape))
print("Shape of the feature matrix is " + str(X_train.shape))

X_train = X_train/255
X_train = X_train.T

#Loading the Test set------------------------

filelist = glob.glob('CarTest/*')
X_test1 = np.array([np.array(Image.open(fname)) for fname in filelist])
Y_test1 = np.zeros((1,X_test1.shape[0]))+1

files = glob.glob('NonCarTest/*')
X_test2 = np.array([np.array(Image.open(fname)) for fname in files])
Y_test2 = np.zeros((1,X_test2.shape[0]))

X_test_orig = np.concatenate((X_test1,X_test2),axis=0)
X_test = X_test_orig.reshape((X_test_orig.shape[0],-1))
Y_test = np.concatenate((Y_test1,Y_test2),axis=1)
print("Shape of test feature mat: " + str(X_test.shape))
print("Shape of test label vec: " + str(Y_test.shape))

X_test = X_test/255
X_test = X_test.T

#-------------------------------------------------------------------------


#Functions-----------------------------------
def sigmoid(Z):
	return 1/(1 + np.exp(-Z)), Z
	
def relu(Z):
	return np.maximum(0,Z), Z

def sigmoid_backward(dA ,activation_cache):
	Z = activation_cache
	a = (sigmoid(Z))[0]
	dZ = dA*((a)*(1-a))
	return dZ

def relu_derivative(Z):
	max_0 = np.maximum(Z, 0)
	return (max_0 >= 0) 	

def relu_backward(dA, activation_cache):
	Z = activation_cache
	dZ = dA*relu_derivative(Z)
	return dZ

def initParams(layer_dims):
	#Input: Layer dimensions for a deep neural network.
	#Output: Initialized parameters dictionary.
	parameters = {}
	L = len(layer_dims)

	for l in range(1,L):
		parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
		parameters['b' + str(l)] = np.zeros((layer_dims[l],1))

	return parameters

def linear_forward(A_prev,W,b):
	#Input: Activation matrix of layer 'l-1', parameters of the current layer 'l'.
	#Output: 'Z' and a 'cache' containing A, W and b.
	Z = np.dot(W,A_prev) + b
	cache = (A_prev, W, b)
	return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
	#Input: Activation matrix of layer 'l-1', parameters W and b of the current layer 'l' and string a 'activation' containing the name of the activation function.
	#output: Activation of current layer 'l' and cache dictionary containing linear cache and activation cache. 

	if activation == "sigmoid":
		Z, linear_cache = linear_forward(A_prev,W,b)
		A, activation_cache = sigmoid(Z)
	elif activation == "relu":
		Z, linear_cache = linear_forward(A_prev,W,b)
		A, activation_cache = relu(Z) 

	cache = (linear_cache, activation_cache)
	return A, cache

def model_forward(X, parameters):
	#Input: Feature matrix X, parameters dictionary
	#Output: Activation of the last layer (i.e prediction), 'caches' list of all 'cache's
	
	caches = [] #caches list
	A = X
	L = len(parameters) // 2 #Integer division

	for l in range(1,L):
		A_prev = A
		A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)],"relu")
		caches.append(cache)
	
	AL, cache = linear_activation_forward(A,parameters['W' + str(L)], parameters['b' + str(L)],"sigmoid")
	caches.append(cache)

	return AL, caches

def compute_cost(AL, Y):
	#Input: Predictions, labels vector
	#Output: cost
	m = Y.shape[1]
	cost = -(1/m)*np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL),axis=1)  
	return cost

def  linear_backward(dZ ,cache):
	A_prev, W, b = cache
	m = A_prev.shape[1]

	dW = (1/m)*np.dot(dZ,A_prev.T)
	db = (1/m)*np.sum(dZ,axis=1,keepdims=True)
	dA_prev = np.dot(W.T,dZ)

	return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
	linear_cache,activation_cache = cache	#Activation cache: Contains Z, Linear cache: contains A_prev, W, b

	if activation == "relu":
		dZ = relu_backward(dA, activation_cache)
		dA_prev, dW, db = linear_backward(dZ ,linear_cache)

	elif activation == "sigmoid":
		dZ = sigmoid_backward(dA, activation_cache)
		dA_prev, dW, db = linear_backward(dZ, linear_cache)

	return dA_prev, dW, db

def model_backward(AL, Y, caches):
	m = Y.shape[1]
	L = len(caches)
	grads = {}
	
	dAL = -np.divide(Y,AL) + np.divide(1-Y,1-AL)		
	current_cache = caches[L-1]
	grads['dA' + str(L-1)],grads['dW'+str(L)],grads['db' + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")

	for l in reversed(range(L-1)):
		current_cache = caches[l] 
		grads['dA' + str(l)],grads['dW' + str(l+1)],grads['db' + str(l+1)] = linear_activation_backward(grads['dA' + str(l+1)], current_cache,"relu")
		
	return grads

def gradDescent(parameters, grads, s, learning_rate,epsilon=10e-8):
	L = len(parameters) // 2 

	for l in range(L):
		parameters['W' + str(l+1)] -= learning_rate*grads['dW' + str(l+1)]/(np.sqrt(s['dW' + str(l+1)]) + epsilon)
		parameters['b' + str(l+1)] -= learning_rate*grads['db' + str(l+1)]/(np.sqrt(s['db' + str(l+1)]) + epsilon)

	return parameters
def rms_init(parameters):
	L = len(parameters) // 2
	s = {}
	for l in range(L):
		s['dW' + str(l+1)] = np.zeros((parameters['W' + str(l+1)]).shape)
		s['db' + str(l+1)] = np.zeros((parameters['b' + str(l+1)]).shape)
	return s
def rms_update(s,grads,i,beta=0.999):
	L = len(s) // 2
	for l in range(L):
		s['dW' + str(l+1)] = s['dW' + str(l+1)]*beta + (1-beta)*np.square(grads['dW' + str(l+1)])
		s['dW' + str(l+1)] = s['dW' + str(l+1)]*beta + (1-beta)*np.square(grads['dW' + str(l+1)]) 
	return s

def correction(s,i,beta):
	L = len(s) // 2
	if i == 0:
		i = 1
	for l in range(L):
		s['dW' + str(l+1)] /= (1-(beta**i))
		s['db' + str(l+1)] /= (1-(beta**i))
	return s

def predict(X, Y, parameters):
	AL,caches = model_forward(X, parameters)
	return (AL > 0.5)

#------------------------------------------------------------------------

#Model

def model(X, Y, layer_dims, learning_rate = 0.01, iterations = 200, print_cost = False):	
	parameters = initParams(layer_dims)
	costs = []
	s = rms_init(parameters)
	beta = 0.999
	for i in range(iterations):
		AL,caches = model_forward(X, parameters)
		cost = compute_cost(AL, Y)
		grads = model_backward(AL, Y, caches)
		s = rms_update(s,grads,i,beta)
		s_corrected = correction(s,i,beta)
		parameters = gradDescent(parameters, grads, s_corrected, learning_rate,10e-8)
		if i%100 == 0 and print_cost == 1:
			print("Cost after %i iterations is %f" %(i,cost))
		if i % 100 == 0:
			costs.append(cost)

	plt.plot(np.squeeze(costs))
	plt.ylabel('cost')
	plt.xlabel('iterations (per tens)')
	plt.title("Learning rate =" + str(learning_rate))
	plt.show()

	return parameters


		 
#------------------------------------------------------------------------
		 
#Main code
layer_dims = [X_train.shape[0],11,1]
parameters = model(X_train, Y_train, layer_dims,0.003,4000,True)

pred_train = predict(X_train,Y_train, parameters)
pred_test = predict(X_test,Y_test,parameters)

m_train = Y_train.shape[1]
m_test = Y_test.shape[1]

correct_train = np.sum(pred_train == Y_train)
correct_test = np.sum(pred_test == Y_test)
accuracy_train = (correct_train/m_train)*100
accuracy_test = (correct_test/m_test)*100

print("Training set accuracy is " + str(accuracy_train) + "%")
print("Test set accuracy is " + str(accuracy_test) + "%")
