import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image
#-------------------------------------------------------------------------------------------------------

#Loading the Training dataset----------------

def hot_encoding(label_vec):
    labels = np.empty(shape=[2,0])
    for i in range(label_vec.shape[1]):
        v = np.zeros((2,1))
        #print(label_vec[0][i])
        v[int(label_vec[0][i])] += 1
        labels = np.concatenate((labels,v),axis=1)
    return labels

filelist = glob.glob('CarDataset/*')
X_train1 = np.array([np.array(Image.open(fname)) for fname in filelist])
Y_train1 = np.zeros((1,X_train1.shape[0]))+1

files = glob.glob('NonCarDataset/*')
X_train2 = np.array([np.array(Image.open(fname)) for fname in files])
Y_train2 = np.zeros((1,X_train2.shape[0]))

X_train_orig = np.concatenate((X_train1,X_train2),axis=0)
X_train = X_train_orig.reshape((X_train_orig.shape[0],-1))
Y_train = np.concatenate((Y_train1,Y_train2),axis=1)
Y_train = hot_encoding(Y_train)
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
Y_test = hot_encoding(Y_test)
print("Shape of test feature mat: " + str(X_test.shape))
print("Shape of test label vec: " + str(Y_test.shape))

X_test = X_test/255
X_test = X_test.T

#-------------------------------------------------------------------------------------------------------

#Functions

def relu(Z):
	return np.maximum(Z,0),Z	#Z is the activation cache

def softmax(Z):
	t = np.exp(Z)
	#print(t) ###
	return t/(np.sum(t,axis=0)),Z			#Z is the activation cache
	
def softmax_backward(activation_cache,Y):	#Arguments - predictions and ground truth labels
	#print(activation_cache.shape) ###
	Z = activation_cache
	A,_ = softmax(Z)
	return (A-Y)

def relu_derivative(Z):
	return (np.maximum(Z,0) > 0)	#derivative of relu function

def relu_backward(dA, activation_cache):		#Uses activation cache 
	Z = activation_cache
	dZ = dA*relu_derivative(Z)
	return dZ

def initParams(layer_dims):
	L = len(layer_dims)	#here length of layer_dims is one more than the number of times the loop has to run
	parameters = {}
	for l in range(1,L):
		parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*0.001
		parameters['b' + str(l)] = np.zeros((layer_dims[l],1))

	return parameters

def initParams_var_one(layer_dims):
	parameters = {}
	L = len(layer_dims)
	for l in range(1,L):
		parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*np.sqrt(1/layer_dims[l-1])
		parameters['b' + str(l)] = np.zeros((layer_dims[l],1))

	return parameters

def linear_forward(A_prev, W, b):
	linear_cache = (A_prev, W, b)
	Z = np.dot(W,A_prev) + b
	return Z, linear_cache

def linear_activation_forward(A_prev, W, b, activation):
	
	if activation == "softmax":
		Z, linear_cache = linear_forward(A_prev, W, b)
		A, activation_cache = softmax(Z)
		#print("AL-size : "+str(A.shape))
	elif activation == "relu":
		Z, linear_cache = linear_forward(A_prev, W, b)
		A, activation_cache = relu(Z)

	cache = (linear_cache, activation_cache)
	#print(Z.shape) ###
	return A, cache

def model_forward(X, parameters):
	
	caches = []
	A = X
	L = len(parameters) // 2

	for l in range(L-1):
		A_prev = A
		A, cache = linear_activation_forward(A_prev, parameters['W' + str(l+1)], parameters['b' + str(l+1)],"relu")
		caches.append(cache)
	AL, cache = linear_activation_forward(A,parameters['W' + str(L)], parameters['b' + str(L)],"softmax")
	caches.append(cache)
	return AL, caches

def compute_cost(AL, Y):
	loss = -Y*np.log(AL)
	#print(AL) ###
	m = Y.shape[1]
	return (1/m)*np.sum((np.sum(loss, axis=0,keepdims=True)),axis=1)

def linear_backward(dZ, linear_cache):
	A_prev, W, b = linear_cache
	m = A_prev.shape[1]
	#print(W.shape)	##
	
	dW = (1/m)*np.dot(dZ, A_prev.T)
	db = (1/m)*np.sum(dZ, axis=1,keepdims=True)
	dA_prev = np.dot(W.T, dZ)
	return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation,Y):
	(linear_cache, activation_cache) = cache

	if activation == "softmax":
		dZ = softmax_backward(activation_cache,Y)
		#print(dZ.shape) ###
		dA_prev, dW, db = linear_backward(dZ, linear_cache)
	elif activation == "relu":
		dZ = relu_backward(dA, activation_cache)
		#print(dZ.shape) ###
		dA_prev, dW, db = linear_backward(dZ, linear_cache)
	return dA_prev, dW, db

def model_backward(AL, Y, caches):
	m = Y.shape[1]
	L = len(caches)
	grads = {}

	current_cache = caches[L-1]
	grads['dA' + str(L-1)], grads['dW' + str(L)], grads['db' + str(L)] = linear_activation_backward(0, current_cache, "softmax", Y)

	for l in reversed(range(L-1)):
		current_cache = caches[l]
		#print(grads['dA' + str(L-1)].shape) ###3\
		grads['dA' + str(l)], grads['dW' + str(l+1)], grads['db' + str(l+1)] = linear_activation_backward(grads['dA'+str(l+1)],current_cache,"relu", Y)

	return grads

def gradDescent(parameters, grads, learning_rate):
	L = len(parameters) // 2

	for l in range(L):
		parameters['W' + str(l+1)] -= learning_rate*grads['dW' + str(l+1)]
		parameters['b' + str(l+1)] -= learning_rate*grads['db' + str(l+1)]

	return parameters

def predict(X, parameters):
	
	AL, _ = model_forward(X, parameters)
	predictions = np.argmax(AL, axis=0)
	return predictions

def compute_accuracy(predictions, Y):
    #m = Y.shape[1]
    print(m)
    Y = np.argmax(Y, axis=0)
    return (1/m)*np.sum(Y == predictions)*100

#--------------------------------------------------------------------------------------------------------------

#Model

def model(X, Y, layer_dims, learning_rate=0.001, iterations=1000,print_cost=False):
	parameters = initParams(layer_dims)
	costs = []
	print("Size of X: " + str(X.shape) + " and size of Y: " + str(Y.shape))
	for i in range(iterations):
		AL, caches = model_forward(X, parameters)
		cost = compute_cost(AL, Y)
		grads = model_backward(AL, Y, caches)
		parameters = gradDescent(parameters, grads, learning_rate)
		if i%100 == 0 and print_cost == True:
			print("Cost after %i iterations is %f" %(i, cost))
		if i%100 == 0:
			costs.append(cost)
	plt.plot(np.squeeze(costs))
	plt.ylabel('cost')
	plt.xlabel('iterations (per tens)')
	plt.title("Learning rate = " + str(learning_rate))
	plt.show()

	return parameters

#-------------------------------------------------------------------------------------------------------------

#Main Code


#Training

layer_dims = [X_train.shape[0],11,Y_train.shape[0]]
parameters = model(X_train, Y_train, layer_dims, 0.03, 2000,1)

#Predictions

#Train
predictions = predict(X_train, parameters)
train_accuracy = compute_accuracy(predictions,Y_train)
#print(Y_train)
#Test
predictions = predict(X_test, parameters)
test_accuracy = compute_accuracy(predictions,Y_test)
#print(predictions)
print("Train accuracy: " + str(train_accuracy))
print("Test accuracy: " + str(test_accuracy))
