from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob

#----------------------------------------------------------------------------------

#Functions

def sigmoid(Z):
	return 1/(1 + np.exp(-Z)), Z
	
def relu(Z):
	return np.maximum(Z,0), Z 

def sigmoid_backward(dA, activation_cache):
	Z = activation_cache
	A = (sigmoid(Z))[0] 
	dZ = dA*((1-A)*(A))
	return dZ

def relu_derivative(Z):
	return (np.maximum(0,Z) > 0)

def relu_backward(dA, activation_cache):
	Z = activation_cache
	dZ = dA*relu_derivative(Z)
	return dZ

def initParams_random(layer_dims):

	parameters = {}
	for l in range(1,len(layer_dims)):
		parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*0.01
		#print(str(layer_dims[l]) + " " + str(layer_dims[l-1]))
		parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
	return parameters

def initParams_selected_var(layer_dims):
	
	parameters = {}
	for l in range(1,len(layer_dims)): #Here we multiply with (1/#inputs_to_layer_l)^(1/2)
		parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1])*np.sqrt(1/layer_dims[l-1])
		#print(str(layer_dims[l]) + " " + str(layer_dims[l-1]))
		parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
	return parameters

def linear_forward(A_prev, W, b):
	Z = np.dot(W,A_prev) + b
	linear_cache = (A_prev, W, b)
	return Z, linear_cache

def linear_activation_forward(A_prev, W, b, activation):
	if activation == "sigmoid":
		Z, linear_cache = linear_forward(A_prev, W, b)
		A, activation_cache = sigmoid(Z)
	elif activation == "relu":
		Z, linear_cache = linear_forward(A_prev, W, b)
		A, activation_cache = relu(Z) 

	cache = (linear_cache, activation_cache)
	return A, cache

def model_forward(X, parameters):
	caches = []
	A = X
	L = len(parameters) // 2

	for l in range(1,L):
		A_prev = A
		A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
		caches.append(cache)

	AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
	caches.append(cache)
	return AL, caches

def model_forward_dropout(X, parameters, keep_probs):
	caches = []
	A_drop = X
	L = len(parameters) // 2
	d = []
	D = np.zeros(X.shape) + 1
	d.append(D)
	for l in range(1, L):
		A_prev = A_drop
		A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
		D = (np.random.rand(A.shape[0],A.shape[1]) < keep_probs)
		d.append(D)
		A_drop = (np.multiply(A,D))/keep_probs
		caches.append(cache)

	AL, cache = linear_activation_forward(A_drop, parameters['W' + str(L)], parameters['b' + str(L)], "sigmoid")
	caches.append(cache)
	#print(len(d))
	return AL, caches, d

def compute_cost(AL, Y):
	m = Y.shape[1]
	cost = (-1/m)*np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL),axis=1) 
	return cost

def compute_cost_reg(AL, Y,parameters, lambd):
	m = Y.shape[1]
	sum_val = 0
	cost = (-1/m)*np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL),axis=1)
	L = len(parameters) // 2
	for l in range(L):
		sum_val += (lambd/(2*m))*np.sum(np.square(parameters['W' + str(l+1)]))
	cost += sum_val
	return cost

def linear_backward(dZ, linear_cache):
	A_prev, W, b = linear_cache
	m = A_prev.shape[1]
	dW = (1/m)*np.dot(dZ, A_prev.T)
	db = (1/m)*np.sum(dZ,axis=1,keepdims=True)
	dA_prev = np.dot(W.T, dZ)
	
	return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
	linear_cache, activation_cache = cache

	if activation == "relu":
		dZ = relu_backward(dA, activation_cache)
		dA_prev, dW, db = linear_backward(dZ, linear_cache)

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
	grads['dA' + str(L-1)], grads['dW' + str(L)], grads['db' + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")
	#print(grads['dW' + str(L)].shape)
	
	for l in reversed(range(L-1)):
		current_cache = caches[l] 
		grads['dA'+str(l)],grads['dW'+str(l+1)],grads['db'+str(l+1)]= linear_activation_backward(grads['dA'+str(l+1)],current_cache,"relu")
		#print(grads['dW' + str(L)].shape)

	return grads

def model_backward_dropout(AL, Y, caches, d, keep_probs):
	
	m = Y.shape[1]
	L = len(caches)
	grads = {}
	dAL = -np.divide(Y,AL) + np.divide(1-Y,1-AL) 
	current_cache = caches[L-1]
	grads['dA' + str(L-1)], grads['dW' + str(L)], grads['db' + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")
	
	for l in reversed(range(L-1)):
		current_cache = caches[l] 
		dA_drop = (grads['dA'+str(l+1)]*d[l+1])/keep_probs
		grads['dA'+str(l)],grads['dW'+str(l+1)],grads['db'+str(l+1)]= linear_activation_backward(dA_drop,current_cache,"relu")

	return grads

def gradDescent(parameters, grads,m, learning_rate=0.001,lambd=0):
	L = len(parameters) // 2
	if lambd != 0:
		for l in range(L):
			parameters['W' + str(l+1)] = (1 - (learning_rate*lambd/m))*parameters['W' + str(l+1)] -learning_rate*grads['dW' + str(l+1)]
			parameters['b' + str(l+1)] -= learning_rate*grads['db' + str(l+1)]
	else:
		for l in range(L):
			parameters['W' + str(l+1)] -= learning_rate*grads['dW' + str(l+1)]
			parameters['b' + str(l+1)] -= learning_rate*grads['db' + str(l+1)]
	return parameters

def predict(X,Y, parameters):
	AL,_ = model_forward(X, parameters)
	return (AL > 0.5)

def model(X, Y, layer_dims, learning_rate = 0.001, iterations = 200, lambd = 0, keep_probs=1, print_cost = True,modified_init=True):
	if modified_init == True:
		parameters = initParams_selected_var(layer_dims)
	else: 
		parameters = initParams_random(layer_dims)
	
	costs = []
	m = Y.shape[1]
	if keep_probs != 1:
		for i in range(iterations):
			AL,caches,d = model_forward_dropout(X,parameters,keep_probs)
			cost = compute_cost(AL, Y)
			grads = model_backward_dropout(AL, Y, caches, d, keep_probs)
			parameters = gradDescent(parameters, grads,m, learning_rate,lambd)
			if i%100 == 0 and print_cost == 1:
				print("Cost after %i iterations is %f" %(i,cost))
			if i%100 == 0:
				costs.append(cost)

	else: 
		for i in range(iterations):
			AL,caches = model_forward(X,parameters)
			if lambd == 0:
				cost = compute_cost(AL, Y)
			else:
				cost = compute_cost_reg(AL,Y,parameters,lambd)
			grads = model_backward(AL, Y, caches)
			parameters = gradDescent(parameters, grads,m, learning_rate,lambd)
			if i%100 == 0 and print_cost == 1:
				print("Cost after %i iterations is %f" %(i,cost))
			if i%100 == 0:
				costs.append(cost)

	plt.plot(np.squeeze(costs))
	plt.ylabel('cost')
	plt.xlabel('iterations (per tens)')
	plt.title('Learning rate = ' + str(learning_rate))
	plt.show()

	return parameters

#----------------------------------------------------------------------------------

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

#----------------------------------------------------------------------------------

#Main Code

layer_dims = [X_train.shape[0],11,1]
parameters = model(X_train, Y_train, layer_dims,0.003,7000,0,0.999,True,True)

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

