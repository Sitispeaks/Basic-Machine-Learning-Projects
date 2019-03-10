import numpy as np
import glob
from PIL import Image

#------------------------------------------------------------------------------------------

#Loading the car dataset
filelist = glob.glob('CarDataset/*')
X_train1 = np.array([np.array(Image.open(fname)) for fname in filelist])
print("Training set positive image dataset size = " + str(X_train1.shape))
pos_egs = X_train1.shape[0]
Y_train1 = np.zeros((1,pos_egs)) + 1


#Loading the bike dataset
filenames = glob.glob('NonCarDataset/*')
X_train2 = np.array([np.array(Image.open(fname)) for fname in filenames])
neg_egs = X_train2.shape[0]
Y_train2 = np.zeros((1,neg_egs))
print("Training set negative image dataset size = " + str(X_train2.shape))


#Appending them
X_train_orig = np.concatenate((X_train1,X_train2),axis = 0)
Y_train = np.concatenate((Y_train1,Y_train2),axis = 1)
X_train = X_train_orig.reshape((X_train_orig.shape[1]*X_train_orig.shape[2]*X_train_orig.shape[3],X_train_orig.shape[0]))
X_train = X_train/255


print("Size of the Training set: " + str(X_train.shape))
print("Size of label vector: " +str(Y_train.shape))

#------------------------------------------------------------------------------------------

#Loading the test set

filelist2 = glob.glob('CarTest/*')
X_test1 = np.array([np.array(Image.open(fname)) for fname in filelist2])
Y_test1 = np.zeros((1,X_test1.shape[0]))+1

filelist3 = glob.glob('NonCarTest/*')
X_test2 = np.array([np.array(Image.open(fname)) for fname in filelist3])
Y_test2 = np.zeros((1,X_test2.shape[0])) 

X_test_orig = np.concatenate((X_test1,X_test2),axis=0)
X_test = X_test_orig.reshape((X_test_orig.shape[1]*X_test_orig.shape[2]*X_test_orig.shape[3],X_test_orig.shape[0]))
X_test = X_test/255 
Y_test = np.concatenate((Y_test1,Y_test2),axis=1)

print("Size of test set feature matrix: " + str(X_test.shape))
print("Size of test set labels vector: " + str(Y_test.shape)) 

#------------------------------------------------------------------------------------------

#Function definitions

def sigmoid(z):
	return 1/(1 + np.exp(-z))
	
def relu(z):
	return np.maximum(z,0) 

def layer_sizes(X,Y):
	n_x = X.shape[0]
	n_y = Y.shape[0]
	return n_x,n_y

def initParams(n_x,n_h,n_y):
	W1 = np.random.randn(n_h,n_x)*0.1
	b1 = np.zeros((n_h,1))
	W2 = np.random.randn(n_y,n_h)*0.1
	b2 = np.zeros((n_y,1))
	
	parameters = {"W1" : W1,
				  "b1" : b1,
				  "W2" : W2,
				  "b2" : b2}
	return parameters

def forward(X, parameters):
	W1 = parameters['W1']
	b1 = parameters['b1']
	W2 = parameters['W2']
	b2 = parameters['b2']

	Z1 = np.dot(W1,X) + b1
	A1 = relu(Z1)
	Z2 = np.dot(W2,A1) + b2
	A2 = sigmoid(Z2)

	cache = {"Z1" : Z1,
			 "A1" : A1,
			 "Z2" : Z2,
			 "A2" : A2}
	return cache

def compute_cost(A2, Y, parameters):
	m = Y.shape[1]
	cost = -(1/m)*np.sum(Y*np.log(A2) + (1-Y)*np.log(1-A2)) 
	return cost

def relu_back(A):
	mat = np.maximum(A,0)
	return (mat >= 0) 

def backward(parameters, cache, X, Y):
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

	dZ1 = np.dot(W2.T,dZ2)*relu_back(A1)
	dW1 = (1/m)*np.dot(dZ1,X.T)
	db1 = (1/m)*np.sum(dZ1,axis=1,keepdims=True)
	
	grads = {"dW1" : dW1,
			 "db1" : db1,
			 "dW2" : dW2,
			 "db2" : db2}
	return grads
	 
def gradDescent(parameters, grads, learning_rate = 0.01):
	
	for l in range(1,3):
		parameters['W' + str(l)] -= learning_rate*grads['dW' + str(l)]
		parameters['b' + str(l)] -= learning_rate*grads['db' + str(l)] 
	return parameters

def predict(parameters, X):
	cache = forward(X, parameters)
	A2 = cache['A2']
	return (A2 > 0.5)

#------------------------------------------------------------------------------------------

#Model

def model(X, Y, n_h, iterations = 200,learning_rate=0.01,print_cost=True):
	n_x,n_y = layer_sizes(X,Y)
	
	parameters = initParams(n_x,n_h,n_y)
	for i in range(iterations):
		cache = forward(X, parameters)
		cost = compute_cost(cache['A2'], Y, parameters)
		if print_cost == True and i%200 == 0 :
			print("cost after " + str(i) + " iterations is " + str(cost))
		grads = backward(parameters, cache, X, Y)
		parameters = gradDescent(parameters, grads, learning_rate)
	return parameters

#------------------------------------------------------------------------------------------
best_acc = 0
#best_params = model(X_train,Y_train,11,8000,0.06,0)
for j in range(1):
#Training code and training set predictions

	parameters = model(X_train, Y_train,11, 6000,0.06,1)

	predictions = predict(parameters, X_train)
	correct_train = np.sum(predictions == Y_train) 
	m = Y_train.shape[1]
	accuracy_train = (correct_train/m)*100
	print("Training accuracy = " + str(accuracy_train) + "%")

#Test predictions

	predictions_test = predict(parameters, X_test)
	correct_test = np.sum(predictions_test == Y_test) 
	m = Y_test.shape[1]
	accuracy_test = (correct_test/m)*100
	print("Test accuracy = " + str(accuracy_test) + "%")
	if best_acc < accuracy_test:
		best_params = parameters
		best_acc = accuracy_test

print("Best accuracy on test set is " + str(best_acc))
np.set_printoptions(threshold=np.nan)
layers = 2
output_files = {}
if best_acc >= 70:
	for l in range(1,layers+1):
		output_files['W' + str(l)] = open('SNN/W'+str(l),"w+")
		output_files['b' + str(l)] = open('SNN/b'+str(l),"w+")
		(output_files['W' + str(l)]).write(str(best_params['W' + str(l)]))
		(output_files['b' + str(l)]).write(str(best_params['b' + str(l)]))
	aa = open('SNN/accuracy',"w+")
	aa.write("Best accuracy = " + str(best_acc) + "%")
		 
