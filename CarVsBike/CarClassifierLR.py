import numpy as np
import matplotlib as plt
import glob
from PIL import Image


#Loading the dataset
#------------------------------------------------------------------------------------------------------------------------

#Loading the car dataset
filelist = glob.glob('CarDataset/*')
X_train1 = np.array([np.array(Image.open(fname)) for fname in filelist])
print("Training set positive image dataset size = " + str(X_train1.shape))
pos_egs = X_train1.shape[0]
Y_train1 = np.zeros((1,pos_egs)) + 1


#Loading the non-car dataset
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

#------------------------------------------------------------------------------------------------------------------------

#Function definitions---------

def sigmoid(z):
	#argument: Any numpy array or number input
	#return value: Numpy array of the same dim as input
	#Info: This function calculates the sigmoid of the input and returns it
	return 1/(1+np.exp(-z))

def initializeParams(n):
	#argument: The number of features
	#return values: A matrix consisting of parameters 'w' and 'b' initialized to zero values
	w = np.zeros((n,1))
	b = 0
	return w,b

def propagate(X,Y,w,b):
	#arguments: Feature matrix X, label vec Y, parameters w and b
	#return values: A dictionary consisting of values of dw and db
	Z = np.dot(w.T,X) + b
	A = sigmoid(Z)
	m = Y.shape[1]
	cost = -(1/m)*np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))
	dZ = A - Y
	dw = (1/m)*np.dot(X,dZ.T)
	db = (1/m)*np.sum(dZ)
	return {"dw" : dw, 
			"db" : db},cost

def gradDescent(X,Y,w,b,alpha,iterations,print_cost=False):
	#arguments: Feature mat X, label vec Y, parameters w and b, learning rate 'alpha' and number of iterations
	#return values: A dictionary containing the optimized parameters w and b respectively 
	for i in range (iterations):
		grads,cost = propagate(X,Y,w,b)
		dw = grads["dw"]
		db = grads["db"]
		w = w - alpha*dw
		b = b - alpha*db
		if(i%100 == 0 and i!=0  and print_cost == True):
			print("The training cost after " + str(i) + " iterations is " + str(cost))
	return {"w" : w,
			"b" : b}

def predict(x,w,b):
	#argumets: Feature vector x, parameters w and b
	#return value: 1 if predicted val is > 0.5 and 0 if <= 0.5
	z = np.dot(w.T,x) + b
	a = sigmoid(z)
	if a > 0.5 :
		return 1
	else: 
		return 0

#------------------------------------------------------------------------------------------------------------------------

#Model -------

w_init,b_init = initializeParams(X_train.shape[0])
learning_rate = 0.01
iterations = 4000
optimized_vals = gradDescent(X_train,Y_train,w_init,b_init,learning_rate,iterations,True)
w = optimized_vals["w"]
b = optimized_vals["b"]


#------------------------------------------------------------------------------------------------------------------------

#Calculating error on training set---------------

correct = 0
m = Y_train.shape[1]
for i in range (Y_train.shape[1]):
	if(predict(X_train[:,i].T, w, b) == Y_train[0,i]):
		correct += 1
percentage = (correct/m)*(100)
print("Training set accuracy: " + str(percentage) + "%")

#------------------------------------------------------------------------------------------------------------------------

#Loading test set--------------------------------------------------------


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
#Predicting results and printing error-----------------------------------

correct_test = 0
m_test = Y_test.shape[1]

for i in range (m_test):
	if(predict(X_test[:,i].T, w, b) == Y_test[0,i]) :
		correct_test += 1
	#print(str(predict(X_test[:,i].T, w, b)) + " " + str(Y_test[0,i]))
percentage_test = (correct_test/m_test)*100
print("Test set accuracy: " + str(percentage_test) + "%")


#------------------------------------------------------------------------

#User Image prediction
def user():
	print("Please place your file inside the UserInput folder")
	user = input("Please enter the filename: ") 
	file2 = 'UserInput/' + str(user)
	x_user_orig = np.array([np.array(Image.open(file2))]) 
	print(x_user_orig.shape)
	x_user = x_user_orig.reshape((x_user_orig.shape[1]*x_user_orig.shape[2]*x_user_orig.shape[3],1))
	x_user = x_user/255

	if(predict(x_user, w, b) == 1) :
		print("The classifier predicts it to be a car !")
	else:
		print("The classifier predicts that it is not a car...")

is_true = int(input("If you want to test on your images please input '1' else input '0': "))
if(is_true == 1):
	user()
