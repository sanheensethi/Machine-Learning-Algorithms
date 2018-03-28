import numpy as np
from time import sleep
#from matplotlib import pyplot as plt
#%matplotlib inline

#Getting Data From File
def getdata(datafile,delimiter_in_file,header_skiping):
    training_data = np.genfromtxt(datafile,delimiter=delimiter_in_file,skip_header = header_skiping)
    x = [] # x-axis data
    y = [] # y-axis data
    features = 2
    for i in range(0,len(training_data)):
        x.append(float(training_data[i][0]))
        y.append(float(training_data[i][1]))
    return x,y

#Getting X Vectors 
def getXvectors(x):
    x0 = np.ones(len(x)) # x,not = [1,1,1,..]
    x1 = x # x1 = [x[0],x[1],x[2]...]
    return np.array([x0,x1]) # return matrix as [[1,1,1...],
      											#[x[0],x[1],x[2]...]]
#Initialing B vectors
def initial_B_vectors(features):
    return np.zeros(features).T # [[0,0,0...]] with transporse and without it the zeros are one below one

#Getting Error
def error(y,new_y):
    return np.sum((new_y-y)**2)/(2 * len(y))

#Gradient Descent Formula
def step_gradient(X,Y,B,alpha):
    Y_new = new_y(B,X)
    loss = Y_new - Y
    gradient = (X.dot(loss)/float(len(Y)))
    B = B - alpha * gradient
    return B
    
#Gradient Descent Runner Upto num Iterations
def gradient_descent_runner(X,Y,B,num,alpha):
    for i in range(num):
        #tracking no. of Iterations
        if (i > 1 and i < 100):
            if i%10 == 0:
                print(i)
        elif (i > 100 and i < 1000):
            if i%100 == 0:
                print(i)
        elif (i > 1000):
            if i%1000 == 0:
                print(i)
                sleep(1)
        # tracking ends
        B = step_gradient(X,Y,B,alpha)
        print(' B ',end = ' = ')
        print(B)
    return B
        
#Getting New y or Predicted y with B and X
def new_y(B,X):
    return np.dot(B,X)

#main Function
def run(file,nooffeatures,delimiter,skip_header,numofiterations,learning_rate):
    x,y = getdata(file,delimiter,skip_header)
    X = getXvectors(x) # x vectors / matrices
    B = initial_B_vectors(nooffeatures) # thetas
    Y = new_y(B,X)
    e = error(y,Y)
    print()
    print('Initial Error : ' + str(e))
    print()
    print('Running Gradient Descent Runner')
    print('Starting...')
    B = gradient_descent_runner(X,y,B,numofiterations,learning_rate)
    print('New B ',end = " = ")
    print(B)
    Y = new_y(B,X)
    e = error(y,Y)
    print()
    print('Final Error : ' + str(e))
    print()


# ----------- Main ------------ #

run('data/train.csv',2,',',1,10000,0.0001)
print()
print('Sanheen Sethi')