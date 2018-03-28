import numpy as np
from time import sleep

def getdata(file,delimiter,skip_header):
    data = np.genfromtxt(file,delimiter = delimiter,skip_header = skip_header,dtype = str)
    x1 = []
    x2 = []
    y = []
    for row in data:
        x1.append(float(row[0]))
        x2.append(float(row[1]))
        y.append(float(row[2]))
    return x1,x2,y

def getXvectors(x1,x2):
    x0 = np.ones(len(x1))
    return np.array([x0,x1,x2])

def getBvectors(parameters):
    return np.zeros(parameters).T

def new_y(B,X):
    return np.dot(B,X)
    
def error(new_y,y):
    return np.sum((new_y - y) ** 2)/( 2 * float(len(y)))

def step_gradient(X,Y,B,learning_rate):
    Y_new = new_y(B,X)
    loss = Y_new - Y
    gradient = (X.dot(loss)/float(len(Y)))
    B = B - learning_rate * gradient
    return B

def gradient_descent_runner(X,Y,B,num,alpha):
    print('Tracking Iterations B and Error...')
    for i in range(num):
        if(i%1000==0):
            print(i)
            print('B',end=' = ')
            print(B)
            print('E',end=' = ')
            ynew = new_y(B,X)
            print(error(ynew,Y))
        B = step_gradient(X,Y,B,alpha)
    return B

def predicty(x1,x2,B):
    x0 = 1
    return B[0] * x0 + B[1] * x1 + B[2] * x2
    
print('Satrting Multiple Regression')
print('Running...')
print()
print('Getting Data...')
x1,x2,y = getdata('student.csv',',',1)
print()
print('Getting X Vectors...')
X = getXvectors(x1,x2)
print()
print('Printing X Vectors')
print(X)
print()
print('Getting B Vectors')
B = getBvectors(3)
print()
print('Printing B vectors')
print(B)
Y = new_y(B,X)
print()
print('Calculating Initial Error...')
print()
e_ini = error(Y,y)
print('Running Gradient Descent...')
print()
print('Running...')
print()
B = gradient_descent_runner(X,y,B,10000,0.0001)
print()
print('Gradient Descent Ends...')
print()
Y = new_y(B,X)
print()
print('Calculating Final Error...')
e_end = error(Y,y)
print()
print()
print('Final Values...')
print('B',end=' = ')
print(B)

print()
print('Initial Error :'+str(e_ini))
print('Ending Error :'+str(e_end))
print()
print('Prediction of Y having x1 = 48 and x2 = 68')
print()
print(predicty(48,68,B))
print()
print('Multiple Regression Ends...')