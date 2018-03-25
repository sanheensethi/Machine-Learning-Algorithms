from numpy import * 
from matplotlib import pyplot as plt

#Getting data from file and spliting it into x axis and y axis
def getdata(datafile,delimiter_in_file,header_skiping):
    training_data = genfromtxt(datafile,delimiter=delimiter_in_file,skip_header = header_skiping,dtype=str)
    x = [] # x-axis data
    y = [] # y-axis data
    for i in range(0,len(training_data)):
        x.append(float(training_data[i][0]))
        y.append(float(training_data[i][1]))
    return x,y


#Error Finder
#formula : 1/2n{summition from i=1 to n (y[i] - (m * x[i] + b))}
def error(m,b,x,y):
    total_error = 0.0
    for i in range(0,len(x)):
        total_error+= ((m * x[i] + b) - y[i]) ** 2
    return total_error/float(len(x))

#Function by which we can get new values of m and b
#Using gradient descent ( Partial Derivative )
#Having Formula : 
# curl/curl{m} = {summition from i = 1 to n of -x[i] * (y[i] - (m * x[i] + b))}
# curl/curl{b} = {summition from i = 1 to n of -(y[i] - (m * x[i] + b))}
def step_gradient(m,b,x,y,learning_rate):
    m_gradient = 0.0 #initial gradient
    b_gradient = 0.0 #initial intercept
    for i in range(0,len(x)):
        m_gradient+=(-x[i] * (y[i]- (m * x[i]) + b))
        b_gradient+=(-(y[i] - ((m * x[i]) + b)))
    #new m using formula of gradient descent
    #new path point = old path point - (alpha) * gradient
    #alpha = learning rate , as what the speed at which we go downward in graph
    #to find minima
    new_m = m - (learning_rate * (m_gradient)/float(len(x)))
    new_b = b - (learning_rate * (b_gradient)/float(len(x)))
    return new_m,new_b

#running gradient descent upto num of iteration say 10,000 time we get 
#different m and b and after last we found m and b 
def gradient_descent_run(old_m,old_b,num_iterations,x,y,learning_rate):
    m = old_m # getting old b
    b = old_b # getting old m
    for i in range(num_iterations):
        #calling step gradient upto num iterations time
        #in this m and b get change again and again upto num iterations
        m,b = step_gradient(m,b,x,y,learning_rate)
    return m,b

#Test file data to run Test on calculated final m and b
#and predict the y
def test(m,b,x):
    y_new = [] #creating empty list of y_new
    print()
    for i in range(0,len(x)):
        #appending y_new list with y_new = mx+b have x input and y output
        y_new.append(m * x[i] + b)
    return y_new 

    
def train(datafile,delimiter_in_file,skiping_header,num,alpha):
    
    #Getting X and Y values from File of training data
    x,y = getdata(datafile,delimiter_in_file,skiping_header)
    m_initial = 0 #initial slope
    b_initial = 0 #initial intercept
    num_iterations = num #num of iterations
    learning_rate = alpha #learning rate
    print()
    print('Training Data Scatter Plot')
    plot_graph(x,y) #plotting Training Data
    print()
    print('Starting Linear Regression with m = '+str(m_initial)+' b = '+str(b_initial)+' error = ' + str(error(m_initial,b_initial,x,y)))
    print('Starting...')
    print()
    m,b = gradient_descent_run(m_initial,b_initial,num_iterations,x,y,learning_rate)
    print()
    print('Ending Linear Regression with m = '+str(m)+' b = '+str(b)+' error = ' + str(error(m,b,x,y)))
    print()
    print('Plotting Best Fit Line')
    plot_final_line(m,b) #Plotting Best Fit Line
    return m,b,error(m,b,x,y)

#plotting Scatter Graph Function
def plot_graph(x,y):
    print('Scatter Plot')
    plt.scatter(x,y)
    plt.show()

#plotting Best Fit Line Function
def plot_final_line(m,b):
    print('Printing Best fit Line')
    x = []
    y = []
    for i in range(0,100):
        x.append(float(i))
    for i in range(0,len(x)):
        y.append(m * x[i] + b)
    plt.plot(x,y)
    plt.show()

def run(trainfile,testfile,delimiter,skip_header,num,alpha):
    #getting m , b ,error from training data main function
    m,b,e = train(trainfile,delimiter,skip_header,num,alpha)
    #getting x and y from test data file
    x,y = getdata(testfile,delimiter,skip_header)
    #plotting graph of actual dataset in test
    print('Ploting actual Data from test file')
    plot_graph(x,y)
    print()
    #Calculating Prediction of y
    y_new = test(m,b,x)
    print()
    #plotting graph of Predicted y with actual x 
    print('Ploting Predicted data of Y ')
    plot_graph(x,y_new)
    print()
    #printing y actual and corresponding to its y_new
    for i in range(0,len(y)):
        print('Y actual = ' + str(y[i]),end = ' | ')
        print('Y Predicted = ' + str(y_new[i]))
    print()
    print('Ending Linear Regression with m = '+str(m)+' b = '+str(b)+' error = ' + str(e))
    print()

# --------------Main---------------- #

run('data/train.csv','data/test.csv',',',1,5000,0.0001)
print()
print('Sanheen Sethi')