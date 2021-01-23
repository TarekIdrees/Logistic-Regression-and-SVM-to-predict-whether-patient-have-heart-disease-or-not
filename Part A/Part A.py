import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read data
path='heart.csv'
data=pd.read_csv(path)
data=data.sample(frac=1)
#Data Normalization
data['trestbps']=(data['trestbps']-data['trestbps'].min())/(data['trestbps'].max()-data['trestbps'].min())
data['chol']=(data['chol']-data['chol'].min())/(data['chol'].max()-data['chol'].min())
data['thalach']=(data['thalach']-data['thalach'].min())/(data['thalach'].max()-data['thalach'].min())
data['oldpeak']=(data['oldpeak']-data['oldpeak'].min())/(data['oldpeak'].max()-data['oldpeak'].min())

#adding a new column called ones before the data
data.insert(0,'Ones',1)

#number of row
#print(data.shape[0])
#number of col
#print(data.shape[1]) 


#seperate X (training data) from y (target variable )
cols=data.shape[1]

x_train=data.iloc[ :231 ,0:cols-1]
#print(x_train)
y_train=data.iloc[ :231,cols-1:cols]
#print(y_train)

x_test=data.iloc[231: ,0:cols-1]
#print(x_test)
y_test=data.iloc[231: ,cols-1:cols]
#print(y_test)


#Convert from data frames to numpy matrices
x_train=np.matrix(x_train.values)
#print(x_train)
x_test=np.matrix(x_test.values)

y_train=np.matrix(y_train.values)
#print(y_train)  
y_test=np.matrix(y_test.values)

theta=np.zeros(cols-1)
theta=np.matrix(theta)
#print(theta)

#divide the data into patient and healthy
data_test=data.iloc[:231,0:cols]
positive=data_test[data_test['target'].isin([1])]
#print(" have heart disease \n",positive)
negative=data_test[data_test['target'].isin([0])]
#print(" does not have heart disease \n",negative)


#plot the data
fig,ax=plt.subplots(figsize=(8,5))
ax.scatter(positive['trestbps'],positive['thalach'],s=40,c='b',marker='o',label='Patient')  
ax.scatter(negative['trestbps'],negative['thalach'],s=40,c='r',marker='x',label='healthy') 
ax.legend()
ax.set_xlabel('trestbps')
ax.set_ylabel('thalach')

def sigmoid(z):
    return 1/(1+np.exp(-z))

nums=np.arange(-10,10,step=1)

#plot sigmoid function
fig,ax=plt.subplots(figsize=(8,5))
ax.plot(nums,sigmoid(nums),'r')


def cost(theta,x,y):
    first=np.multiply(-y,np.log(sigmoid(x*theta.T))) #y=1
    second=np.multiply((1-y),np.log(1-sigmoid(x*theta.T))) #y=0
    return np.sum(first-second)/(len(x))

thiscost=cost(theta,x_train,y_train)
print("initial cost =",thiscost)


#initialize variables for learning rate and iterations
alpha=0.5
iters=1000

def gradient(theta,x,y,alpha,iters):
    temp=np.matrix(np.zeros(theta.shape))
    parameters=int(theta.ravel().shape[1])
    coste=np.zeros(iters)
    for i in range(iters):
        error=sigmoid(x*theta.T)-y
        for j in range(parameters):
            term= np.multiply(error,x[:,j])
            temp[0,j]=temp[0,j]-(alpha/len(x)*np.sum(term))
        theta=temp
        coste[i]=cost(theta,x,y)
        #print("Cost of iteration",i," = ",coste[i])
    return theta,coste


g,coste=gradient(theta,x_train,y_train, alpha, iters)

#draw error graph
fig,ax=plt.subplots(figsize=(7,5))
ax.plot(np.arange(iters),coste,'r')
ax.set_xlabel('Iteration')
ax.set_ylabel('Cost')
ax.set_title('Error vs Training Data')

print("Best theta of : ")
print(g)

#Calculate Y_predict 
def Calc_y_predict (theta,x_test):
    return (sigmoid(x_test*theta.T))

print("The value of Y_predict of X_test :")
print(Calc_y_predict(g, x_test))

def predict(theta,x):
    probability=sigmoid(x*theta.T)
    return [1 if x>=0.5 else 0 for x in probability]
theta_min=np.matrix(g)
prediction= predict(theta_min,x_test)
print("New Predict = ", prediction)

correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0
           for (a, b) in zip(prediction, y_test)]
accuracy = (sum(map(int, correct)) / len(correct) *100)
print(' accuracy={0} %'.format(accuracy))



