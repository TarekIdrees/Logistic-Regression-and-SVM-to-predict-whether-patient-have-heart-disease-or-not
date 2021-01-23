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
data['age']=(data['age']-data['age'].min())/(data['age'].max()-data['age'].min())
data['oldpeak']=(data['oldpeak']-data['oldpeak'].min())/(data['oldpeak'].max()-data['oldpeak'].min())
data['thal']=(data['thal']-data['thal'].min())/(data['thal'].max()-data['thal'].min())
data['ca']=(data['ca']-data['ca'].min())/(data['ca'].max()-data['ca'].min())
data['slope']=(data['slope']-data['slope'].min())/(data['slope'].max()-data['slope'].min())
data['restecg']=(data['restecg']-data['restecg'].min())/(data['restecg'].max()-data['restecg'].min())
data['cp']=(data['cp']-data['cp'].min())/(data['cp'].max()-data['cp'].min())


#adding a new column called ones before the data
data.insert(0,'Ones',1)

#number of row
#print(data.shape[0])
#number of col
#print(data.shape[1]) 


#seperate X (training data) from y (target variable )
cols=data.shape[1]   
################## All Coloms #########################
x_train_all=data.iloc[ :231 ,0:cols-1]
#print(x_train_all)
y_train_all=data.iloc[ :231,cols-1:cols]
#print(y_train_all)

x_test_all=data.iloc[231: ,0:cols-1]
#print(x_test_all)
y_test_all=data.iloc[231: ,cols-1:cols]
#print(y_test_all)


#Convert from data frames to numpy matrices
x_train_all=np.matrix(x_train_all.values)
#print(x_train_all)
x_test_all=np.matrix(x_test_all.values)

y_train_all=np.matrix(y_train_all.values)
#print(y_train_all)  
y_test_all=np.matrix(y_test_all.values)

w_all=np.zeros(x_train_all.shape[1])
w_all=np.matrix(w_all)

################# Three Coloms #########################
x_train_Three=data.iloc[ :231 ,0:4]
#print(x_train_Three)
y_train_Three=data.iloc[ :231,cols-1:cols]
#print(y_train_Three)

x_test_Three=data.iloc[231: ,0:4]
#print(x_test_Three)
y_test_Three=data.iloc[231: ,cols-1:cols]
#print(y_test_Three)


#Convert from data frames to numpy matrices
x_train_Three=np.matrix(x_train_Three.values)
#print(x_train_Three)
x_test_Three=np.matrix(x_test_Three.values)

y_train_Three=np.matrix(y_train_Three.values)
#print(y_train_Three)  
y_test_Three=np.matrix(y_test_Three.values)

w_Three=np.zeros(x_train_Three.shape[1])
w_Three=np.matrix(w_Three)

alpha=0.01
iteraion=1000
lamda=1/iteraion



def GradientDescent(X,y,w,alpha,lamda,iteration):
        for i in range (iteration):
            for j in range(len(X)):
                    if (y[j] * np.dot(w,X[j].T)) < 1: 
                       w = w + alpha * (np.dot(y[j] ,X[j]) + (-2 * (lamda * w)))
                    else:
                      w = w - alpha * (2 * lamda * w)
        return w

                
def predict(x_test,w):
    output=np.dot(w, x_test.T)  
    return np.sign(output)
    
    


def accurcy(w,x_testing,y_testing):
    counter=0
    for i in range(len(x_testing)):
        if (y_testing[i] * np.dot(w,x_testing[i].T)) >= 1:
            counter+=1
    return (counter/len(x_testing))*100


##################### Y_predict of All Coloms ######################
w_aft_all=GradientDescent(x_train_all,y_train_all,w_all,alpha,lamda,iteraion)
print("Best W of X_all after Gradient Descent :")
print(w_aft_all)
print("") 

print("predicted Y  of X_all : ")
predict_val_all=predict(x_test_all,w_aft_all)
print(predict_val_all)


print("Accuracy of X_All= ")
print(accurcy(w_aft_all,x_test_all,y_test_all))


##################### Y_predict of Three ######################
w_aft_Three=GradientDescent(x_train_Three,y_train_Three,w_Three,alpha,lamda,iteraion)
print("Best W of X_Three after Gradient Descent :")
print(w_aft_Three)
print("") 

print("predicted Y  of X_Three : ")
predict_val_Three=predict(x_test_Three,w_aft_Three)
print(predict_val_Three)

print("Accuracy of X_Three = ")
print(accurcy(w_aft_Three,x_test_Three,y_test_Three))