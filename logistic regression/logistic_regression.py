# logistic regression
# sample data from - https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

from numpy.lib.npyio import save
import pandas as pd
import numpy as np
import math
from io import StringIO

def sigmoid(x):
    return 1/(1+np.exp(-x))

class logistic_regression:

    def __init__(self,bias = 0, weights = [], accuraccy_rate = 0.0):
        self.bias = bias
        self.weights = weights
        self.accuracy_rate = accuraccy_rate

    def load_weights(self,weightfile,biasfile):
        txtbias = open(biasfile,'r')
        rowver = np.loadtxt(weightfile)
        self.weights = np.array([rowver]).T
        self.bias = float(txtbias.read())

        
    def save_weigths(self,weightfile,biasfile):
        txtbias = open(biasfile,'w')
        np.savetxt(weightfile,self.weights)
        txtbias.write(str(self.bias))
        txtbias.close()


    def fit(self,Y,X,alpha=0,epoch=100):

        m = X.shape[1] # numbers of independent variables/features(x)
        n = Y.shape[0] # numbers of samples

        # bias and weight values are default to zero
        self.weights = np.zeros((1,m)).T
        self.bias = 0

        for i in range(epoch+1):

            Y_predict = sigmoid(np.dot(X,self.weights)+self.bias) # hypotesis/prediction

            # derivatives of weights and biases in respect to the cost function
            dw = 1/n*np.dot(X.T, (Y_predict-Y))
            db = 1/n*np.sum(Y_predict-Y)

            # gradient descent, weight and bias updates
            self.weights -= alpha*dw
            self.bias -= alpha*db

        self.accuracy_rate = 1-(np.sum(np.absolute(Y_predict-Y))/Y.shape[0])


    def predict(self,x):

        if(len(x)!=self.weights.shape[0]):
            print('Error X features did not match the numbers of weights')
            exit()
        
        return round(sigmoid(np.sum(np.dot(x,self.weights))+self.bias),5)

    def status(self):
        print('--------------------------')
        print('Bias = ',self.bias)
        print('Weights : \n',self.weights)
        

if __name__ == '__main__':

    # 1.) formating, cleaning and extracting only the necessary data
    # THERE IS NO NORMALIZATION APPLIED TO THE DATASET
    file  = pd.read_csv('wdbc.csv',header=None)
    BreastCancerWisconsinDataSet = file.drop([0,5,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31], axis=1) # drop some columns


    # 2.) get the dependent variables
    diagnosis = []

    for i in range(BreastCancerWisconsinDataSet.shape[0]):

        if(BreastCancerWisconsinDataSet[1][i]=='M'):
            diagnosis.append(1)
        else:
            diagnosis.append(0)

    Y = np.array([diagnosis],dtype=np.float128).T


    # 3.) get the independent 
    BreastCancerWisconsinDataSet = BreastCancerWisconsinDataSet.drop([1],axis=1)
    X = np.array(BreastCancerWisconsinDataSet,dtype=np.float128)

    # 4.) create an instance of the model
    wdbc_classifier = logistic_regression()

    # 5.) train the model or load a trained weights into the models
    # wdbc_classifier.load_weights('w_sample1.txt','b_sample1.txt')  # a trained weights from "alpha=0.00125,epoch=40000000"
    wdbc_classifier.fit(Y,X,alpha=0.005,epoch=100000)
    wdbc_classifier.save_weigths('w_sample.txt','b_sample.txt')  # if you want to save the weights and bias after training
    # 0.0003

    # 6.) (optional) - displays the trained weights an bias of the model
    # wdbc_classifier.status()

    # 7.) (optional) - display sample prediction of the first 25 records
    # for i in range(25):
    #     y_pred = wdbc_classifier.predict(X[i])
    #     print('row ',i+1,' = ',y_pred,sep='',end='')
    #     if y_pred >= 0.5:
    #         print(' = Malignant')
    #     else:
    #         print(' = Benign')
    
    # 8.) accuraccy for loaded weights
    # Y_predict = sigmoid(np.dot(X,wdbc_classifier.weights)+wdbc_classifier.bias)
    # acc = 1-(np.sum(np.absolute(Y_predict-Y))/Y.shape[0])
    # print("Model Accuracy = ",acc*100)

    # 8.) accuraccy after training
    print('Accuracy : ',wdbc_classifier.accuracy_rate*100)