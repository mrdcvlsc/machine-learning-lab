# Multi Linear Regression - this is an example with n numbers of independent variables

import numpy as np
import math
import matplotlib.pyplot as plt

class model_mlr:

    def __init__(self,B=None):
        self.__B = B

    def regress(self,x):
        y_hat = self.__B[0][0]
        for i in range(len(x)):
            y_hat = y_hat + (self.__B[0][i+1]*x[i])
        return y_hat
    
    def train(self,Xij_data,Yi_data):
        
        # function to get the beta coefficients
        
        inner = X.T.dot(X) # get the X transpose multiplied by X (X^T*X)

        inner_inverse = np.linalg.inv(inner) # get the inverse of the result of X transpose multiplied by X (X^T*X)^-1

        inverse_mul_XT = inner_inverse.dot(X.T) # multiply the result again to X transpose ((X^T*X)^-1)*X^T

        b_hat = inverse_mul_XT.dot(y.T) # then multiply it to the y observed values
        # if your 'y' is a row form '1xN' matrix, just transpose it to become a 'Nx1' column matrix

        # and that is all of your coefficients
        # now make the b_hat from Nx1 to 1xN matrix form(this is just optional)
        self.__B = b_hat.T

        # B_hat = (X^T*X)^-1*X^T*y

    def status(self):
        print('Beta Coefficients :')
        for i in range(len(self.__B[0])):
            print('B',i,' = ',self.__B[0][i],sep='')

if __name__ == '__main__':
    
    test1 = model_mlr()
    
    y =  np.array([[-3.7,3.5,2.5,11.5,5.7]]) # dependent varaible
    
    x1 = [3,4,5,6,2] # independent variable 1
    x2 = [8,5,7,3,1] # independent variable 2
    x0_ones = np.full(len(x1),1) # add an array of ones as independent variable 0, to address the B0 coefficient

    # the X matrix should have each independent variables as columns, see the result below
    X = np.array([x0_ones,x1,x2])
    X = X.T # we transpose here to make our x variables align as columns not by rows

    print('X : \n',X)
    print('y : \n',y)

    test1.train(X,y)

    test1.status()

    print('Test : ',test1.regress([3,8]))
