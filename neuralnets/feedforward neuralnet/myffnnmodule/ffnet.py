# feedforward neural network
from numpy import minimum
from numpy.lib import math
from .activations import *

from .weightinit import KaimingHe
from .weightinit import *

def max_probability(y_hat):
    return np.argmax(y_hat, 0)

def average_accuracy(Y_hat, Y):
    return np.sum(Y_hat == Y) / Y.size

class layer:

    def __init__(self,num_inputs,num_outputs,activate_layer=ReLu,weight_initializer_in=KaimingHe,output_weight_counts=0):
        
        # random weight initialization
        # self.W = np.random.rand(num_outputs,num_inputs)-0.5
        # self.b = np.random.rand(num_outputs,1)-0.5
        
        # UNIFORM KAIMING HE RANDOM INITIALIZATION - best used with ReLu activation function
        minimum, maximum = weight_initializer_in.uniform(input_weight_count=num_inputs,output_weight_count=output_weight_counts)

        self.W = np.random.uniform(minimum,maximum,(num_outputs,num_inputs))
        self.b = np.random.uniform(minimum,maximum,(num_outputs,1))
        
        self.activation_class = activate_layer
    
    def forward(self,inputs):
        self.X = inputs
        self.Z = np.dot(self.W,self.X) + self.b
        self.A = self.activation_class.activate(self.Z)

    def derivative_output(self,y_hot_target,m):
        self.dZ = self.A-y_hot_target
        self.dW = (1/m) * self.dZ.dot(self.X.T)
        self.db = (1/m) * np.sum(self.dZ,axis=1,keepdims=True)

    def derivative_hidden(self,next_weights,next_dZ,m):
        self.dZ = next_weights.T.dot(next_dZ) * self.activation_class.derivative(self.Z)
        self.dW = (1/m) * self.dZ.dot(self.X.T)
        self.db = (1/m) * np.sum(self.dZ,axis=1,keepdims=True)

class nnet:

    def __init__(self,input_count,output_class_count):
        self.input_count = input_count
        self.classification_count = output_class_count
        self.layers = None
        self.layerCount = 0

    def one_hot(self,Y):
        one_hot_Y = np.zeros((Y.size, self.classification_count))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y
        
    def addLayer(self,num_outputs,activate=ReLu,weight_initializer=KaimingHe,output_weight_counts=0):
        if self.layerCount == 0:
            temp = layer(self.input_count,num_outputs,activate_layer=activate,weight_initializer_in=weight_initializer,output_weight_counts=output_weight_counts)
            self.layers = [temp]
        else:
            temp = layer(self.layers[self.layerCount-1].b.shape[0],num_outputs,activate_layer=activate,weight_initializer_in=weight_initializer,output_weight_counts=output_weight_counts)
            self.layers.append(temp)
        self.layerCount+=1

    def feedForward(self,X):
        for i in range(self.layerCount):
            if i==0:
                self.layers[i].forward(X)
            elif i==self.layerCount-1:
                self.layers[i].forward(self.layers[i-1].A)
            else:
                self.layers[i].forward(self.layers[i-1].A)
    
    def feedForward_output(self,X):
        for i in range(self.layerCount):
            if i==0:
                self.layers[i].forward(X)
            elif i==self.layerCount-1:
                self.layers[i].forward(self.layers[i-1].A)
            else:
                self.layers[i].forward(self.layers[i-1].A)
        
        maxval = -1
        maxindex = -1
        for i in range(self.layers[-1].A.shape[0]):
            if self.layers[-1].A[i,0] > maxval:
                maxval = self.layers[-1].A[i,0]
                maxindex = i
                
        return maxindex

    def backPropagate(self,Y,m):
        last_index = self.layerCount-1
        Y_hot = self.one_hot(Y)

        for i in range(self.layerCount):
            if i==0:
                self.layers[last_index-i].derivative_output(Y_hot,m)
            else:
                self.layers[last_index-i].derivative_hidden(
                    self.layers[last_index-i+1].W,
                    self.layers[last_index-i+1].dZ,m)

    def updateWb(self,alpha):
        for i in range(self.layerCount):
            self.layers[i].W = self.layers[i].W - (alpha*self.layers[i].dW)
            self.layers[i].b = self.layers[i].b - (alpha*self.layers[i].db)

    def gradient_descent(self,X,Y_target_labels,alpha=0.01,epoch=100):
        m = X.shape[1]
        print('\n---- Gradient descent ---------------------------------------------------')
        print('X input shape = \n',X.shape)
        print('\nY_target_labels shape input = \n',Y_target_labels.shape)
        print('-------------------------------------------------------------------------')
        if self.classification_count != self.layers[self.layerCount-1].b.shape[0]:
            last_layer_count = self.layers[self.layerCount-1].b.shape[0]
            print('Error ffnet.py : your last layers output count does not match your initialized class count')
            print('initialized class count : ',self.classification_count)
            print('last layer output count : ',self.layers[self.layerCount-1].b.shape[0])
            exit(1)
        for i in range(epoch):
            self.feedForward(X)
            self.backPropagate(Y_target_labels,m)
            self.updateWb(alpha)
            if i % 10 == 0:
                print("\nepoch : ", i)
                accuracy = self.network_accuracy(Y_target_labels)*100
                print('Neural network accuracy of : ',round(accuracy,4),'%',sep='')

    def network_accuracy(self,y_target_labels):
        y_hat_labeled = max_probability(self.layers[self.layerCount-1].A)
        return average_accuracy(y_hat_labeled, y_target_labels)
        
    def network_shapes(self):
        for i in range(self.layerCount):
            print('\nShapes of Layer ',(i+1))
            print('\tW[',i+1,'] = ',self.layers[i].W.shape)
            print('\tb[',i+1,'] = ',self.layers[i].b.shape)