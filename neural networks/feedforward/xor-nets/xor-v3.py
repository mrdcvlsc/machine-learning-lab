# dynamic backpropagation implementation

import time
import math
import numpy as np
from numpy._typing import NDArray

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

sigmoid_v = np.vectorize(sigmoid)

def sigmoid_dx(x):
    return sigmoid(x) * (1 - sigmoid(x))

sigmoid_dx_v = np.vectorize(sigmoid_dx)

def cost(y_pred, y_target):
    return (y_pred - y_target)**2

cost_v = np.vectorize(cost)

def cost_dy_pred(y_pred, y_target):
    return 2*(y_pred - y_target)

cost_dy_pred_v = np.vectorize(cost_dy_pred)

class xor_net:
    def __init__(self, layers: list[int], alpha: float):
        self.alpha = alpha
        self.layers = layers
        
        # just initializing a list of size len(layers) that contains empty numpy arrays at the begining.
        # this list's elements will contain the collection of the resulting weighted sums of the inputs
        # for a current layer before activation. when filled the elements will be an N x 1 numpy array.
        self.Z : list[NDArray[np.float64]] = []
        self.DX_Z : list[NDArray[np.float64]] = [] # this is for the partial derivatives
        i = 1
        while i < len(layers):
            self.Z.append(np.zeros((layers[i], 1)))
            self.DX_Z.append(np.zeros((layers[i], 1)))
            i += 1
        
        # same as self.Z (N x 1 in size) - this list will contain the collection of activated nodes
        # with the sole exception of the very first element which is just the input layer that does
        # not have any activation, but should be normalized.
        self.Y : list[NDArray[np.float64]] = []
        self.DX_Y : list[NDArray[np.float64]] = []
        i = 0
        while i < len(layers):
            self.Y.append(np.zeros((layers[i], 1)))
            self.DX_Y.append(np.zeros((layers[i], 1)))
            i += 1

        # initialize random weight (but for this example we will actually edit the weights to some specific values)
        self.W = []     
        self.DX_W = []     
        i = 0
        while i < len(layers) - 1:
            self.W.append(
                np.random.uniform(low=0, high=1, size=(layers[i + 1], layers[i]))
            )
            self.DX_W.append(
                np.zeros((layers[i + 1], layers[i]))
            )
            i += 1

        self.training_input = np.array([[0, 0], [0, 1] , [1, 0], [1, 1]])
        self.training_output = np.array([0, 1, 1, 0])

    def feedforward(self, x1, x2):
        self.Y[0][0][0] = x1
        self.Y[0][1][0] = x2

        l = 0
        while l < len(self.layers) - 1:
            np.dot(self.W[l], self.Y[l], out=self.Z[l])
            self.Y[l + 1] = sigmoid_v(self.Z[l])
            l += 1

            # note:
            #   self.Y[l] is the input for the current layer
            #   self.Z[l] is the weighted output of the current layer
            #   self.Y[l + 1] is the activated output of the current layer

        # since the last layer of an XOR network is just one
        # we get the scalar element of the final 1x1 matrix in the
        # last element of the self.Y outputs
        return self.Y[len(self.Y) - 1][0][0]

    def backpropagation(self, target_y):
        self.DX_Y[len(self.layers) - 1] = cost_dy_pred_v(
            self.Y[len(self.layers) - 1],
            np.array([[target_y]])
        )

        l = 0
        while l < len(self.layers) - 1:
            np.multiply(self.DX_Y[len(self.Y) - 1 - l], sigmoid_dx_v(self.Z[len(self.Z) - 1 - l]), out=self.DX_Z[len(self.Z) - 1 - l])
            
            np.outer(self.DX_Z[len(self.Z) - 1 - l], self.Y[len(self.layers) - 2 - l], out=self.DX_W[len(self.W) - 1 - l])
            
            np.dot(self.W[len(self.W) - 1 - l].T, self.DX_Z[len(self.Z) - 1 - l], out=self.DX_Y[len(self.Y) - 2 - l])

            np.multiply(self.DX_W[len(self.W) - 1 - l], self.alpha, out=self.DX_W[len(self.W) - 1 - l])
            np.subtract(self.W[len(self.W) - 1 - l], self.DX_W[len(self.W) - 1 - l], out=self.W[len(self.W) - 1 - l])

            l += 1

    def test(self):
        i = 0
        while i < 4:
            in_x1 = self.training_input[i][0]
            in_x2 = self.training_input[i][1]
            print("xor(", in_x1, ", ", in_x2, ") = ", self.feedforward(in_x1, in_x2))
            i += 1

if __name__ == "__main__":

    xor = xor_net([2, 2, 1], 0.19)

    xor.W[0] = np.array([
        [0.5, 0.9],
        [0.1, 0.75]
    ])

    xor.W[1] = np.array([
        [0.85, 0.2]
    ])

    print("initial test:")
    xor.test()
    print("\nxor.WL1 = \n", xor.W[0])
    print("\nxor.WL2 = ", xor.W[1])

    start_time = time.time()
    epoch = 0
    while epoch < 25_000:
        i = 0
        while i < 4:
            in_x1 = xor.training_input[i][0]
            in_x2 = xor.training_input[i][1]
            xor.feedforward(in_x1, in_x2)
            xor.backpropagation(xor.training_output[i])
            # print("xor(", in_x1, ", ", in_x2, ") : target = ", xor.training_output[i])
            i += 1
        epoch += 1
    end_time = time.time()

    print("final test:\n")
    xor.test()
    print("\nxor.WL1 = \n", xor.W[0])
    print("\nxor.WL2 = ", xor.W[1])

    print("--- %s seconds ---" % (end_time - start_time))