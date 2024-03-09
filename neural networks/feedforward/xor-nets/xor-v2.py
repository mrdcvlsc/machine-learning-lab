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
        self.Z : list[NDArray[np.float64]] = [np.array([])] * (len(layers) - 1)
        
        # same as self.Z (N x 1 in size) - this list will contain the collection of activated nodes
        # with the sole exception of the very first element which is just the input layer that does
        # not have any activation, but should be normalized.
        self.Y : list[NDArray[np.float64]] = [np.array([])] * len(layers)

        self.W = []
        
        i = 0

        while i < len(layers) - 1:
            self.W.append(
                np.random.uniform(low=0, high=1, size=(layers[i + 1], layers[i]))
            )
            i += 1

        self.training_input = np.array([[0, 0], [0, 1] , [1, 0], [1, 1]])
        self.training_output = np.array([0, 1, 1, 0])

    def feedforward(self, x1, x2):
        self.Y[0] = np.array([
            [x1],
            [x2]
        ])

        l = 0
        while l < len(self.layers) - 1:
            self.Z[l] = np.dot(self.W[l], self.Y[l])
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
        # get the partial derivatives of the cost function
        # /w respect to the final Y function/output (input to the cost function).
        dx_cost_Y = cost_dy_pred_v(
            self.Y[len(self.layers) - 1],
            np.array([[target_y]])
        )

        dx_Y_Z = 0

        l = 0
        while l < len(self.layers) - 1:
            # get the partial derivatives of the Y function/output (activation function)
            # with respect to Z function/outputs.
            dx_Y_Z = sigmoid_dx_v(self.Z[len(self.Z) - 1 - l])

            # get the partial derivatives of the cost function /w respect to
            # the Z function/output of the current layer.
            # note:
            #   we can get this partial derivative with the HADAMARD MULTIPLICATION of
            #   the partial derivative of the cost function with respect to the current
            #   Y output of the layer, hadamard multiplied by the partial derivative of
            #   the Y function/output /w respect to the Z inputs of the current layer.
            dx_cost_Z = dx_cost_Y * dx_Y_Z

            # get the partial derivative of the Z function/output of the current
            # layer with respect to the weight of the current layer.
            dx_Z_W = self.Y[len(self.layers) - 2 - l]

            # get the partial derivative of the cost function /w respect to
            # the weights of the current layer.
            # note:
            #   like the previous operation above /w hadamard multiplication...
            #   this time we can get this partial derivative easily by getting
            #   the OUTER PRODUCT of the partial derivative of the cost function /w
            #   respect to the current Z and the partial derivative of the Z function
            #   /w respect to the weights.
            dx_cost_W = np.outer(dx_cost_Z, dx_Z_W)

            # we then apply sochastic gradient descent to the current weights
            # of the neural network
            old_weights = self.W[len(self.W) - 1 - l]
            self.W[len(self.W) - 1 - l] = self.W[len(self.W) - 1 - l] - (self.alpha * dx_cost_W)

            # ----------------------------------------------------------------------------------

            # then we update the next partial derivative of the cost function with respect
            # to the previous layer's Y.

            # get the partial derivative of the Z function/output with respect to its
            # X (previous layer's Y) input, this is just the transpose of the weight of the current layer before the update.
            dx_Z_X = old_weights.T

            # get the partial derivative of the cost function /w respect to the 
            # current X (previous layer's Y) input, and make it our new partial derivative
            # of the cost function /w respect to the Y for the next previous layer
            dx_cost_Y = np.dot(dx_Z_X, dx_cost_Z)
            l += 1
        
    
    def print_last_iteration_stat(self):
        print("x    = \n", self.Y[0])
        print("W_L1 = \n", self.W[0])
        
        print("\nZ1 = \n", self.Z[0])
        print("A1 = \n", self.Y[1])
        print("W_L2 = ", self.W[1])
        
        print("\nZ2 = ", self.Z[2])
        print("A2 = ", self.Y[3])

    def test(self):
        i = 0
        while i < 4:
            in_x1 = self.training_input[i][0]
            in_x2 = self.training_input[i][1]
            print("xor(", in_x1, ", ", in_x2, ") = ", self.feedforward(in_x1, in_x2))
            i += 1

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