# raw implementation of xor neural network
# all gradients/partial derivatives are calculated and
# mapped by hand and the neural network code is not dynamic

import time
import math
import numpy as np

LEARNING_RATE = 0.55
EPOCH = 2_000

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

sigmoid_v = np.vectorize(sigmoid)

def sigmoid_dx(x):
    return sigmoid(x) * (1 - sigmoid(x))

sigmoid_dx_v = np.vectorize(sigmoid_dx)

def cost(y_pred, y_target):
    return (y_pred - y_target)**2

def cost_dy_pred(y_pred, y_target):
    return 2*(y_pred - y_target)

class xor_net:
    def __init__(self, w1, w2, w3, w4, w5, w6, alpha):
        self.W1 = np.array([
            [w1, w2],
            [w3, w4]
        ])

        self.B1 = np.ones(shape=(2, 1))

        self.W2 = np.array([
            [w5, w6]
        ])

        self.B2 = np.ones(shape=(1, 1))

        self.alpha = alpha
        
        self.training_input = np.array([[0, 0], [0, 1] , [1, 0], [1, 1]])
        self.training_output = np.array([0, 1, 1, 0])

    def feedforward(self, x1, x2):
        self.X = np.array([
            [x1],
            [x2]
        ])

        # the result of the dot product function for layer 1 [dim: 2x1]
        self.Z1 = np.dot(self.W1, self.X) + self.B1

        # the result of the activation function for the layer 1 [dim: 2x1]
        self.Y1 = sigmoid_v(self.Z1)

        # the result of the dot product function for layer 2 [dim: 1x1]
        self.Z2 = np.dot(self.W2, self.Y1) + self.B2

        # the result of the activation function for the layer 2 [dim: 1x1]
        self.Y2 = sigmoid_v(self.Z2)

        return self.Y2[0][0]

    def backpropagation(self, target_y):

        # get the partial derivative(s) of the cost function with respect to the
        # input(s) (which is the result(s) of the last activation function)
        dx_COST_Y2 = cost_dy_pred(self.Y2[0][0], target_y)

        #----------------------------------------------------------------------------------------------------------------------------

        # get the partial derivative(s) of the activation function
        # with respect to it's input(s) (the current layer's dot product result).
        dx_Y2_Z2 = sigmoid_dx(self.Z2[0][0]) # [dim: 1x1]

        # get the parital derivative of the cost function with respect to dot product
        # function as input along the computational graph (THIS IS CHAIN RULE!)
        dx_COST_Z2 = np.array([[dx_COST_Y2 * dx_Y2_Z2]]) # [dim: 1x1]

        # get the partial derivatives of the dot product
        # function with respect to the weights as input
        dx_Z2_W2 = self.Y1 # [dim: 1x1]

        # multiply the partial derivatives together to get the partial derivatives of the
        # cost function with respect to the current weights of the layer (THIS IS CHAIN RULE!)
        dx_COST_W2 = np.outer(dx_COST_Z2, dx_Z2_W2) # [dim: 1x2]

        # get the new weights by stepping by an "alpha" through the direction of the gradients
        NEW_W2 = self.W2 - (self.alpha * dx_COST_W2)

        dx_Z2_B2 = np.ones(shape=(1, 1))  # [dim: 1x1]
        dx_COST_B2 = dx_COST_Z2 * dx_Z2_B2  # [dim: 1x1]
        self.B2 = self.B2 - (self.alpha * dx_COST_B2)

        # get the partial derivative of the dot product function with
        # respect to the current input nodes as input
        dx_Z2_Y1 = self.W2.T # [dim: 2x1]

        # the old weight won't be used to any calculations anymore
        # so we update the current weight with the new one
        self.W2 = NEW_W2

        # get the total derivatives of the cost function with respect to each input nodes for the
        # current layer so that we can propagate it to the previous layer. (THIS IS CHAIN RULE!)
        dx_COST_Y1 = np.dot(dx_Z2_Y1, dx_COST_Z2) # [dim: 2x1]

        #----------------------------------------------------------------------------------------------------------------------------

        # follow the same steps above for the current layer

        dx_Y1_Z1 = sigmoid_dx_v(self.Z1) # [dim: 2x1]

        dx_COST_Z1 = dx_COST_Y1 * dx_Y1_Z1 # [dim: 2x1]

        dx_Z1_W1 = self.X # [dim: 2x1]
    
        dx_COST_W1 = np.outer(dx_COST_Z1, dx_Z1_W1) # [dim: 2x2]

        NEW_W1 = self.W1 - (self.alpha * dx_COST_W1)

        self.W1 = NEW_W1

        dx_Z1_B1 = np.ones(shape=(2, 1))  # [dim: 2x1]
        dx_COST_B1 = dx_COST_Z1 * dx_Z1_B1  # [dim: 2x1]
        self.B1 = self.B1 - (self.alpha * dx_COST_B1) # [dim: 2x1]

        # unlike in the previous layer, here we don't need to do the 3rd to the last
        # and the last line of code because there is no previous layer to propagate on

    def test(self):
        i = 0
        while i < 4:
            in_x1 = self.training_input[i][0]
            in_x2 = self.training_input[i][1]
            print("xor(", in_x1, ", ", in_x2, ") = ", self.feedforward(in_x1, in_x2))
            i += 1

xor = xor_net(0.5, 0.9, 0.1, 0.75, 0.85, 0.2, LEARNING_RATE)
# xor = xor_net(0, 0, 0, 0, 0, 0, 0.5)

print("initial test:")
xor.test()
print("\nxor.WL1 = \n", xor.W1)
print("\nxor.WL2 = ", xor.W2)
print("\nxor.B1 = ", xor.B1)
print("\nxor.B2 = ", xor.B2)

start_time = time.time()
epoch = 0
while epoch < EPOCH:
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
print("\nxor.WL1 = \n", xor.W1)
print("\nxor.WL2 = ", xor.W2)
print("\nxor.B1 = ", xor.B1)
print("\nxor.B2 = ", xor.B2)

print("--- %s seconds ---" % (end_time - start_time))