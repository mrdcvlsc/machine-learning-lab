# improved version of the original implementation
# the code is more dynamic than the previous one

import math
import numpy as np

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

class AffineNet:
    def __init__(self, layerNodes, alpha):
        self.alpha = alpha

        self.layerNodes = layerNodes
        
        self.Z = [0] * len(layerNodes)
        self.Y = [0] * len(layerNodes)

        self.weights = []
        
        l = 0

        while l < len(layerNodes) - 1:
            self.weights.append(
                np.random.uniform(low=0, high=1, size=(layerNodes[l + 1], layerNodes[l]))
            )
            l += 1

        self.training_input = np.array([[0, 0], [0, 1] , [1, 0], [1, 1]])
        self.training_output = np.array([0, 1, 1, 0])

    def feedforward(self, X):
        # add dimension checks later
        self.X = np.array(X).T
        self.Z[0] = self.X

        l = 0
        while l < len(self.layerNodes) - 1:
            self.Z[l + 1] = np.dot(self.weights[l], self.Z[l])
            self.Y[l + 1] = sigmoid_dx_v(self.Z[l + 1])

        return self.Y[len(self.Y) - 1]

    def backpropagation(self, y_target):
        l = len(self.layerNodes) - 1
        dx_cost_Y = cost_dy_pred_v(self.Y[l])
        dx_Y_Z = 0
        dx_Z_W = 0

        while l >= 0:
            dx_Y_Z = sigmoid_dx_v(self.Z[l])
            dx_Z_W = self.Y[l - 1]

            l -= 1

class xor_net:
    def __init__(self, w1, w2, w3, w4, w5, w6, alpha):
        self.W1 = np.array([
            [w1, w2],
            [w3, w4]
        ])

        self.W2 = np.array([
            [w5, w6]
        ])

        self.alpha = alpha
        
        self.training_input = np.array([[0, 0], [0, 1] , [1, 0], [1, 1]])
        self.training_output = np.array([0, 1, 1, 0])

    def feedforward(self, x1, x2):
        self.X = np.array([
            [x1],
            [x2]
        ])

        self.Z1 = np.dot(self.W1, self.X)
        self.Y1 = sigmoid_v(self.Z1)

        self.Z2 = np.dot(self.W2, self.Y1)
        self.Y2 = sigmoid_v(self.Z2)

        return self.Y2[0][0]

    def backpropagation(self, target_y):
        dx_cost_Y2 = cost_dy_pred(self.Y2[0][0], target_y)
        dx_Y2_Z2 = sigmoid_dx(self.Z2[0][0])
        dx_Z2_W2 = self.Y1.T
        dx_Z2_Y1 = self.W2.T

        dx_cost_W2 = dx_cost_Y2 * dx_Y2_Z2 * dx_Z2_W2
        new_W2 = self.W2 - (self.alpha * dx_cost_W2)

        dx_cost_Y1 = dx_cost_Y2 * dx_Y2_Z2 * dx_Z2_Y1
        dx_Y1_Z1 = sigmoid_dx_v(self.Z1)
        dx_Z1_W1 = np.array([self.X.flatten(), self.X.flatten()])
        
        dx_cost_W1 = dx_cost_Y1 * dx_Y1_Z1 * dx_Z1_W1
        new_W1 = self.W1 - (self.alpha * dx_cost_W1)
        
        # print("dx_cost_W1 = ", dx_cost_W1)
        # print("dx_cost_Y2 = ", dx_cost_Y2)
        # print("dx_Y2_Z2  = ", dx_Y2_Z2)
        # print("dx_Z2_W2  = ", dx_Z2_W2)
        # print("dx_Z2_Y1  = \n", dx_Z2_Y1)
        # print("dx_cost_W2 = ", dx_cost_W2)
        # print("dx_cost_Y1 = \n", dx_cost_Y1)
        # print("dx_Y1_Z1  = \n", dx_Y1_Z1)
        # print("dx_Z1_W1  = \n", dx_Z1_W1)

        self.W1 = new_W1
        self.W2 = new_W2
        
        # print("new_W1+   = \n", self.W1)
        # print("new_W2+   = ", self.W2) 
    
    def print_last_iteration_stat(self):
        print("x    = \n", self.X)
        print("W_L1 = \n", self.W1)
        
        print("\nZ1 = \n", self.Z1)
        print("A1 = \n", self.Y1)
        print("W_L2 = ", self.W2)
        
        print("\nZ2 = ", self.Z2)
        print("A2 = ", self.Y2)

    def test(self):
        i = 0
        while i < 4:
            in_x1 = self.training_input[i][0]
            in_x2 = self.training_input[i][1]
            print("xor(", in_x1, ", ", in_x2, ") = ", self.feedforward(in_x1, in_x2))
            i += 1

xor = xor_net(0.5, 0.9, 0.1, 0.75, 0.85, 0.2, 0.19)

print("initial test:")
xor.test()
print("\nxor.WL1 = \n", xor.W1)
print("\nxor.WL2 = ", xor.W2)

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

print("final test:\n")
xor.test()
print("\nxor.WL1 = \n", xor.W1)
print("\nxor.WL2 = ", xor.W2)