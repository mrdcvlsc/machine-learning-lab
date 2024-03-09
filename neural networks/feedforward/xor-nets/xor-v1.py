# raw implementation of xor neural network
# all gradients/partial derivatives are calculated and
# mapped by hand and the neural network code is not dynamic

import time
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

def cost_dy_pred(y_pred, y_target):
    return 2*(y_pred - y_target)

class xor_net:
    def __init__(self, w1, w2, w3, w4, w5, w6, alpha):
        self.W_L1 = np.array([
            [w1, w2],
            [w3, w4]
        ])

        self.W_L2 = np.array([
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

        self.Z1 = np.dot(self.W_L1, self.X)
        self.A1 = sigmoid_v(self.Z1)

        self.Z2 = np.dot(self.W_L2, self.A1)
        self.A2 = sigmoid_v(self.Z2)

        return self.A2[0][0]

    def backpropagation(self, target_y):
        cost_Al2_dx = cost_dy_pred(self.A2[0][0], target_y)
        Al2_Zl2_dx = sigmoid_dx(self.Z2[0][0])
        Zl2_Wl2_dx = self.A1.T
        Zl2_Al1_dx = self.W_L2.T

        cost_Wl2_dx = cost_Al2_dx * Al2_Zl2_dx * Zl2_Wl2_dx
        New_W_L2 = self.W_L2 - (self.alpha * cost_Wl2_dx)

        cost_Al1_dx = cost_Al2_dx * Al2_Zl2_dx * Zl2_Al1_dx
        Al1_Zl1_dx = sigmoid_dx_v(self.Z1)
        Zl1_Wl1_dx = np.array([self.X.flatten(), self.X.flatten()])
        
        cost_Wl1_dx = cost_Al1_dx * Al1_Zl1_dx * Zl1_Wl1_dx
        New_W_L1 = self.W_L1 - (self.alpha * cost_Wl1_dx)
        
        # print("cost_Wl1_dx = ", cost_Wl1_dx)
        # print("cost_Al2_dx = ", cost_Al2_dx)
        # print("Al2_Zl2_dx  = ", Al2_Zl2_dx)
        # print("Zl2_Wl2_dx  = ", Zl2_Wl2_dx)
        # print("Zl2_Al1_dx  = \n", Zl2_Al1_dx)
        # print("cost_Wl2_dx = ", cost_Wl2_dx)
        # print("cost_Al1_dx = \n", cost_Al1_dx)
        # print("Al1_Zl1_dx  = \n", Al1_Zl1_dx)
        # print("Zl1_Wl1_dx  = \n", Zl1_Wl1_dx)

        self.W_L1 = New_W_L1
        self.W_L2 = New_W_L2
        
        # print("New_W_L1+   = \n", self.W_L1)
        # print("New_W_L2+   = ", self.W_L2) 
    
    def print_last_iteration_stat(self):
        print("x    = \n", self.X)
        print("W_L1 = \n", self.W_L1)
        
        print("\nZ1 = \n", self.Z1)
        print("A1 = \n", self.A1)
        print("W_L2 = ", self.W_L2)
        
        print("\nZ2 = ", self.Z2)
        print("A2 = ", self.A2)

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
print("\nxor.WL1 = \n", xor.W_L1)
print("\nxor.WL2 = ", xor.W_L2)

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
print("\nxor.WL1 = \n", xor.W_L1)
print("\nxor.WL2 = ", xor.W_L2)

print("--- %s seconds ---" % (end_time - start_time))