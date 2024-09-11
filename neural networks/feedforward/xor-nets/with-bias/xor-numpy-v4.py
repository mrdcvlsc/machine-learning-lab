# TODO: Look for optimization, make it as fast as xor-v1.py if possible

import time
import math
import numpy as np
from numpy._typing import NDArray

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

cost_v = np.vectorize(cost)

def cost_dy_pred(y_pred, y_target):
    return 2*(y_pred - y_target)

cost_dy_pred_v = np.vectorize(cost_dy_pred)

trainingInput = [
    np.array([[0], [0]]),
    np.array([[0], [1]]),
    np.array([[1], [0]]),
    np.array([[1], [1]])
]

trainingOutput = [
    np.array([[0]]),
    np.array([[1]]),
    np.array([[1]]),
    np.array([[0]])
]

class AffineLayer:
    def __init__(self, inputs, outputs, min = 0,  max = 1, hasBias=False):
        """
        Initialize a layer and all its weights.

        Parameters
        ----------
        `inputs` : `int`
            The number of input nodes of the layer.
        
        `outputs` : `int`
            The number of output nodes of the layer.

        `min = ? optional` : `int : (default 0)`
            Minimum possible value of the weights during random initialization.
        
        `max = ? optional` : `int : (default 1)`
            Maximum possible value of the weights during random initialization.

        `hasBias = ? optional` : `bool : (default 1)`
            Initialize with biases.
        """

        self.W = np.random.uniform(low=min, high=max, size=(outputs, inputs))
        '''Layer Weights'''

        self.hasBias = hasBias

        if hasBias:
            self.B = np.ones(shape=(outputs, 1))
            '''Layer Biases'''

    def forward(self, inputNodes):
        """
        Apply forward pass to input,
        the output nodes of this method are not activated yet.
        One might need to pass the output of this method
        to an activation function.

        Parameters
        ----------
        `inputNodes` : `NDArray[np.float64]`
            An `Nx1` numpy array matrix as input nodes to the layer.
        
        Returns
        -------
        `NDArray[np.float64]`
            A `Mx1` numpy array that contains the weighted input.
        """
        
        self.X = inputNodes

        if self.hasBias:
            self.Z = np.dot(self.W, self.X) + self.B
        else:
            self.Z = np.dot(self.W, self.X)

        return self.Z

    def propagate(self, dxCost_previous_Z, alpha):
        """
        Update the weights of the layer and get the
        partial derivatives the cost function with
        respect to the previous input nodes.

        Parameters
        ----------
        `dxCost_previous_Z` : `NDArray[np.float64]`
            An `Nx1` numpy array matrix that is the partial
            derivative of the cost function with respect
            to the previous non-activated output nodes or
            dot product function inputs.
        
        `alpha` : `float`
            The learning rate of the current layer use for SDG.
        
        Returns
        -------
        `NDArray[np.float64]`
            A `Mx1` numpy array that contains the partial
            derivatives of the cost function with respect
            to the input nodes of the current layer.
        """

        dx_Z_W = self.X
        dx_Cost_W = np.outer(dxCost_previous_Z, dx_Z_W)
        
        dx_Z_X = self.W.T
        dx_Cost_X = np.dot(dx_Z_X, dxCost_previous_Z)

        self.W = self.W - (alpha * dx_Cost_W)

        if self.hasBias:
            # the partial derivatives of the dot product function with respect to the bias
            # are just 1s, so there's no need to multiply 1s to the partial derivatives of
            # the cost function with respect to the input Z (dot product function inputs result) to save computation.
            dx_Cost_B = dxCost_previous_Z
            self.B = self.B - (alpha * dx_Cost_B)

        return dx_Cost_X

# TODO: Create an abstract class for this neural network class.
class MyXorNet:
    def __init__(self, LearningRate):
        self.LearningRate = LearningRate
        
        self.L1 = AffineLayer(2, 2, hasBias=True)
        self.L2 = AffineLayer(2, 1, hasBias=True)

    # TODO: Make this an abstract function.
    def feedforward(self, x):
        x = self.L1.forward(x)
        x = sigmoid_v(x)

        x = self.L2.forward(x)
        x = sigmoid_v(x)

        return x
    
    # TODO: Make this an abstract function.
    def backpropagate(self, y, target):
        dx_cost_Y = cost_dy_pred_v(y, target)

        dx_cost_Z = dx_cost_Y * sigmoid_dx_v(self.L2.Z)
        dx_cost_Y = self.L2.propagate(dx_cost_Z, alpha=self.LearningRate)

        dx_cost_Z = dx_cost_Y * sigmoid_dx_v(self.L1.Z)
        dx_cost_Y = self.L1.propagate(dx_cost_Z, alpha=self.LearningRate)
    
    def test(self):
        i = 0
        while i < 4:
            # in_x1 = self.training_input[i][0]
            # in_x2 = self.training_input[i][1]
            print("xor(", trainingInput[i][0][0], ", ", trainingInput[i][1][0], ") = ", self.feedforward(trainingInput[i])[0][0])
            i += 1

xor = MyXorNet(LEARNING_RATE)

xor.L1.W = np.array([
    [0.5, 0.9],
    [0.1, 0.75]
])

xor.L2.W = np.array([
    [0.85, 0.2]
])

print("initial test:")
xor.test()
print("\nxor.WL1 = \n", xor.L1.W)
print("\nxor.WL2 = ", xor.L2.W)
print("\nxor.B1 = ", xor.L1.B)
print("\nxor.B2 = ", xor.L2.B)

start_time = time.time()
epoch = 0
while epoch < EPOCH:
    i = 0
    while i < 4:
        y = xor.feedforward(trainingInput[i])
        xor.backpropagate(y, trainingOutput[i])
        # print("xor(", in_x1, ", ", in_x2, ") : target = ", xor.training_output[i])
        i += 1
    epoch += 1
end_time = time.time()

print("final test:\n")
xor.test()
print("\nxor.WL1 = \n", xor.L1.W)
print("\nxor.WL2 = ", xor.L2.W)
print("\nxor.B1 = ", xor.L1.B)
print("\nxor.B2 = ", xor.L2.B)

print("--- %s seconds ---" % (end_time - start_time))