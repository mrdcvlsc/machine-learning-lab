# pytorch implementation using only it's torch.tenssor and auto-grad

import torch
import time

LEARNING_RATE = 0.55
EPOCH = 2_000

def tnsr_to_np(t: torch.Tensor):
    return t.clone().detach().numpy()

def tnsr_to_np_int(t: torch.Tensor):
    return tnsr_to_np(t).astype(int)

class XorNet:
    def __init__(self, w1, w2, w3, w4, w5, w6, alpha):
        self.W1 = torch.tensor([
            [w1, w2],
            [w3, w4]
        ], requires_grad=True, dtype=torch.float64)

        self.B1 = torch.ones((2, 1), requires_grad=True, dtype=torch.float64)

        self.W2 = torch.tensor([
            [w5, w6]
        ], requires_grad=True, dtype=torch.float64)

        self.B2 = torch.ones((1, 1), requires_grad=True, dtype=torch.float64)

        self.alpha = alpha
        
        # inputs and target outputs does not require gradients
        self.training_input = torch.tensor([[0, 0], [0, 1] , [1, 0], [1, 1]]).double()
        self.training_output = torch.tensor([0, 1, 1, 0]).double()

    def feedforward(self, x1, x2):
        self.X = torch.tensor([
            [x1],
            [x2]
        ], dtype=torch.float64)

        # the result of the dot product function for layer 1 [dim: 2x1]
        self.Z1 = torch.mm(self.W1, self.X) + self.B1

        # the result of the activation function for the layer 1 [dim: 2x1]
        self.Y1 = torch.sigmoid(self.Z1)

        # the result of the dot product function for layer 2 [dim: 1x1]
        self.Z2 = torch.mm(self.W2, self.Y1) + self.B2

        # the result of the activation function for the layer 2 [dim: 1x1]
        self.Y2 = torch.sigmoid(self.Z2)

        return self.Y2[0, 0]

    def backpropagation_V2(self, target_y):
        loss = torch.nn.MSELoss()
        cost = loss(self.Y2, torch.tensor([[target_y]], dtype=torch.float64))
        cost.backward()

        # equivalent to optimizer.step()
        with torch.no_grad():
            self.W2 -= self.W2.grad * self.alpha
            self.B2 -= self.B2.grad * self.alpha
            self.W1 -= self.W1.grad * self.alpha
            self.B1 -= self.B1.grad * self.alpha

        # equivalent to optimizer.zero_grad()
        self.W2.grad.zero_()
        self.B2.grad.zero_()
        self.W1.grad.zero_()
        self.B1.grad.zero_()

    def test(self):
        i = 0
        while i < 4:
            in_x1 = self.training_input[i, 0]
            in_x2 = self.training_input[i, 1]
            print("xor(", tnsr_to_np_int(in_x1), ", ", tnsr_to_np_int(in_x2), ") = ", tnsr_to_np(self.feedforward(in_x1, in_x2)))
            i += 1

xor = XorNet(0.5, 0.9, 0.1, 0.75, 0.85, 0.2, LEARNING_RATE)

print("initial test:")
xor.test()
print("\nxor.WL1 = \n", tnsr_to_np(xor.W1))
print("\nxor.WL2 = ", tnsr_to_np(xor.W2))
print("\nxor.B1 = ", tnsr_to_np(xor.B1))
print("\nxor.B2 = ", tnsr_to_np(xor.B2))

start_time = time.time()
epoch = 0
while epoch < EPOCH:
    i = 0
    while i < 4:
        in_x1 = xor.training_input[i, 0]
        in_x2 = xor.training_input[i, 1]
        xor.feedforward(in_x1, in_x2)
        xor.backpropagation_V2(xor.training_output[i])
        # print("xor(", in_x1, ", ", in_x2, ") : target = ", xor.training_output[i])
        i += 1
    epoch += 1
end_time = time.time()

print("final test:\n")
xor.test()
print("\nxor.WL1 = \n", tnsr_to_np(xor.W1))
print("\nxor.WL2 = ", tnsr_to_np(xor.W2))
print("\nxor.B1 = ", tnsr_to_np(xor.B1))
print("\nxor.B2 = ", tnsr_to_np(xor.B2))

print("--- %s seconds ---" % (end_time - start_time))