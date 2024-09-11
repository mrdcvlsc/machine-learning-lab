# pytorch implementation using torch.nn module and torch.optimizer

import torch
import time
import numpy as np

LEARNING_RATE = 0.55
EPOCH = 2_000

def tnsr_to_np(t: torch.Tensor):
    return t.clone().detach().numpy()

def tnsr_to_np_int(t: torch.Tensor):
    return tnsr_to_np(t).astype(int)

# inherit the nn.Module class
class XorNet(torch.nn.Module):
    def __init__(self, w1, w2, w3, w4, w5, w6, alpha):

        # initialize the nn.Module class
        super(XorNet, self).__init__()

        self.activation_fn = torch.nn.Sigmoid()
        self.linear1 = torch.nn.Linear(2, 2, bias=True, dtype=torch.float64)
        self.linear2 = torch.nn.Linear(2, 1, bias=True, dtype=torch.float64)

        # edit weights and biases of the linear layers to my preferred values
        with torch.no_grad():
            self.linear1.weight.copy_(torch.tensor([
                [w1, w2],
                [w3, w4]
            ], dtype=torch.float64))

            self.linear2.weight.copy_(torch.tensor([
                [w5, w6]
            ], dtype=torch.float64))

            # biases in nn.Linear layer are just 1D tensors not
            # 2D tensors with 1 colum or 1 row

            self.linear1.bias.copy_(torch.ones((2), dtype=torch.float64))
            self.linear2.bias.copy_(torch.ones((1), dtype=torch.float64))

        # inputs and target outputs does not require gradients
        self.training_input = torch.tensor([[0, 0], [0, 1] , [1, 0], [1, 1]]).double()
        self.training_output = torch.tensor([0, 1, 1, 0]).double()

        # set a learning rate
        self.alpha = alpha

    # implement a "forward" function for the model (feedforward)
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        x = self.linear2(x)
        x = self.activation_fn(x)
        return x

    def feedforward(self, x1, x2):
        x = torch.tensor([
            [x1],
            [x2]
        ], dtype=torch.float64)

        # the input shape for a nn.Linear layer is not a column matrix
        # but instead it should be a row matrix
        x = x.T

        x = self.forward(x)

        return x[0, 0]

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
print("\nxor.WL1 = \n", tnsr_to_np(xor.linear1.weight))
print("\nxor.WL2 = ", tnsr_to_np(xor.linear2.weight))
print("\nxor.B1 = ", np.array([tnsr_to_np(xor.linear1.bias)]).T)
print("\nxor.B2 = ", np.array([tnsr_to_np(xor.linear2.bias)]))

# setup need for training
optimizer = torch.optim.SGD(xor.parameters(), lr=xor.alpha)
loss_fn = torch.nn.MSELoss()

def train_one_backward_pass(i):
    in_x1 = xor.training_input[i, 0]
    in_x2 = xor.training_input[i, 1]

    inputs  = torch.tensor([[in_x1, in_x2]], dtype=torch.float64)
    targets = torch.tensor([[xor.training_output[i]]], dtype=torch.float64)    

    # gradients are accumulated in pytorch tensors
    # so we need to zero them out every backward pass
    optimizer.zero_grad()

    output = xor(inputs)

    # calculate the loss/cost of the network
    loss = loss_fn(output, targets)

    # calculate the gradients of the network
    loss.backward()

    # update the network weights and biases with the specified learning rate
    optimizer.step()

start_time = time.time()

# training with torch.nn.Linear

epoch = 0
while epoch < EPOCH:
    i = 0
    while i < 4:
        train_one_backward_pass(i)
        # print("xor(", in_x1, ", ", in_x2, ") : target = ", xor.training_output[i])
        i += 1
    epoch += 1

end_time = time.time()

print("final test:\n")
xor.test()
print("\nxor.WL1 = \n", tnsr_to_np(xor.linear1.weight))
print("\nxor.WL2 = ", tnsr_to_np(xor.linear2.weight))
print("\nxor.B1 = ", np.array([tnsr_to_np(xor.linear1.bias)]).T)
print("\nxor.B2 = ", np.array([tnsr_to_np(xor.linear2.bias)]))

print("--- %s seconds ---" % (end_time - start_time))