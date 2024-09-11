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

        # tensor board variables
        self.tensorBoardTrainingRecordCount = 0
        self.tensorBoardValidationRecordCount = 0

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

from torch.utils.tensorboard.writer import SummaryWriter
import datetime

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/xor_trainer_{}'.format(timestamp))

def train_one_backward_pass(i, tb_writer: SummaryWriter | None):
    running_loss = 0.
    last_loss = 0.

    in_x1 = xor.training_input[i, 0]
    in_x2 = xor.training_input[i, 1]

    inputs  = torch.tensor([[in_x1, in_x2]], dtype=torch.float64)
    targets = torch.tensor([[xor.training_output[i]]], dtype=torch.float64)    

    optimizer.zero_grad()
    output: torch.Tensor = xor(inputs)

    loss: torch.Tensor = loss_fn(output, targets)
    loss.backward()

    optimizer.step()

    # ======= gather report in tensor board =======
    if (tb_writer != None):
        
        # torch.Tensor.item() returns the scalar value of any tensor that
        # has only one element no matter what the dimension is
        running_loss += loss.item()

        if i == 3:
            last_loss = running_loss / 4
            xor.tensorBoardTrainingRecordCount += 4
            tb_writer.add_scalar('Loss/train', last_loss, xor.tensorBoardTrainingRecordCount)
            running_loss = 0.

        return last_loss

start_time = time.time()

# training with torch.nn.Linear
epoch = 0
best_validation_loss = 1_000_000.
while epoch < EPOCH:
    i = 0
    ave_loss = 0
    while i < 4:
        xor.train(True)
        ave_loss = train_one_backward_pass(i, writer)
        xor.train(False)
        # print("xor(", in_x1, ", ", in_x2, ") : target = ", xor.training_output[i])
        i += 1

    validation_running_loss = 0.
    i = 0
    while i < 4:
        in_x1 = xor.training_input[i, 0]
        in_x2 = xor.training_input[i, 1]

        inputs  = torch.tensor([[in_x1, in_x2]], dtype=torch.float64)
        target = torch.tensor([[xor.training_output[i]]], dtype=torch.float64)    
        
        xor.train(False)
        outputs = xor(inputs)
        validation_loss: torch.Tensor = loss_fn(outputs, target)
        validation_running_loss += validation_loss.item()
        i += 1

    ave_validation_loss = validation_running_loss / 4.
    writer.add_scalars('Training vs. Validation Loss', {
        'Training' : ave_loss,
        'Validation' : ave_validation_loss
    }, epoch + 1)
    writer.flush()

    # track best performance and save the model's state

    if ave_validation_loss * 1.75 < best_validation_loss:
        best_validation_loss = ave_validation_loss
        model_path = 'xor_{}_{}'.format(timestamp, epoch)
        torch.save(xor.state_dict(), model_path)

    epoch += 1

end_time = time.time()

print("final test:\n")
xor.test()
print("\nxor.WL1 = \n", tnsr_to_np(xor.linear1.weight))
print("\nxor.WL2 = ", tnsr_to_np(xor.linear2.weight))
print("\nxor.B1 = ", np.array([tnsr_to_np(xor.linear1.bias)]).T)
print("\nxor.B2 = ", np.array([tnsr_to_np(xor.linear2.bias)]))

print("--- %s seconds ---" % (end_time - start_time))