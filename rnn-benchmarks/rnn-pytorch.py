import time
import optparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os

# Parameters
optparser = optparse.OptionParser()
optparser.add_option("-n", "--network_type", default='RNN', help="Network type (rnn, lstm, basic_lstm)")
optparser.add_option("-l", "--hidden_size", default=100, type='int', help="Hidden layer size")
optparser.add_option("-i", "--input_size", default=100, type='int', help="Input layer size")
optparser.add_option("-s", "--seq_length", default=30, type='int', help="Sequence length")
optparser.add_option("-b", "--batch_size", default=20, type='int', help="Batch size")
opts = optparser.parse_args()[0]

network_type = opts.network_type
print(network_type)
hidden_size = opts.hidden_size
seq_length = opts.seq_length
batch_size = opts.batch_size

n_batch = 100
n_samples = batch_size * n_batch
input_size = hidden_size #opts.input_size

# Data
#If batch_first=True
xinput = np.random.rand(batch_size, seq_length, hidden_size).astype(np.float32)
#If batch_first=False
#xinput = np.random.rand(seq_length, batch_size, hidden_size).astype(np.float32)
ytarget = np.random.rand(batch_size, seq_length, hidden_size).astype(np.float32)

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNNModel, self).__init__()
        # Number of hidden dimensions
        self.hidden_size = hidden_size

        # Number of optparser.add_option("-l", "--hidden_size", default=100, type='int', help="Hidden layer size")hidden layers
        self.num_layers = num_layers

        # RNN
        #self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True,
                          #nonlinearity='relu')
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

        #Commenting FC
        # Readout layer
        #self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros

        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        # One time step
        out, hn = self.rnn(x, h0)
        #out = self.fc(out[:, -1, :])
        return out

X0_batch = torch.tensor(xinput,dtype = torch.float)
Y0_batch = torch.tensor(ytarget,dtype = torch.float)

net = RNNModel(input_size, hidden_size, 1, hidden_size)

params = 0
for param in list(net.parameters()):
    sizes = 1
    for el in param.size():
        sizes = sizes * el
    params += sizes
print('# network parameters: ' + str(params))

optimizer = optim.Adam(net.parameters())
criterion = nn.MSELoss()

start = time.time()
for i in range(n_batch):
    optimizer.zero_grad()
    output = net(X0_batch)
end = time.time()
print("Forward:")
print("--- %i samples in %s seconds (%1.5f samples/s, %1.7f s/sample) ---" % (n_samples, end - start, n_samples / (end - start), (end - start) / n_samples))


start = time.time()
for i in range(n_batch):
    optimizer.zero_grad()
    output = net(X0_batch)
    loss = criterion(output, Y0_batch)
    loss.backward()
    optimizer.step()
end = time.time()

print("Forward + Backward:")
print("--- %i samples in %s seconds (%1.5f samples/s, %1.7f s/sample) ---" % (n_samples, end - start, n_samples / (end - start), (end - start) / n_samples))


