import time
import optparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
#import random

# Parameters
optparser = optparse.OptionParser()
optparser.add_option("-n", "--network_type", default='lstm', help="Network type (rnn, lstm, basic_lstm)")
optparser.add_option("-i", "--input_size", default=100, type='int', help="Input layer size")
optparser.add_option("-l", "--hidden_size", default=100, type='int', help="Hidden layer size")
optparser.add_option("-s", "--seq_length", default=30, type='int', help="Sequence length")
optparser.add_option("-b", "--batch_size", default=1, type='int', help="Batch size")
opts = optparser.parse_args()[0]

#Input size should be same as hidden size
network_type = opts.network_type
print(network_type)
input_size = opts.hidden_size #opts.input_size
hidden_size = opts.hidden_size
seq_length = opts.seq_length
batch_size = opts.batch_size

n_batch = 100
n_samples = batch_size * n_batch

# Data
#If batch_first=True
#xinput = np.random.rand(batch_size, seq_length, hidden_size).astype(np.float32)
xinput = np.random.rand(seq_length,batch_size,hidden_size).astype(np.float32)
xinput = xinput.transpose(1,0,2)
#If batch_first=False
#xinput = np.random.rand(seq_length, batch_size, hidden_size).astype(np.float32)
#ytarget = np.random.rand(batch_size, hidden_size).astype(np.float32)
ytarget = np.random.rand(batch_size, seq_length, hidden_size).astype(np.float32)

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(Net, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        
        #Commented FC layer
        # Readout layer
        #self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we dmodel = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)on't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        #Commented FC layer
        #out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out

X0_batch = torch.tensor(xinput,dtype = torch.float) 
Y0_batch = torch.tensor(ytarget,dtype = torch.float) 


net = Net(input_size, hidden_size, 1, hidden_size)

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
for i in range(0, n_batch):
    optimizer.zero_grad()
    output = net(X0_batch)
end = time.time()
print("Forward:")
print("--- %i samples in %s seconds (%1.5f samples/s, %1.7f s/sample) ---" % (n_samples, end - start, n_samples / (end - start), (end - start) / n_samples))


start = time.time()
for i in range(0, n_batch):
    optimizer.zero_grad()
    output = net(X0_batch)
    loss = criterion(output, Y0_batch)
    loss.backward()
    optimizer.step()
end = time.time()


print("Forward + Backward:")
print("--- %i samples in %s seconds (%1.5f samples/s, %1.7f s/sample) ---" % (n_samples, end - start, n_samples / (end - start), (end - start) / n_samples))

