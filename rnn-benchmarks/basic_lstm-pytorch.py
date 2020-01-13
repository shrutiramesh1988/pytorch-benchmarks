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
optparser.add_option("-n", "--network_type", default='basic_lstm', help="Network type (rnn, lstm, basic_lstm)")
optparser.add_option("-l", "--hidden_size", default=100, type='int', help="Hidden layer size")
optparser.add_option("-s", "--seq_length", default=30, type='int', help="Sequence length")
optparser.add_option("-b", "--batch_size", default=20, type='int', help="Batch size")
opts = optparser.parse_args()[0]

network_type = opts.network_type
print(network_type)
hidden_size = opts.hidden_size
hidden_size = opts.hidden_size
seq_length = opts.seq_length
batch_size = opts.batch_size

n_batch = 100
n_samples = batch_size * n_batch

# Data
xinput = np.random.rand(seq_length, batch_size, hidden_size).astype(np.float32)
ytarget = np.random.rand(batch_size, hidden_size).astype(np.float32)
h_lstm = Variable(torch.zeros(batch_size, hidden_size))
c_lstm = Variable(torch.zeros(batch_size, hidden_size))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size, bias=True)
        #self.fc = nn.Linear(hidden_size,hidden_size, bias=False)

    def forward(self, x):
        h_lstm = Variable(torch.zeros(batch_size, hidden_size))
        c_lstm = Variable(torch.zeros(batch_size, hidden_size))
        output = []
        for i in range(seq_length):
            h_lstm, c_lstm = self.lstm(x[i], (h_lstm, c_lstm))
            output.append(h_lstm)
        h1 = torch.stack(output)
        h2 = h1[-1, :, :]
        return h2
        #h3 = self.fc(h2)
        #return h3

X0_batch = torch.tensor(xinput,dtype = torch.float) 
Y0_batch = torch.tensor(ytarget,dtype = torch.float) 

net = Net()

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

