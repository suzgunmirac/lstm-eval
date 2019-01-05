## Import relevant libraries and dependencies

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import random

## Single-layer LSTM architecture
class MyLSTM(nn.Module):
    def __init__(self, hidden_dim, vocab_size, n_layers):
        super(MyLSTM, self).__init__()
        ## LSTM parameters
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.hidden_layers = n_layers

        ## Layers
        self.lstm = nn.LSTM(vocab_size, hidden_dim, n_layers)
        self.linear = nn.Linear(hidden_dim, vocab_size+1) ## vocab_size + 1 ('T' - term. symbol)
        self.sigmoid = nn.Sigmoid ()

    ## Initialize the hidden and cell states of the LSTM with zeros.
    def init_hidden (self): 
        return (torch.zeros (self.hidden_layers, 1, self.hidden_dim)),(torch.zeros (self.hidden_layers, 1, self.hidden_dim))

    def forward(self, input, hidden0):
        ## Apply the LSTM layer
        output, hidden = self.lstm(input, hidden0)
        ## Apply the linear layer
        output = self.linear(output)
        output = output.view (len(output), self.vocab_size + 1)
        ## Apply the sigmoid layer
        output = self.sigmoid (output)

        return output, hidden