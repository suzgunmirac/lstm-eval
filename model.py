# author: ms

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import random

#LSTM model
class MyLSTM(nn.Module):
    def __init__(self, hidden_dim, vocab_size, n_layers):
        super(MyLSTM, self).__init__()
        # parameters
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.hidden_layers = n_layers

        # layers
        self.lstm = nn.LSTM(vocab_size, hidden_dim, n_layers)
        self.linear = nn.Linear(hidden_dim, vocab_size+1) ## vocab_size + 1 because of the termination symbol T
        self.sigmoid = nn.Sigmoid ()

    def init_hidden (self): 
        return (torch.zeros (self.hidden_layers, 1, self.hidden_dim)),(torch.zeros (self.hidden_layers, 1, self.hidden_dim))

    def forward(self, input, hidden0):
        # apply the LSTM layer
        output, hidden = self.lstm(input, hidden0)

        # apply the linear layer
        output = self.linear(output)
        
        # apply the softmax layer
        output = output.view (len(output), self.vocab_size + 1)
        
        output = self.sigmoid (output)
        return output, hidden
