from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networks
from IPython.display import HTML

'''
    Network and synthesis parameters
'''
input_file_name = "data/speech.txt"
n_hidden = 100
seq_length = 25
syn_length = 200
n_epochs = 50
learning_rate = 0.01

'''
    Prepare input data
'''
# Raw text data
book_data = open(input_file_name, encoding="utf-8").read().strip()
# List of unique characters
book_chars = list(set(book_data))
# K: Number of classes (unique characters)
K = len(book_chars)

# Create structures mapping class indexes to chars
char_to_ind = dict()
ind_to_char = dict()
for i in range(K):
    char_to_ind[book_chars[i]] = i
    ind_to_char[i] = book_chars[i]

# Set random seed for reproducibility
# (doesn't seem to work very well)
manualSeed = 999
# manualSeed = random.randint(1, 10000)
print("Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

'''
    Create network and loss criterion
'''
net = networks.RNN(K, n_hidden, K)
criterion = nn.NLLLoss()
optimizer = optim.Adagrad(net.parameters(), lr=learning_rate)

'''
    Utility functions
'''
# Convert a string to a one-hot [len(str), 1, K] tensor representation of all character classes in the string.
def stringToTensor(str):
    tensor = torch.zeros(len(str), 1, K)
    for i in range(len(str)):
        tensor[i, 0, char_to_ind[str[i]]] = 1
    return tensor

# Convert a string to a non-one-hot [len(str)] tensor representation of all character classes in the string. Used for labels in NLLLoss, which are not one-hot.
def stringToTensorNLLLabel(str):
    tensor = torch.zeros(len(str))
    for i in range(len(str)):
        tensor[i] = char_to_ind[str[i]]
    return tensor.type(torch.LongTensor)

# Convert an integer to a [1, K] one-hot tensor representation
def toOneHot(val):
    tensor = torch.zeros(1, K)
    tensor[0, val] = 1
    return tensor

# Convert an array of character class integers (non-one-hot) to the corresponding string representation
def indsToString(inds):
    str = ""
    for i in inds:
        str += ind_to_char[i]
    return str

# Synthesize a text sequence of length n
def synthesize(n):
    hidden = net.initHidden()
    net.zero_grad()
    prev_char = toOneHot(char_to_ind['.'])
    word = []
    for i in range(n):
        '''
            Select the next character based on the probability distribution.
            Note: we don't always select the most likely character, as that would
            likely result in repeating phrases such as "the the the the the..."
        '''
        p, hidden = net(prev_char, hidden)
        p = torch.exp(p)
        cp = p.cumsum(1)
        a = random.uniform(0, 1)
        b = (cp - a)
        c = (b > 0)
        ixs = c.nonzero()
        ii = ixs[0,1].item()
        prev_char = toOneHot(ii)
        word.append(ii)
    return word

# Train the network on a single character sequence
def train(input_line_tensor, target_line_tensor, hidden):
    target_line_tensor.unsqueeze_(-1)
    # hidden = net.initHidden()
    net.zero_grad()
    optimizer.zero_grad()

    loss = 0

    for i in range(input_line_tensor.size(0)):
        output, hidden = net(input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()
    optimizer.step()

    # for p in net.parameters():
    #     p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item(), hidden.detach()

'''
    Train the network
'''
for epoch in range(n_epochs):
    i=0
    hidden = net.initHidden()
    while i < (len(book_data) - seq_length):
        X_chars = book_data[i:i+seq_length]
        Y_chars = book_data[i+1:i+seq_length+1]
        X = stringToTensor(X_chars)
        Y = stringToTensorNLLLabel(Y_chars)
        output, loss, hidden = train(X, Y, hidden)
        print(loss)
        i += seq_length

'''
    Synthesize some text
'''
text_inds = synthesize(syn_length)
print("Synthesized text:\n", indsToString(text_inds))
