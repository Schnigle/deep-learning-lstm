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
import time
from IPython.display import HTML

'''
    Network and synthesis parameters
'''
input_file_name = "data/speech.txt"
n_hidden = 50
seq_length = 25
syn_length = 500
n_epochs = 200
learning_rate = 0.1
use_cuda = False
'''
    Note: Using the GPU is currently only beneficial for very large network
    sizes since the batches are processed sequentially. For smaller networks
    GPU is much slower than CPU.
'''

if use_cuda and not torch.cuda.is_available():
    print("No CUDA support found. Switching to GPU mode. ")
    use_cuda = False

print("Parameters: ")
print("\tHidden nodes M: ", n_hidden)
print("\tSequence length: ", seq_length)
print("\tLearning rate: ", learning_rate)
print("\tNumber of epochs: ", n_epochs)
print("\tGPU: ", use_cuda)

# Set random seed for reproducibility
# seed = 999
seed = random.randint(1, 10000)
print("\tRandom seed: ", seed)
random.seed(seed)
torch.manual_seed(seed)
print()

'''
    Prepare input data
'''
if use_cuda:
    # float_type = torch.cuda.FloatTensor
    # long_type = torch.cuda.LongTensor
    device = torch.device("cuda")
else:
    # float_type = torch.float
    # long_type = torch.long
    device = torch.device("cpu")
    
# Raw text data
book_data = open(input_file_name, encoding="utf-8").read().strip()
# List of unique characters
book_chars = sorted(list(set(book_data)))
# K: Number of classes (unique characters)
K = len(book_chars)

# Create structures mapping class indexes to chars
char_to_ind = dict()
ind_to_char = dict()
for i in range(K):
    char_to_ind[book_chars[i]] = i
    ind_to_char[i] = book_chars[i]

'''
    Create network and loss criterion
'''
net = networks.RNN(K, n_hidden, K)
if use_cuda:
    net = net.cuda()
criterion = nn.NLLLoss()
optimizer = optim.Adagrad(net.parameters(), lr=learning_rate)

'''
    Utility functions
'''
# Convert a string to a one-hot [len(str), 1, K] tensor representation of all character classes in the string.
def stringToTensor(str):
    tensor = torch.zeros(len(str), 1, K, device=device)
    for i in range(len(str)):
        tensor[i, 0, char_to_ind[str[i]]] = 1
    return tensor

# Convert a string to a non-one-hot [len(str)] tensor representation of all character classes in the string. Used for labels in NLLLoss, which are not one-hot.
def stringToTensorNLLLabel(str):
    tensor = torch.zeros(len(str), device=device, dtype=torch.long)
    for i in range(len(str)):
        tensor[i] = char_to_ind[str[i]]
    return tensor

# Convert an integer to a [1, K] one-hot tensor representation
def toOneHot(val):
    tensor = torch.zeros(1, K, device=device)
    tensor[0, val] = 1
    return tensor

# Convert an array of character class integers (non-one-hot) to the corresponding string representation
def indsToString(inds):
    str = ""
    for i in inds:
        str += ind_to_char[i]
    return str

# Returns a random index from the weights. The probability of an index being selected is given by its corresponding weight value.
# (experimentally verified)
def randomSampleFromWeights(weights):
    csum = weights.cumsum(0)
    a = random.uniform(0, 1)
    b = (csum - a)
    c = (b > 0)
    ixs = c.nonzero()
    ii = ixs[0].item()
    return ii

# Synthesize a text sequence of length n
def synthesize(n):
    hidden = net.initHidden().to(device)
    net.zero_grad()
    prev_char = toOneHot(char_to_ind['.'])
    word = []
    for i in range(n):
        '''
            Select the next character based on the probability distribution.
            Note: we don't always select the most likely character, as that would
            likely result in repeating phrases such as "the the the the the..."
        '''
        output, hidden = net(prev_char, hidden)
        # Convert output to probability weights. 
        # exp is needed to invert log softmax.
        output = torch.exp(output)
        char_index = randomSampleFromWeights(output[0])
        word.append(char_index)
        prev_char = toOneHot(char_index)
    return word

# Train the network on a single character sequence
def train(input_seq_tensor, target_seq_tensor, hidden):
    target_seq_tensor.unsqueeze_(-1)
    # hidden = net.initHidden()
    net.zero_grad()
    optimizer.zero_grad()

    loss = 0

    # Loop through each character in the sequence
    for i in range(input_seq_tensor.size(0)):
        output, hidden = net(input_seq_tensor[i], hidden)
        l = criterion(output, target_seq_tensor[i])
        loss += l

    loss.backward()
    optimizer.step()

    return output, loss.item(), hidden.detach()

'''
    Train the network
'''

print("Training progress: ")

smooth_loss = 0
smooth_interpolation_rate = 0.02
smooth_loss_vec = []
loss_vec = []
# Current inner loop iteration (total)
current_iteration = 0
expected_number_of_iterations = (len(book_data) / seq_length) * n_epochs    # (approximative)

start_time = time.time()

# One epoch = one full run through the training data (such as goblet_book.txt)
for epoch in range(n_epochs):
    i=0
    hidden = net.initHidden().to(device)
    # One iteration = one sequence of text data (such as 25 characters)
    while i < (len(book_data) - seq_length):
        X_chars = book_data[i:i+seq_length]
        Y_chars = book_data[i+1:i+seq_length+1]
        X = stringToTensor(X_chars)
        Y = stringToTensorNLLLabel(Y_chars)
        output, loss, hidden = train(X, Y, hidden)
        if current_iteration == 0:
            smooth_loss = loss
        else:
            smooth_loss = smooth_loss * (1 - smooth_interpolation_rate) + loss * smooth_interpolation_rate
        
        percent_done = round((current_iteration / expected_number_of_iterations) * 100)
        if current_iteration % 10 == 0:
            print("\t" + str(percent_done) + " % done. Smooth loss: " +  str("{:.2f}").format(smooth_loss), end="\r")
        i += seq_length
        current_iteration += 1
        loss_vec.append(loss)
        smooth_loss_vec.append(smooth_loss)

end_time = time.time() - start_time
print("\t100% done. Smooth loss: " +  str("{:.2f}").format(smooth_loss))
print()
print("Total training time: " + str(round(end_time)) + " seconds")

'''
    Synthesize some text
'''
text_inds = synthesize(syn_length)
print()
print("Synthesized text:")
print("\t" + indsToString(text_inds))
print()

'''
    Plot loss
    
    TODO: Instead of plotting "unsmooth" loss, do multiple runs 
    and plot average loss and standard deviation between runs.
'''
plt.plot(loss_vec, 'lightblue')
plt.plot(smooth_loss_vec, 'blue')
plt.legend(['Iteration loss', 'Smooth loss'])
title = "Baseline RNN loss evolution (M=" + str(n_hidden) + ", seq_len=" + str(seq_length) + ", eta=" + str(learning_rate) + ")"
plt.title(title)
plt.xlabel("Training iteration")
plt.ylabel("Training loss")
plt.xlim(0, len(smooth_loss_vec))
# Note: y-max is quite arbitrary and depends on the loss metric and data
plt.ylim(0, 150)
plt.show()
