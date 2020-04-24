'''
    Entry point for training a character-level RNN.
'''

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
import time
import networks
import data
import training_char
import utility

'''
    Network and synthesis parameters
'''
input_file_name = "data/speech.txt"
n_hidden = 50
seq_length = 25
syn_length = 500
n_epochs = 200
learning_rate = 0.1
seed = random.randint(1, 10000)
# seed = 999
use_cuda = False
'''
    Note: Using the GPU is currently only beneficial for very large network
    sizes since the batches are processed sequentially. For smaller networks
    GPU is much slower than CPU.
'''

if use_cuda and not torch.cuda.is_available():
    print("No CUDA support found. Switching to GPU mode. ")
    use_cuda = False

print("Input file: ")
print("\t" + input_file_name)
print()
print("Parameters: ")
print("\tHidden nodes M: ", n_hidden)
print("\tSequence length: ", seq_length)
print("\tLearning rate: ", learning_rate)
print("\tNumber of epochs: ", n_epochs)
print("\tRandom seed: ", seed)
print("\tGPU: ", use_cuda)
print()

if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
data = data.CharacterData(input_file_name, device)

'''
    Create network and loss criterion
'''
net = networks.RNN(data.K, n_hidden, data.K)
if use_cuda:
    net = net.cuda()
criterion = nn.NLLLoss()
optimizer = optim.Adagrad(net.parameters(), lr=learning_rate)

loss_vec, smooth_loss_vec = training_char.train_net(net, criterion, optimizer, data, n_hidden, seq_length, n_epochs, learning_rate, device)

'''
    Synthesize some text
'''
text_inds = utility.synthesize_characters(data, net, syn_length, device)
print()
print("Synthesized text:")
print("\t" + data.indsToString(text_inds))
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
