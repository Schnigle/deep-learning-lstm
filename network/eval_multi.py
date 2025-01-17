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
import rnn_char_net
import rnn_char_train
import lstm_char_net
import lstm_char_train
import data
import utility
import sys

#Check input arguments
args = sys.argv
if len(args) < 2:
	print("Please specify a save file(s).")
	exit()

file_paths = []
for i in range(1,len(args),1):
    file_path = args[i]
    if not os.path.isfile(file_path):
    	print("File not found:", file_path)
    	exit()
    file_paths.append(file_path)

num_runs = len(args)-1
losses = []
s_losses = []
labels = []

checkpoint = torch.load(file_paths[0])
min_length_losses = len(checkpoint['loss_vec'])
min_length_s_losses = len(checkpoint['smooth_loss_vec'])
for i in range(num_runs):
    checkpoint = torch.load(file_paths[i])
    if len(checkpoint['loss_vec']) < min_length_losses:
        min_length = checkpoint['loss_vec']
    if len(checkpoint['smooth_loss_vec']) < min_length_s_losses:
        min_length = checkpoint['smooth_loss_vec']
for i in range(num_runs):
    checkpoint = torch.load(file_paths[i])
    labels.append(checkpoint['config_text'])
    losses.append(checkpoint['loss_vec'][0:min_length_losses])
    s_losses.append(checkpoint['smooth_loss_vec'][0:min_length_s_losses])

losses = np.array(losses)
s_losses = np.array(s_losses)
plot_labels = []

#Create plot
for i in range(num_runs):
    plot_labels.append("Loss for: " + labels[i])
    plot_labels.append("Smooth loss for: " + labels[i])
    plt.plot(losses[i])
    plt.plot(s_losses[i])
plt.legend(plot_labels, prop={'size':7})
title = "Loss evolution of multiple training runs."
plt.title(title)
plt.xlabel("Training iteration")
plt.ylabel("Training loss")
#plt.xlim(0, len(smooth_loss_vec))
# Note: y-max is quite arbitrary and depends on the loss metric and data
#plt.ylim(0, 150)
if not os.path.exists('graphs'):
    os.makedirs('graphs')
plt.savefig('graphs/multi_plot.png')
plt.show()
