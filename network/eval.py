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
import lstm_word_net
import lstm_word_train
import data
import utility
import sys

syn_length = 1000
seed = random.randint(1, 10000)
# seed = 999

args = sys.argv
if len(args) < 2:
	print("Please specify a save file.")
	exit()
file_path = args[1]
if not os.path.isfile(file_path):
	print("File not found:", file_path)
	exit()

torch.manual_seed(seed)
random.seed(seed)

checkpoint = torch.load(file_path)
module_id = checkpoint['module_id']
if module_id == 'lstm_char':
	data = data.CharacterData(checkpoint['input_file_name'], torch.device('cpu'))
	net = lstm_char_net.RNN_LSTM(checkpoint['K'], checkpoint['n_hidden'], checkpoint['K'], checkpoint['n_layers'])
	synth = lstm_char_train.synthesize_characters
elif module_id == 'rnn_char':
	data = data.CharacterData(checkpoint['input_file_name'], torch.device('cpu'))
	net = rnn_char_net.RNN(checkpoint['K'], checkpoint['n_hidden'], checkpoint['K'])
	synth = rnn_char_train.synthesize_characters
elif module_id == 'lstm_word':
	data = data.WordData(checkpoint['input_file_name'], torch.device('cpu'), 0)
	net = lstm_word_net.RNN_LSTM(checkpoint['K'], checkpoint['n_hidden'], checkpoint['K'], checkpoint['n_layers'], data.K, checkpoint['embedding_dim'])
	synth = lstm_word_train.synthesize_characters
net.load_state_dict(checkpoint['model_state_dict'])
loss_vec = checkpoint['loss_vec']
smooth_loss_vec = checkpoint['smooth_loss_vec']

'''
    Synthesize some text
'''
text_inds = synth(data, net, syn_length, torch.device('cpu'))
print()
print("Synthesized text:")
print("\t" + data.indsToString(text_inds))
print()

print(checkpoint['config_text'])

'''
    Plot loss
    
    TODO: Instead of plotting "unsmooth" loss, do multiple runs 
    and plot average loss and standard deviation between runs.
'''
val_loss_vec = checkpoint['val_loss_vec']
smooth_val_loss_vec = []
smooth_val = val_loss_vec[0]
interpolation_rate = 0.01
val_factor = round(len(loss_vec) / len(val_loss_vec))
for val in val_loss_vec:
	smooth_val = smooth_val * (1 - interpolation_rate) + val * interpolation_rate
	smooth_val_loss_vec.append(smooth_val)
# iterations_per_epoch = round(len(loss_vec) / checkpoint['n_epochs'])
iterations_per_epoch = round(data.n_samples / checkpoint['batch_size'] / checkpoint['seq_length'])
plt.plot(smooth_loss_vec, 'blue')
plt.plot(range(iterations_per_epoch, len(smooth_val_loss_vec) * val_factor + iterations_per_epoch, val_factor), smooth_val_loss_vec, 'orange')
# print(val_loss_vec)
# plt.scatter(range(0, len(val_loss_vec)), val_loss_vec, 'lightblue')
plt.legend(['Training loss', 'Validation loss'])
title = "Loss evolution of " + checkpoint['config_text']
plt.title(title)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.xlim(0, len(smooth_loss_vec))
# Note: y-max is quite arbitrary and depends on the loss metric and data
plt.ylim(0, 10)
plt.show()
