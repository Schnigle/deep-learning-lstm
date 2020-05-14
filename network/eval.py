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
import rnn_bert_net
import rnn_char_train
import rnn_word_net
import rnn_word_net_naive
import rnn_word_train
import rnn_bert_train
import lstm_char_net
import lstm_char_train
import lstm_word_net
import lstm_word_train
import data
import utility
import sys

syn_length = 1000
syn_beam_search = False
beam_search_width = 6
beam_search_sampler = 'Weighted' # 'WeightedNoReplacement', 'Weighted', 'Random' and 'Topk'
seed = random.randint(1, 10000)
seed = 999
interpolation_rate_train = 0.01
interpolation_rate_val = 0.1

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
net, data_loader, synth = utility.loadNet(checkpoint, syn_beam_search)
net.load_state_dict(checkpoint['model_state_dict'])
loss_vec = checkpoint['loss_vec']
smooth_loss_vec = checkpoint['smooth_loss_vec']

'''
    Synthesize some text
'''
if checkpoint['module_id'] == 'rnn_bert':
	text = synth(data_loader, net, syn_length, torch.device('cpu'))
else:
	if syn_beam_search:
		text_inds = synth(data_loader, net, syn_length, torch.device('cpu'), beam_search_width, beam_search_sampler)
	else:
		text_inds = synth(data_loader, net, syn_length, torch.device('cpu'))
	text = data_loader.indsToString(text_inds)
print()
print("Synthesized text:")
print("\t" + text)
print()
print(checkpoint['config_text'])

smooth_loss_vec = utility.smoothed(loss_vec, interpolation_rate_train)
print("Final smooth training loss: ", smooth_loss_vec[-1])

'''
    Plot loss
    
    TODO: Instead of plotting "unsmooth" loss, do multiple runs 
    and plot average loss and standard deviation between runs.
'''
plt.plot(smooth_loss_vec, 'blue')
if 'val_loss_vec' in checkpoint:
	val_loss_vec = checkpoint['val_loss_vec']
	smooth_val_loss_vec = utility.smoothed(val_loss_vec, interpolation_rate_val)
	print("Final smooth validation loss: ", smooth_val_loss_vec[-1])
	val_scale_factor = round(len(loss_vec) / len(val_loss_vec))
	iterations_per_epoch = round(data_loader.n_samples / checkpoint['batch_size'] / checkpoint['seq_length'])
	plt.plot(range(iterations_per_epoch, len(smooth_val_loss_vec) * val_scale_factor + iterations_per_epoch, val_scale_factor), smooth_val_loss_vec, 'orange')
	
plt.legend(['Training loss', 'Validation loss'])
title = "Loss evolution of " + checkpoint['config_text']
plt.title(title)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.xlim(0, len(smooth_loss_vec))
# Note: y-max is quite arbitrary and depends on the loss metric and data
plt.ylim(0, 10)
print(flush=True)
plt.show()
