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
import rnn_bert_train
import lstm_char_net
import lstm_char_train
import data
import utility
import sys

syn_length = 500
syn_beam_search = True
beam_search_width = 30
beam_search_sampler = 'WeightedNoReplacement' # 'WeightedNoReplacement', 'Weighted', 'Random' and 'Topk'
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
	net = lstm_char_net.RNN_LSTM(checkpoint['K'], checkpoint['n_hidden'], checkpoint['K'], checkpoint['n_layers'])
	synth = lstm_char_train.synthesize_characters
elif module_id == 'rnn_char':
	net = rnn_char_net.RNN(checkpoint['K'], checkpoint['n_hidden'], checkpoint['K'])
	if syn_beam_search:
		synth = rnn_char_train.synthesize_characters_beam
	else:
		synth = rnn_char_train.synthesize_characters
elif module_id == 'rnn_bert':
	net = rnn_bert_net.RNN(checkpoint['K'], checkpoint['n_hidden'], checkpoint['K'])
	synth = rnn_bert_train.synthesize_words
net.load_state_dict(checkpoint['model_state_dict'])
loss_vec = checkpoint['loss_vec']
smooth_loss_vec = checkpoint['smooth_loss_vec']
if module_id == 'rnn_bert':
	data = data.VecData(checkpoint['input_file_name'], torch.device('cpu'))
else:
	data = data.CharacterData(checkpoint['input_file_name'], torch.device('cpu'))

'''
    Synthesize some text
'''
if module_id == 'rnn_bert':
	text = synth(data, net, syn_length, torch.device('cpu'))
else:
	if syn_beam_search:
		text_inds = synth(data, net, syn_length, torch.device('cpu'), beam_search_width, beam_search_sampler)
	else:
		text_inds = synth(data, net, syn_length, torch.device('cpu'))
	text = data.indsToString(text_inds)
print()
print("Synthesized text:")
print("\t" + text)
print()

'''
    Plot loss
    
    TODO: Instead of plotting "unsmooth" loss, do multiple runs 
    and plot average loss and standard deviation between runs.
'''
plt.plot(loss_vec, 'lightblue')
plt.plot(smooth_loss_vec, 'blue')
plt.legend(['Iteration loss', 'Smooth loss'])
title = "Loss evolution of " + checkpoint['config_text']
plt.title(title)
plt.xlabel("Training iteration")
plt.ylabel("Training loss")
plt.xlim(0, len(smooth_loss_vec))
# Note: y-max is quite arbitrary and depends on the loss metric and data
plt.ylim(0, 150)
plt.show()
