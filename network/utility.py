'''
	Module for miscellaneous utility functions.
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
import rnn_char_net
import rnn_char_train
import rnn_word_net
import rnn_word_net_naive
import rnn_word_train
import rnn_bert_net
import rnn_bert_train
import lstm_char_net
import lstm_char_train
import lstm_word_net
import lstm_word_train
import data
import utility
import sys

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

def loadNet(checkpoint, use_beam_search):
	module_id = checkpoint['module_id']
	if module_id == 'lstm_char':
		data_loader = data.CharacterData(checkpoint['input_file_name'], torch.device('cpu'), 0)
		net = lstm_char_net.RNN_LSTM(checkpoint['K'], checkpoint['n_hidden'], checkpoint['K'], checkpoint['n_layers'])
		if use_beam_search:
			synth = lstm_char_train.synthesize_characters_beam
		else:
			synth = lstm_char_train.synthesize_characters
	elif module_id == 'rnn_char':
		data_loader = data.CharacterData(checkpoint['input_file_name'], torch.device('cpu'), 0)
		net = rnn_char_net.RNN(checkpoint['K'], checkpoint['n_hidden'], checkpoint['K'])
		synth = rnn_char_train.synthesize_characters
		if use_beam_search:
			synth = rnn_char_train.synthesize_characters_beam
		else:
			synth = rnn_char_train.synthesize_characters
	elif module_id == 'lstm_word':
		data_loader = data.WordData(checkpoint['input_file_name'], torch.device('cpu'), 0)
		net = lstm_word_net.RNN_LSTM(checkpoint['K'], checkpoint['n_hidden'], checkpoint['K'], checkpoint['n_layers'], data_loader.K, checkpoint['embedding_dim'])
		if use_beam_search:
			synth = lstm_word_train.synthesize_characters_beam
		else:
			synth = lstm_word_train.synthesize_characters
	elif module_id == 'rnn_word':
		data_loader = data.WordData(checkpoint['input_file_name'], torch.device('cpu'), 0)
		net = rnn_word_net.RNN(checkpoint['K'], checkpoint['n_hidden'], checkpoint['K'], data_loader.K, checkpoint['embedding_dim'])
		if use_beam_search:
			synth = rnn_word_train.synthesize_characters_beam
		else:
			synth = rnn_word_train.synthesize_characters
	elif module_id == 'rnn_word_naive':
		data_loader = data.WordData(checkpoint['input_file_name'], torch.device('cpu'), 0)
		net = rnn_word_net_naive.RNN(checkpoint['K'], checkpoint['n_hidden'], checkpoint['K'], checkpoint['embedding_dim'])
		if use_beam_search:
			synth = rnn_word_train.synthesize_characters_beam
		else:
			synth = rnn_word_train.synthesize_characters
	elif module_id == 'rnn_bert':
		net = rnn_bert_net.RNN(checkpoint['K'], checkpoint['n_hidden'], checkpoint['K'])
		data_loader = data.VecData(checkpoint['input_file_name'], torch.device('cpu'))
		synth = rnn_bert_train.synthesize_words
	return net, data_loader, synth
