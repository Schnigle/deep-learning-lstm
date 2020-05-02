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
import lstm_char_net
import time
import utility
import torch.nn.functional as F
from IPython.display import HTML

# Train the network on a single character sequence
def train_batch(net, criterion, optimizer, input_seq_tensor, target_seq_tensor, hidden):
	# NOTE: Pytorch LSTM expects 3D tensors with dimensions SEQUENCE_LENGTH x BATCH_SIZE x N_CLASSES.
	# Currently we are training one sequence sample at the time to make it easier to compare to the baseline RNN,
	# giving each input tensor (representing a char) the dimensions 1 x 1 x N_CLASSES.
	input_seq_tensor.unsqueeze_(1)
	target_seq_tensor.unsqueeze_(-1)
	net.zero_grad()
	optimizer.zero_grad()

	loss = 0

	# Loop through each character in the sequence
	for i in range(input_seq_tensor.size(0)):
		output, hidden = net(input_seq_tensor[i], hidden)
		l = criterion(output[0], target_seq_tensor[i])
		loss += l

	loss.backward()
	optimizer.step()

	return output, loss.item(), (hidden[0].detach(), hidden[1].detach())

def train_net(net, criterion, optimizer, data, n_hidden, seq_length, n_epochs, learning_rate, device):
	print("Training progress: ")
	smooth_loss = 0
	smooth_interpolation_rate = 0.02
	smooth_loss_vec = []
	loss_vec = []
	# Current inner loop iteration (total)
	current_iteration = 0
	expected_number_of_iterations = (len(data.text_data) / seq_length) * n_epochs    # (approximative)

	start_time = time.time()

	# One epoch = one full run through the training data (such as goblet_book.txt)
	for epoch in range(n_epochs):
		i=0
		hidden = net.initHidden(device)
		# One iteration = one sequence of text data (such as 25 characters)
		while i < (len(data.text_data) - seq_length):
			X_chars = data.text_data[i:i+seq_length]
			Y_chars = data.text_data[i+1:i+seq_length+1]
			X = data.stringToTensor(X_chars)
			Y = data.stringToTensorNLLLabel(Y_chars)
			output, loss, hidden = train_batch(net, criterion, optimizer, X, Y, hidden)
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

	total_time = time.time() - start_time
	print("\t100% done. Smooth loss: " +  str("{:.2f}").format(smooth_loss))
	print()
	print("Total training time: " + str(round(total_time)) + " seconds")

	return loss_vec, smooth_loss_vec

# Synthesize a text sequence of length n
def synthesize_characters(data, net, n, device):
	hidden = net.initHidden(device)
	net.zero_grad()
	prev_char = data.toOneHot(data.char_to_ind['.'])
	word = []
	for i in range(n):
		'''
			Select the next character based on the probability distribution.
			Note: we don't always select the most likely character, as that would
			likely result in repeating phrases such as "the the the the the..."
		'''
		prev_char.unsqueeze_(0)
		output, hidden = net(prev_char, hidden)
		output = output[0]
		# Convert output to probability weights. 
		output = F.softmax(output, dim=1)
		char_index = utility.randomSampleFromWeights(output[0])
		word.append(char_index)
		prev_char = data.toOneHot(char_index)
	return word
