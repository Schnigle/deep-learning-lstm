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
import rnn_char_net
import time
import utility
import torch.nn.functional as F
import copy

# Train the network on a single character sequence
def train_batch(net, criterion, optimizer, input_seq_tensor, target_seq_tensor, hidden):
	target_seq_tensor.unsqueeze_(-1)
	input_seq_tensor.unsqueeze_(-1)
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

	return output, loss.item() / input_seq_tensor.size(0), hidden.detach()

def train_net(net, criterion, optimizer, data, n_hidden, seq_length, n_epochs, learning_rate, device):
	print("Training progress: ")
	smooth_loss = 0
	smooth_interpolation_rate = 0.02
	smooth_loss_vec = []
	loss_vec = []
	val_loss_vec = []
	# Current inner loop iteration (total)
	current_iteration = 0
	expected_number_of_iterations = (len(data.train_data) / (seq_length)) * n_epochs    # (approximative)

	start_time = time.time()

	state_dict_save = None
	last_validation_loss = 0

	# One epoch = one full run through the training data (such as goblet_book.txt)
	for epoch in range(n_epochs):
		if state_dict_save != None:
			net.load_state_dict(state_dict_save)
		i=0
		hidden = net.initHidden(device)
		# One iteration = one sequence of text data (such as 25 characters)
		while i < (len(data.train_data) - seq_length):
			X_words = data.train_data[i:i+seq_length]
			Y_words = data.train_data[i+1:i+seq_length+1]
			X = torch.zeros(seq_length, dtype=torch.long, device=device)
			Y = torch.zeros(seq_length, dtype=torch.long, device=device)
			for k in range(seq_length):
				X[k] = data.word_to_ind[X_words[k]]
				Y[k] = data.word_to_ind[Y_words[k]]
			output, loss, hidden = train_batch(net, criterion, optimizer, X, Y, hidden)
			if current_iteration == 0:
				smooth_loss = loss
			else:
				smooth_loss = smooth_loss * (1 - smooth_interpolation_rate) + loss * smooth_interpolation_rate
			
			percent_done = round((current_iteration / expected_number_of_iterations) * 100)
			print("\t" + str(percent_done) + "% done. Smooth loss: " +  str("{:.2f}").format(smooth_loss) + ". Last validation loss: " + str("{:0.2f}").format(last_validation_loss) + "\t\t\t", end="\r")
			i += seq_length
			current_iteration += 1
			loss_vec.append(loss)
			smooth_loss_vec.append(smooth_loss)
		# Dynamic evaluation - let evaluation update weights and revert to the previous weights when training
		state_dict_save = copy.deepcopy(net.state_dict())
		i=0
		first_val_batch = True
		hidden = net.initHidden(device)
		# One iteration = one sequence of text data (such as 25 characters)
		while i + seq_length + 1 < len(data.val_data):
			X_words = data.val_data[i:i+seq_length]
			Y_words = data.val_data[i+1:i+seq_length+1]
			X = torch.zeros(seq_length, dtype=torch.long, device=device)
			Y = torch.zeros(seq_length, dtype=torch.long, device=device)
			for k in range(seq_length):
				X[k] = data.word_to_ind[X_words[k]]
				Y[k] = data.word_to_ind[Y_words[k]]
			output, loss, hidden = train_batch(net, criterion, optimizer, X, Y, hidden)
			i += seq_length
			if first_val_batch:
				last_validation_loss = loss
				first_val_batch = False
			else:
				last_validation_loss = last_validation_loss * (1 - smooth_interpolation_rate * 2) + loss * smooth_interpolation_rate * 2
			val_loss_vec.append(loss)
			print("\t" + str(percent_done) + "% done. Smooth loss: " +  str("{:.2f}").format(smooth_loss) + ". Last validation loss: " + str("{:0.2f}").format(last_validation_loss) + "\t\t\t", end="\r")
			

	total_time = time.time() - start_time
	print("\t100% done. Smooth loss: " +  str("{:.2f}").format(smooth_loss) + ". Last validation loss: " + str("{:0.2f}").format(last_validation_loss) + "\t\t\t", end="\r")
	print()
	print("Total training time: " + str(round(total_time)) + " seconds")

	return loss_vec, smooth_loss_vec, val_loss_vec

# Synthesize a text sequence of length n
def synthesize_characters(data, net, n, device):
	hidden = net.initHidden(device)
	net.zero_grad()
	prev_char = torch.tensor([data.word_to_ind["."]], dtype=torch.long, device=device)
	sentence = []
	for i in range(n):
		'''
			Select the next character based on the probability distribution.
			Note: we don't always select the most likely character, as that would
			likely result in repeating phrases such as "the the the the the..."
		'''
		output, hidden = net(prev_char, hidden)
		# Convert output to probability weights. 
		output = F.softmax(output, dim=1)
		char_index = utility.randomSampleFromWeights(output[0])
		sentence.append(char_index)
		prev_char = torch.tensor([char_index], dtype=torch.long, device=device)
	return sentence

def synthesize_characters_beam(data, net, n, device, k, sampler='Topk'):
	print("Error: Beam search has not yet been implemented for this network.")
	exit()
