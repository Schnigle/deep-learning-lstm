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

# Train the network on a single character sequence
def train_batch(net, criterion, optimizer, input_seq_tensor, target_seq_tensor, hidden_tuple):
	# NOTE: Pytorch LSTM expects input to be 3D tensors with dimensions SEQUENCE_LENGTH x BATCH_SIZE x N_CLASSES.
	(hidden, cell) = hidden_tuple
	net.zero_grad()
	optimizer.zero_grad()
	output, (hidden, cell) = net(input_seq_tensor, (hidden, cell))
	# For some reason we need to switch some dimensions
	target_seq_tensor.transpose_(0, 1)
	output.transpose_(0, 1)
	output.transpose_(1, 2)
	# TODO: Multiply with sequence length to better match sequential approach?
	loss = criterion(output, target_seq_tensor)

	loss.backward()
	optimizer.step()

	return output, loss.item(), (hidden.detach(), cell.detach())

def train_net(net, criterion, optimizer, data, n_hidden, seq_length, n_epochs, learning_rate, batch_size, device):
	print("Training progress: ")
	smooth_loss = 0
	smooth_interpolation_rate = 0.02
	smooth_loss_vec = []
	loss_vec = []
	# Current inner loop iteration (total)
	current_iteration = 0
	expected_number_of_iterations = (len(data.word_data) / (seq_length * batch_size)) * n_epochs    # (approximative)

	start_time = time.time()

	# One epoch = one full run through the training data (such as goblet_book.txt)
	for epoch in range(n_epochs):
		i=0
		hidden = net.initHidden(batch_size, device)
		# One iteration = one sequence of text data (such as 25 characters)
		while i + batch_size * seq_length + 1 < len(data.word_data):
			# prepare data batches
			# X_batch = torch.zeros(seq_length, batch_size, 1, dtype=torch.long).to(device)
			X_batch = torch.zeros(batch_size, seq_length, dtype=torch.long).to(device)
			Y_batch = torch.zeros(batch_size, seq_length, dtype=torch.long).to(device)
			# X_batch = []
			# Y_batch = []
			for j in range(batch_size):
				X_chars = data.word_data[i:i+seq_length]
				Y_chars = data.word_data[i+1:i+seq_length+1]
				# X = data.stringToTensor(X_chars)
				X = torch.zeros(seq_length, dtype=torch.long)
				Y = torch.zeros(seq_length, dtype=torch.long)
				for k in range(seq_length):
					X[k] = data.word_to_ind[X_chars[k]]
					# print("k", k)
					# print("Y", Y.size())
					# print("Y_chars", len(Y_chars))
					# print("X_chars", len(X_chars))
					Y[k] = data.word_to_ind[Y_chars[k]]
					# print([data.word_to_ind[X_chars[k]]])
					# print(torch.tensor([data.word_to_ind[X_chars[k]]]))
				# Y = data.stringToTensorNLLLabel(Y_chars)
				X_batch[j, :] = X
				Y_batch[j, :] = Y
				i += seq_length
				# print(Y_batch)
			X_batch.transpose_(0, 1)
			Y_batch.transpose_(0, 1)
			output, loss, hidden = train_batch(net, criterion, optimizer, X_batch, Y_batch, hidden)
			if current_iteration == 0:
				smooth_loss = loss
			else:
				smooth_loss = smooth_loss * (1 - smooth_interpolation_rate) + loss * smooth_interpolation_rate
			
			current_iteration += 1
			percent_done = round((current_iteration / expected_number_of_iterations) * 100)
			if current_iteration % 5 == 0:
				print("\t" + str(percent_done) + " % done. Smooth loss: " +  str("{:.2f}").format(smooth_loss), end="\r")
			loss_vec.append(loss)
			smooth_loss_vec.append(smooth_loss)

	total_time = time.time() - start_time
	print("\t100% done. Smooth loss: " +  str("{:.2f}").format(smooth_loss))
	print()
	print("Total training time: " + str(round(total_time)) + " seconds")

	return loss_vec, smooth_loss_vec

# Synthesize a text sequence of length n
def synthesize_characters(data, net, n, device):
	hidden = net.initHidden(1, device)
	net.zero_grad()
	prev_char = torch.tensor([data.word_to_ind["."]], dtype=torch.long, device=device)
	sentence = []
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
		sentence.append(char_index)
		prev_char = torch.tensor([char_index], dtype=torch.long, device=device)
	return sentence
