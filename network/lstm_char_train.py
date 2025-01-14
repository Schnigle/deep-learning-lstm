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
import copy

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
	val_loss_vec = []
	# Current inner loop iteration (total)
	current_iteration = 0
	expected_number_of_iterations = (len(data.train_data) / (seq_length * batch_size)) * n_epochs    # (approximative)

	start_time = time.time()
	
	state_dict_save = None
	last_validation_loss = 0

	# One epoch = one full run through the training data (such as goblet_book.txt)
	for epoch in range(n_epochs):
		if state_dict_save != None:
			net.load_state_dict(state_dict_save)
		i=0
		hidden = net.initHidden(batch_size, device)
		# One iteration = one sequence of text data (such as 25 characters)
		while i < (len(data.train_data) - seq_length * batch_size):
			# prepare data batches
			X_batch = torch.zeros(seq_length, batch_size, data.K).to(device)
			Y_batch = torch.zeros(seq_length, batch_size, dtype=torch.long).to(device)
			for j in range(batch_size):
				X_chars = data.train_data[i:i+seq_length]
				Y_chars = data.train_data[i+1:i+seq_length+1]
				X = data.stringToTensor(X_chars)
				Y = data.stringToTensorNLLLabel(Y_chars)
				X_batch[:, j, :] = X.squeeze(1)
				Y_batch[:, j] = Y
				i += seq_length
			output, loss, hidden = train_batch(net, criterion, optimizer, X_batch, Y_batch, hidden)
			if current_iteration == 0:
				smooth_loss = loss
			else:
				smooth_loss = smooth_loss * (1 - smooth_interpolation_rate) + loss * smooth_interpolation_rate
			
			current_iteration += 1
			percent_done = round((current_iteration / expected_number_of_iterations) * 100)
			print("\t" + str(percent_done) + "% done. Smooth loss: " +  str("{:.2f}").format(smooth_loss) + ". Last validation loss: " + str("{:0.2f}").format(last_validation_loss) + "\t\t\t", end="\r")
			loss_vec.append(loss)
			smooth_loss_vec.append(smooth_loss)
		# Dynamic evaluation - let evaluation update weights and revert to the previous weights when training
		state_dict_save = copy.deepcopy(net.state_dict())
		i=0
		first_val_batch = True
		hidden = net.initHidden(batch_size, device)
		# One iteration = one sequence of text data (such as 25 characters)
		while i + batch_size * seq_length + 1 < len(data.val_data):
			# prepare data batches
			X_batch = torch.zeros(seq_length, batch_size, data.K).to(device)
			Y_batch = torch.zeros(seq_length, batch_size, dtype=torch.long).to(device)
			for j in range(batch_size):
				X_chars = data.val_data[i:i+seq_length]
				Y_chars = data.val_data[i+1:i+seq_length+1]
				X = data.stringToTensor(X_chars)
				Y = data.stringToTensorNLLLabel(Y_chars)
				X_batch[:, j, :] = X.squeeze(1)
				Y_batch[:, j] = Y
				i += seq_length
			output, loss, hidden = train_batch(net, criterion, optimizer, X_batch, Y_batch, hidden)
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
	hidden = net.initHidden(1, device)
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

def synthesize_characters_beam(data, net, n, device, k, sampler='Topk'):

	def synthesize_step(words, scores, hiddens):
		# Takes words, scores and hiddens and expands them
		n_in = len(words)
		nextWords = [0]*data.K*n_in
		nextScores = torch.empty(data.K*n_in)
		nextHiddens = [0]*n_in
		for i in range(n_in):
			output, nextHiddens[i] = net(data.toOneHot(words[i][-1]).unsqueeze_(0), hiddens[i])
			output = output[0]
			nextWords[i*data.K:(i+1)*data.K] = [words[i] + [x] for x in range(data.K)]
			nextScores[i*data.K:(i+1)*data.K] = F.softmax(output, dim=1)[0]*scores[i]
		return nextWords, nextScores, nextHiddens

	hidden = net.initHidden(1, device)
	net.zero_grad()

	firstchar = data.char_to_ind['.']
	words = [[firstchar]]
	scores = torch.tensor([1])
	hiddens = [hidden]
	for i in range(n):
		# print([data.indsToString(x) for x in words])
		# print(scores)
		words, scores, hiddens = synthesize_step(words, scores, hiddens)
		scores = scores/torch.sum(scores) # normalize to prevent underflow

		# reduces the expanded words, scores and hiddens, to always have k of them
		if sampler == 'WeightedNoReplacement':
			tmp_scores = scores[:]
			idx = [0]*k
			for i in range(k):
				idx[i] = utility.randomSampleFromWeights(tmp_scores)
				tmp_scores[idx[i]] = 0
				tmp_scores = tmp_scores/torch.sum(tmp_scores)
		elif sampler == 'Weighted':
			idx = [utility.randomSampleFromWeights(scores) for x in range(k)]
		elif sampler == 'Random':
			idx = torch.randint(len(words), (1,k))[0].tolist()
		elif sampler == 'Topk':
			idx = torch.topk(scores, k=k, dim=0, largest=True)[1].tolist()

		idx2 = [x//data.K for x in idx]
		words = [words[x] for x in idx]
		scores = scores[idx]
		hiddens = [hiddens[x] for x in idx2]

	idx = torch.argmax(scores)
	word = words[idx][1:]
	return word
