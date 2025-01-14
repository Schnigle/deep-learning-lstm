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
import lstm_word_net
import lstm_word_train
import data
import utility

'''
    Network and synthesis parameters
'''
# input_file_name = "data/speech_long.txt"
input_file_name = "data/goblet_book.txt"
save_file_name = "lstm_word_save.pt"
n_hidden = 400
n_layers = 2
seq_length = 25
syn_length = 500
syn_beam_search = False
beam_search_width = 30
beam_search_sampler = 'Weighted' # 'WeightedNoReplacement', 'Weighted', 'Random' and 'Topk'
n_epochs = 1
learning_rate = 0.01
batch_size = 20
embedding_dim = 1000
validation_factor = 0.1
# seed = random.randint(1, 10000)
seed = 999
use_cuda = True
'''
    Note: Using the GPU is currently only beneficial for very large network
    sizes since the batches are processed sequentially. For smaller net_lstm
    GPU is much slower than CPU.
'''
if use_cuda and not torch.cuda.is_available():
    print("No CUDA support found. Switching to GPU mode. ")
    use_cuda = False
if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

'''
    Create network and loss criterion
'''
torch.manual_seed(seed)
random.seed(seed)
data = data.WordData(input_file_name, device, validation_factor)
net = lstm_word_net.RNN_LSTM( data.K, n_hidden, data.K, n_layers, data.K, embedding_dim)
if use_cuda:
    net = net.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adagrad(net.parameters(), lr=learning_rate)

print("Input file: ")
print("\t" + input_file_name)
print()
print("Parameters: ")
print("\tHidden nodes M: ", n_hidden)
print("\tLSTM layers: ", n_layers)
print("\tSequence length: ", seq_length)
print("\tEmbedding size: ", embedding_dim)
print("\tLearning rate: ", learning_rate)
print("\tNumber of epochs: ", n_epochs)
print("\tBatch size: ", batch_size)
print("\tRandom seed: ", seed)
print("\tGPU: ", use_cuda)
print("\tValidation data factor: ", validation_factor)
print()

loss_vec, smooth_loss_vec, val_loss_vec = lstm_word_train.train_net(net, criterion, optimizer, data, n_hidden, seq_length, n_epochs, learning_rate, batch_size, device)

'''
    Save network and training data
'''
save_folder = 'saves'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
module_id = "lstm_word"
config_text = "LSTM (M=" + str(n_hidden) + ", n_layers=" + str(n_layers) + ", embed_dim=" + str(embedding_dim) + ", seq_len=" + str(seq_length) + ", eta=" + str(learning_rate) + ", batch_size=" + str(batch_size) + ")"
torch.save({
    'model_state_dict' : net.state_dict(),
    'optimizer_state_dict' : optimizer.state_dict(),
    'loss_vec' : loss_vec,
    'smooth_loss_vec' : smooth_loss_vec,
    'val_loss_vec' : val_loss_vec,
    'validation_factor' : validation_factor,
    'n_hidden' : n_hidden,
    'n_layers' : n_layers,
    'embedding_dim' : embedding_dim,
    'batch_size' : batch_size,
    'K' : data.K,
    'seq_length' : seq_length,
    'n_epochs' : n_epochs,
    'learning_rate' : learning_rate,
    'input_file_name' : input_file_name,
    'config_text' : config_text,
    'module_id' : module_id,
    'seed' : seed,
},  save_folder + "/" + save_file_name)

'''
    Synthesize some text
'''
if syn_beam_search:
    text_inds = lstm_word_train.synthesize_characters_beam(data, net, syn_length, device, beam_search_width, beam_search_sampler)
else:
    text_inds = lstm_word_train.synthesize_characters(data, net, syn_length, device)
print()
print("Synthesized text:")
print("\t" + data.indsToString(text_inds))
print()
