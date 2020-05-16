'''
	Module for data loading, pre-processing, format conversion etc.
'''

import torch
import re
import math
from transformers import BertModel, BertTokenizer
import utility
import torch.nn.functional as F
import os

class CharacterData():
	def __init__(self, file_path, device, validation_factor):
		# Raw text data
		self.text_data = open(file_path, encoding="utf-8").read().strip()
		# List of unique characters
		self.text_chars = sorted(list(set(self.text_data)))
		# K: Number of classes (unique characters)
		self.K = len(self.text_chars)

		# Create structures mapping class indexes to chars
		self.char_to_ind = dict()
		self.ind_to_char = dict()
		for i in range(self.K):
			self.char_to_ind[self.text_chars[i]] = i
			self.ind_to_char[i] = self.text_chars[i]
		self.device = device
		self.n_samples = len(self.text_data)
		train_samples = round(len(self.text_data) * (1 - validation_factor))
		self.train_data = self.text_data[0:train_samples]
		self.val_data = self.text_data[train_samples + 1:]

	# Convert a string to a one-hot [len(str), 1, K] tensor representation of all character classes in the string.
	def stringToTensor(self, str):
		tensor = torch.zeros(len(str), 1, self.K, device=self.device)
		for i in range(len(str)):
			tensor[i, 0, self.char_to_ind[str[i]]] = 1
		return tensor

	# Convert a string to a non-one-hot [len(str)] tensor representation of all character classes in the string. Used for labels in NLLLoss, which are not one-hot.
	def stringToTensorNLLLabel(self, str):
		tensor = torch.zeros(len(str), device=self.device, dtype=torch.long)
		for i in range(len(str)):
			tensor[i] = self.char_to_ind[str[i]]
		return tensor

	# Convert an integer to a [1, K] one-hot tensor representation
	def toOneHot(self, val):
		tensor = torch.zeros(1, self.K, device=self.device)
		tensor[0, val] = 1
		return tensor

	# Convert an array of character class integers (non-one-hot) to the corresponding string representation
	def indsToString(self, inds):
		str = ""
		for i in inds:
			str += self.ind_to_char[i]
		return str

class WordData():
	def __init__(self, file_path, device, validation_factor):
		# Raw text data
		self.device = device
		self.text_data = open(file_path, encoding="utf-8").read().strip()
		# Keep some special characters as "words"
		self.special_chars = [".", ",", "?", "!", ":", ";", "\"", "(", ")", "\n", "[WS]", "[EQ]", "[SQ]"]
		self.eos_chars = ".!?"
		# Use special characters for start and end qoutes
		self.text_data = self.text_data.replace("\" ", " [EQ] ").replace("\"\n", " [EQ] \n ").replace(" \"", " [SQ] ").replace("\n\"", " \n [SQ] ").replace("\"", " ")
		for char in self.special_chars:
			self.text_data = self.text_data.replace(char, " " + char + " ")
		# Split on space and newline
		self.text_data = re.sub("(\t| )+", " ", self.text_data)
		self.word_data = self.text_data.split(" ")
		# Partition into training and validation data
		train_samples = round(len(self.word_data) * (1 - validation_factor))
		self.train_data = self.word_data[0:train_samples]
		self.val_data = self.word_data[train_samples + 1:]
		self.words = sorted(list(set(self.word_data)))
		self.K = len(self.words)
		self.word_to_ind = dict()
		self.ind_to_word = dict()
		for i in range(self.K):
			self.word_to_ind[self.words[i]] = i
			self.ind_to_word[i] = self.words[i]
		self.n_samples = len(self.word_data)

	def isWord(self, str):
		return not self.special_chars.__contains__(str)

	def indsToString(self, inds):
		str = ""
		previous_token = "."
		for i in inds:
			token = self.ind_to_word[i]
			if token == "[EQ]":
				str += "\" "
			elif token == "[SQ]":
				str += " \""
			elif self.isWord(token):
				if self.isWord(previous_token) or self.eos_chars.__contains__(previous_token) or previous_token == "," or previous_token == ";" or previous_token == ":":
					str += " "
				str += token
			else:
				if token == "(":
					str += " ("
				elif token == ")":
					str += ") "
				else:
					str += token
			previous_token = token
		return str
		
class VecData():
	def __init__(self, file_path, device):
		# Raw text data
		self.text_data = [x.strip() for x in open(file_path, encoding="utf-8").read().split('.')]
		# BERT model
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
		# ids (BERT input)
		self.ids_list = self.text2ids_list(self.text_data)
		self.ids = self.ids_list2ids(self.ids_list)
		self.tokens = self.ids2tokens(self.ids)
		# Word embeddings
		self.vec_data = self.ids_list2vecs(self.ids_list, file_path)
		# K: Number of classes (dimensions of word embedding)
		self.K = self.vec_data.shape[2]

		self.device = device

	def text2ids_list(self, text):
		ids = [torch.tensor([self.tokenizer.encode(t, add_special_tokens=True)]) for t in text]
		return ids

	def ids_list2ids(self, ids_list):
		ids = torch.empty(1, 0, dtype=torch.long)
		for one_id in ids_list:
			length = one_id.size(1)
			one_id = one_id.narrow(1, 1, length - 2)
			ids = torch.cat([ids, one_id], dim=1)
		return ids

	def ids2tokens(self, ids):
		tokens = self.tokenizer.convert_ids_to_tokens(ids[0, :])
		return tokens

	def ids_list2vecs(self, ids_list, file_path):
		name = file_path[:-4] + "_bert_embedding.pt"
		if os.path.isfile(name):
			vecs = torch.load(name)
		else:
			vecs = torch.empty(0, 1, 768)
			n = len(ids_list) 
			for i, ids in enumerate(ids_list):
				print("\t {} % done.".format(round(i/n*100)), end="\r")
				with torch.no_grad():
					output = self.model(ids)
					hidden_states = output[2]
				vecs_batch = F.normalize(torch.stack(hidden_states[:4]).sum(0), dim=2).transpose(0, 1)
				length = vecs_batch.size(0)
				vecs_batch = vecs_batch.narrow(0, 1, length - 2)
				vecs = torch.cat([vecs, vecs_batch], dim=0)
			torch.save(vecs, name)
		return vecs
	
	def veclike2vec(self, vec):
		similarity = F.cosine_similarity(self.vec_data.squeeze(1), vec)
		sm = F.softmax(similarity, dim=0)
		idx = utility.randomSampleFromWeights(sm)
		vec = self.vec_data[idx, :, :]
		id = self.ids[0, idx]
		return vec, id
