'''
	Module for data loading, pre-processing, format conversion etc.
'''

import torch
import math
from transformers import BertModel, BertTokenizer

class CharacterData():
	def __init__(self, file_path, device):
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

class VecData():
	def __init__(self, file_path, device):
		# Raw text data
		self.text_data = [x.strip() for x in open(file_path, encoding="utf-8").read().split('.')]
		# BERT model
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
		# ids (BERT input)
		self.ids_list = self.text2ids(self.text_data)
		# Word embeddings
		self.vec_data = self.ids2vecs(self.ids_list)
		# K: Number of classes (dimensions of word embedding)
		self.K = self.vec_data.shape[2]

		self.device = device

	def text2ids(self, text):
		ids = [torch.tensor([self.tokenizer.encode(t, add_special_tokens=True)]) for t in text]
		return ids

	def ids2vecs(self, ids_list):
		# n_ids = ids.shape[1]
		# size_batch = 512
		# n_batch = math.ceil(n_ids/size_batch)
		vecs = torch.empty(0, 1, 768)
		for ids in ids_list:
			with torch.no_grad():
				hidden_states = self.model(ids)[2]
			vecs_batch = torch.stack(hidden_states[:4]).sum(0).transpose(0, 1)
			vecs = torch.cat([vecs, vecs_batch], dim=0)
		return vecs

	def vecs2text(self, vecs):
		# Convert n*K matrix of word embeddings to text 

		# n_vec_data = self.vec_data.shape[0]
		# trimmed_vec_data = self.vec_data.narrow(0, 1, n_vec_data - 2)
		dotted = torch.mm(self.vec_data.squeeze(1), vecs.transpose(0, 1))
		argmax = dotted.max(0)[1]
		ids = torch.empty(1, 0, dtype=torch.long)
		# print(self.ids[0].shape)
		# exit()
		for one_id in self.ids_list:
			ids = torch.cat([ids, one_id], dim=1)
		ids = ids.squeeze(0)[argmax]
		text = self.tokenizer.decode(ids)
		return text

