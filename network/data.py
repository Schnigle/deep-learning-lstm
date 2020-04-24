'''
	Module for data loading, pre-processing, format conversion etc.
'''

import torch

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
