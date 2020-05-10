'''
	Module for data loading, pre-processing, format conversion etc.
'''

import torch
import re

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

class WordData():
	def __init__(self, file_path, device):
		# Raw text data
		self.device = device
		self.text_data = open(file_path, encoding="utf-8").read().strip()
		# self.text_data = "Hello   \tthere\n  General kebono"
		# Keep some special characters as "words"
		self.special_chars = [".", ",", "?", "!", ":", ";", "\"", "(", ")", "\n", "[WS]", "[EQ]"]
		# self.capitalized_words = ["i", "harry", "ron", "hermione", "weasley", "malfoy", "dumbledore", "dudley", "sirius", "mcgonagall"]
		self.eos_chars = ".!?"
		# self.text_data = self.text_data.replace(" ", "[WS]")
		self.text_data = self.text_data.replace("\" ", " [EQ] ").replace("\"\n", " [EQ] \n ").replace(" \"", " [SQ] ").replace("\n\"", " \n [SQ] ")
		for char in self.special_chars:
			self.text_data = self.text_data.replace(char, " " + char + " ")
		# self.text_data = self.text_data.replace(".", " . ").replace("?", " ? ").replace("!", " ! ").replace(",", " , ").replace("\"", " \" ").replace("(", " ( ").replace(")", " ) ")
		# self.text_data = re.sub("(\t| )+", "[wspace]", self.text_data)
		self.text_data = re.sub("(\t| )+", " ", self.text_data)
		self.word_data = self.text_data.split(" ")
		# print(self.text_data)
		self.words = sorted(list(set(self.word_data)))
		self.K = len(self.words)
		self.word_to_ind = dict()
		self.ind_to_word = dict()
		for i in range(self.K):
			self.word_to_ind[self.words[i]] = i
			self.ind_to_word[i] = self.words[i]
		# List of unique characters
		# self.text_chars = sorted(list(set(self.text_data)))
		# # K: Number of classes (unique characters)
		# self.K = len(self.text_chars)

		# # Create structures mapping class indexes to chars
		# self.char_to_ind = dict()
		# self.ind_to_char = dict()
		# for i in range(self.K):
		# 	self.char_to_ind[self.text_chars[i]] = i
		# 	self.ind_to_char[i] = self.text_chars[i]

	def isWord(self, str):
		return not self.special_chars.__contains__(str)

	def capitalize(self, str):
		return str
		# return str[0].upper() + str[1:len(str)]

	def indsToString(self, inds):
		str = ""
		capitalize = True
		previous_token = "."
		for i in inds:
			token = self.ind_to_word[i]
			if token == "[EQ]":
				str += "\" "
			elif token == "\"":
				str += " \""
			# if token == "[WS]":
			# 	str += " "
			elif self.isWord(token):
				if self.isWord(previous_token) or self.eos_chars.__contains__(previous_token) or previous_token == "," or previous_token == ";" or previous_token == ":":
					str += " "
				if capitalize or self.capitalized_words.__contains__(token):
					str += self.capitalize(token)
				else:
					str += token
				capitalize = False
			else:
				str += token
				if self.eos_chars.__contains__(token):
					capitalize = True
				# if self.eos_chars.__contains__(token) or token == "," or token == ";" or (token == "\"" and previous_token == ","):
				# 	str += " "
				# if not self.eos_chars.__contains__(previous_token):
				# 	str += " "
			previous_token = token
		return str
