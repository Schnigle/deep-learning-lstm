'''
	Module for miscellaneous utility functions.
'''

import random
import torch

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

# Synthesize a text sequence of length n
def synthesize_characters(data, net, n, device):
	hidden = net.initHidden().to(device)
	net.zero_grad()
	prev_char = data.toOneHot(data.char_to_ind['.'])
	word = []
	for i in range(n):
		'''
			Select the next character based on the probability distribution.
			Note: we don't always select the most likely character, as that would
			likely result in repeating phrases such as "the the the the the..."
		'''
		output, hidden = net(prev_char, hidden)
		# Convert output to probability weights. 
		# exp is needed to invert log softmax.
		output = torch.exp(output)
		char_index = randomSampleFromWeights(output[0])
		word.append(char_index)
		prev_char = data.toOneHot(char_index)
	return word
