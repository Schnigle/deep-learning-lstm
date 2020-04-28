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
