import torch
import torch.nn as nn
import torch.nn.functional as F

import random

class CNNet(nn.Module):
	'''
	Re-implementation of the network from
	"Learning to Find Good Correspondences"
	Yi, Trulls, Ono, Lepetit, Salzmann, Fua
	CVPR 2018
	'''

	def __init__(self, blocks):
		'''
		Constructor.
		'''
		super(CNNet, self).__init__()

		# network takes 5 inputs per correspondence: 2D point in img1, 2D point in img2, and 1D side information like a matching ratio
		self.p_in = nn.Conv2d(5, 128, 1, 1, 0)

		# list of residual blocks
		self.res_blocks = []

		for i in range(0, blocks):
			self.res_blocks.append((
				nn.Conv2d(128, 128, 1, 1, 0),
				nn.BatchNorm2d(128),	
				nn.Conv2d(128, 128, 1, 1, 0),
				nn.BatchNorm2d(128),	
				))

		# register list of residual block with the module
		for i, r in enumerate(self.res_blocks):
			super(CNNet, self).add_module(str(i) + 's0', r[0])
			super(CNNet, self).add_module(str(i) + 's1', r[1])
			super(CNNet, self).add_module(str(i) + 's2', r[2])
			super(CNNet, self).add_module(str(i) + 's3', r[3])

		# output are 1D sampling weights (log probabilities)
		self.p_out =  nn.Conv2d(128, 1, 1, 1, 0)

	def forward(self, inputs):
		'''
		Forward pass, return log probabilities over correspondences.

		inputs -- 4D data tensor (BxCxNx1)
		B -> batch size (multiple image pairs)
		C -> 5 values (2D coordinate + 2D coordinate + 1D side information)
		N -> number of correspondences
		1 -> dummy dimension
		
		'''
		batch_size = inputs.size(0)
		data_size = inputs.size(2) # number of correspondences

		x = inputs
		x = F.relu(self.p_in(x))
		
		for r in self.res_blocks:
			res = x
			x = F.relu(r[1](F.instance_norm(r[0](x)))) 
			x = F.relu(r[3](F.instance_norm(r[2](x))))
			x = x + res

		log_probs = F.logsigmoid(self.p_out(x))

		# normalization in log space such that probabilities sum to 1
		log_probs = log_probs.view(batch_size, -1)
		normalizer = torch.logsumexp(log_probs, dim=1)
		normalizer = normalizer.unsqueeze(1).expand(-1, data_size)
		log_probs = log_probs - normalizer
		log_probs = log_probs.view(batch_size, 1, data_size, 1)

		return log_probs
