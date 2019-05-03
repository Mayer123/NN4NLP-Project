import sys
sys.path.append('../')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from utils import utils

class NoveltyNTN(nn.Module):
	"""docstring for NoveltyNTN"""
	def __init__(self, input_dim, out_dim):
		super(NoveltyNTN, self).__init__()
		self.bilinear = nn.Bilinear(input_dim, input_dim, out_dim, bias=False)
		self.mu = nn.Linear(out_dim, 1, bias=False)

	def forward(self, d, S):
		# d.shape = [batch_size, 1, input_dim]
		# S.shape = [batch_size, K, input_dim]
		d = d.expand(-1, S.shape[0], -1)
		scores = self.bilinear(d, S)
		scores = torch.tanh(scores)
		scores = self.mu(scores)
		return scores
