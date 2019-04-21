import sys
sys.path.append('../')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class End2EndModel(nn.Module):
	"""docstring for End2EndModel"""
	def __init__(self, ir_model, rc_model):
		super(End2EndModel, self).__init__()
		self.ir_model = ir_model
		self.rc_model = rc_model

	def forward(self, q, c, qlen, clen):
		c_scores = self.ir_model(q, c, qlen, clen)
		