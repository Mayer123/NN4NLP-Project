import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class GaussianKernel(object):
	"""docstring for GaussianKernel"""
	def __init__(self, mean, std):
		super(GaussianKernel, self).__init__()
		self.mean = mean
		self.std = std

	def __call__(self, x):
		sim = -0.5 * (x-self.mean)**2 / self.std**2
		counts = torch.sum(sim.reshape(-1, sim.shape[1] * sim.shape[2]), dim=1)		
		return counts
		

class ConvKNRM(nn.Module):
	"""docstring for ConvKNRM"""
	def __init__(self, init_emb=None, emb_trainable=True, vocab_size=None, 
					emb_dim=100, nfilters=128, max_ngram=3, xmatch_ngrams=True, 
					nkernels=11, sigma=0.1, exact_sigma=0.001):
		super(ConvKNRM, self).__init__()
		if init_emb is not None:
			self.emb = nn.Embedding.from_pretrained(init_emb, 
										freeze=(not emb_trainable))
		else:
			self.emb = nn.Embedding(vocab_size, emb_dim)

		self.convs = nn.ModuleList([])
		for i in range(1,max_ngram+1):	
			c = nn.Sequential(
				nn.ConstantPad1d((0,i-1), 0.0),
				nn.Conv1d(emb_dim, nfilters, i),
				nn.ReLU()
				)					
			self.convs.append(c)

		self.kernels = []
		for i in range(nkernels):
			mu = 1/(nkernels-1) + 2*i/(nkernels-1) - 1

			if mu > 1:
				self.kernels.append(GaussianKernel(1., exact_sigma))
			else:
				self.kernels.append(GaussianKernel(mu, sigma))

		self.xmatch_ngrams = xmatch_ngrams

		self.linear = nn.Linear(nkernels * max_ngram**2, 1)
	def forward(self, q, d):
		q_emb = self.emb(q).transpose(1,2)
		d_emb = self.emb(d).transpose(1,2)		
		print(q_emb.shape, d_emb.shape)
		q_conv = [conv(q_emb) for conv in self.convs]
		d_conv = [conv(d_emb) for conv in self.convs]

		print(q_conv[1].shape, d_conv[1].shape)

		counts = []
		for qi in range(len(q_conv)):
			for di in range(len(d_conv)):
				if not self.xmatch_ngrams and qi != di:
					continue

				qng = q_conv[qi]
				dng = d_conv[di]

				sim = torch.bmm(qng.transpose(1,2), dng)
				kernel_counts = torch.stack([K(sim) for K in self.kernels], dim=1)
				print(kernel_counts.shape)
				counts.append(kernel_counts)
		counts = torch.cat(counts, dim=1)
		print(counts.shape)
		score = torch.tanh(self.linear(counts))
		return score