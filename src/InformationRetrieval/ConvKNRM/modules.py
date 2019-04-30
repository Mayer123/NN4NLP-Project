import sys
sys.path.append('/home/mshah1/narrativeQA/NN4NLP-Project/src')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ReadingComprehension.IterativeReattentionAligner.modules import (InteractiveAligner, 
													SelfAligner, 
													Summarizer, 
													AnswerPointer)
from utils.utils import *

class GaussianKernel(object):
	"""docstring for GaussianKernel"""
	def __init__(self, mean, std):
		super(GaussianKernel, self).__init__()
		self.mean = mean
		self.std = std

	def __call__(self, x):
		sim = torch.exp(-0.5 * (x-self.mean)**2 / self.std**2)		
		return sim
		

class ConvKNRM(nn.Module):
	"""docstring for ConvKNRM"""
	def __init__(self, init_emb=None, emb_trainable=True, vocab_size=None, 
					emb_dim=100, nfilters=64, max_ngram=3, xmatch_ngrams=False, 
					nkernels=11, sigma=0.1, exact_sigma=0.001, dropout=0.3):
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
				nn.Tanh()
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

		self.evidence_collector = nn.Sequential(
			nn.Conv1d(emb_dim, emb_dim, 5, padding=2),			
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Conv1d(emb_dim, emb_dim, 3, padding=1)
			)	

		self.interactive_aligner = InteractiveAligner(emb_dim)
		self.self_aligner = SelfAligner(emb_dim)
		self.summarizer = Summarizer(emb_dim)

		self.linear = nn.Linear(nkernels * max_ngram**(1 + int(xmatch_ngrams)), 1, bias=False)
	def forward(self, q, d, qlen, dlen):
		q_emb = self.emb(q[:,:,0])
		d_emb = self.emb(d[:,:,0])

		q_conv = [conv(q_emb.transpose(1,2)) for conv in self.convs]
		d_conv = [conv(d_emb) for conv in self.convs]

		counts = []
		for qi in range(len(q_conv)):
			for di in range(len(d_conv)):
				if not self.xmatch_ngrams and qi != di:
					continue

				qng = q_conv[qi]
				dng = d_conv[di]

				sim = torch.bmm(qng.transpose(1,2), dng)
				# print(qng.shape, dng.shape, sim.shape)
				kernel_counts = []
				for K in self.kernels:
					probs = K(sim)					
					qt_match_count = torch.sum(probs, dim=2)					
					total_count = torch.sum(torch.log1p(qt_match_count), dim=1)
					kernel_counts.append(total_count)
				kernel_counts = torch.stack(kernel_counts, dim=1)
				# kernel_counts = torch.stack([K(sim) for K in self.kernels], dim=1)
				# print(kernel_counts.shape)
				counts.append(kernel_counts)
		counts = torch.cat(counts, dim=1)
		# print(counts.shape)		
		score = self.linear(counts)
		return score