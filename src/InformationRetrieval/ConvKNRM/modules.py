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
					emb_dim=100, nfilters=32, max_ngram=3, xmatch_ngrams=False, 
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

		self.linear = nn.Linear(nkernels * max_ngram**(1 + int(xmatch_ngrams)), 1, bias=False)
		self.dropout = nn.Dropout(dropout)
	def embed(self, x):
		x_emb = self.dropout(self.emb(x[:,:,0]))
		x_emb = x_emb / torch.norm(x_emb, dim=2, keepdim=True)
		return x_emb

	def score(self, q_emb, d_emb, qlen, dlen):
		q_conv = [conv(q_emb.transpose(1,2)) for conv in self.convs]
		d_conv = [conv(d_emb.transpose(1,2)) for conv in self.convs]

		counts = []
		for qi in range(len(q_conv)):
			for di in range(len(d_conv)):
				if not self.xmatch_ngrams and qi != di:
					continue

				qng = q_conv[qi]
				dng = d_conv[di]

				sim = torch.bmm(qng.transpose(1,2), dng)
				sim = self.dropout(sim)
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
		score = self.linear(counts).squeeze(1)
		return score

	def forward(self, q, d, qlen, dlen):
		q_emb = self.embed(q)
		d_emb = self.embed(d)
		score = self.score(q_emb, d_emb, qlen, dlen)
		return score

	def getSentScores(self, q, c, qlen, clen, batch_size):		
		scores = torch.zeros(c.shape[0]).float().to(c.device)
		n_batches = (c.shape[0] + batch_size - 1) // batch_size
		for i in range(n_batches):
			c_batch = c[i*batch_size:(i+1)*batch_size]			
			clen_batch = clen[i*batch_size:(i+1)*batch_size]

			q_batch = q[i*batch_size:(i+1)*batch_size]
			qlen_batch = qlen[i*batch_size:(i+1)*batch_size]

			# print(c_batch.shape, clen_batch.shape)
			# print(q_batch.shape, qlen_batch.shape)

			scores[i*batch_size:(i+1)*batch_size] = self.score(q_batch, c_batch, 
															qlen_batch, clen_batch)
		return scores

	def forward_singleContext(self, q, d, qlen, dlen, batch_size=1024):
		q = self.embed(q)
		d = self.embed(d)

		scores = []
		for i in range(q.shape[0]):
			q_ = q[i, :qlen[i]]
			q_ = q_.expand(d.shape[0], -1, -1)

			qlen_ = qlen[i:i+1].expand(q_.shape[0])	

			# print(torch.cuda.memory_allocated(0) / (1024)**3)
			# c_scores = self.score(q_, d, qlen_, dlen)
			c_scores = self.getSentScores(q_, d, qlen_, dlen, batch_size)
			# print(c_scores.shape)

			scores.append(c_scores)

		scores = torch.stack(scores, dim=0)
		return scores

if __name__ == '__main__':
	q = torch.LongTensor(torch.randint(10, size=(2,10,2)).long())
	d = torch.LongTensor(torch.randint(3, size=(3,4,2)).long())
	qlen = torch.LongTensor([1,2])
	dlen =  torch.LongTensor([2,3])

	m = ConvKNRM(vocab_size=20)
	print(m.forward_singleContext(q, d, qlen, dlen))