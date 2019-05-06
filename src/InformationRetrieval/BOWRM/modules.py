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

class BOWRM(nn.Module):
	"""docstring for ConvKNRM"""
	def __init__(self, init_emb=None, emb_trainable=True, vocab_size=None, 
					emb_dim=100, dropout=0.3, chunk_size=-1):
		super(BOWRM, self).__init__()
		if init_emb is not None:
			self.emb = nn.Embedding.from_pretrained(init_emb, 
										freeze=(not emb_trainable))
		else:
			self.emb = nn.Embedding(vocab_size, emb_dim)
		
		self.proj = nn.Linear(2*emb_dim, 2*emb_dim, bias=False)
		self.linear = nn.Linear(2*emb_dim, 1, bias=False)
		self.dropout = nn.Dropout(dropout)
		self.chunk_size = chunk_size

	def embed(self, x):
		x_emb = self.emb(x[:,:,0])
		return x_emb

	def score(self, q_emb, d_emb, qlen, dlen):		
		q_sum = torch.mean(q_emb, dim=1)
		d_sum = torch.mean(d_emb, dim=1)		
		bow = torch.cat((q_sum, d_sum), dim=1)
		proj = F.relu(self.proj(bow))
		scores = self.linear(proj).squeeze(1)
		return scores

	def forward(self, q, d, qlen, dlen):
		q_emb = self.embed(q)
		d_emb = self.embed(d)

		if self.chunk_size > 0:
			batch_size = q_emb.shape[0]

			clip_idx = q_emb.shape[1] % self.chunk_size
			if clip_idx > 0:
				q_emb = q_emb[:, :-clip_idx]
			q_emb = q_emb.view(-1, self.chunk_size, q_emb.shape[2])

			clip_idx = d_emb.shape[1] % self.chunk_size
			if clip_idx > 0:
				d_emb = d_emb[:, :-clip_idx]
			d_emb = d_emb.view(-1, self.chunk_size, d_emb.shape[2])

			score = self.score(q_emb, d_emb, None, None)
			score = score.view(batch_size,-1)
			score = torch.max(score, dim=1)
		else:
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