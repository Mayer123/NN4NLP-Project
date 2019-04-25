import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ReadingComprehension.IterativeReattentionAligner.modules import (InteractiveAligner, 
													SelfAligner, 
													Summarizer, 
													AnswerPointer)

class GaussianKernel(object):
	"""docstring for GaussianKernel"""
	def __init__(self, mean, std):
		super(GaussianKernel, self).__init__()
		self.mean = mean
		self.std = std

	def __call__(self, x):
		sim = torch.exp(-0.5 * (x-self.mean)**2 / self.std**2)		
		return sim
		

class AttentionRM(nn.Module):
	"""docstring for ConvKNRM"""
	def __init__(self, init_emb=None, emb_trainable=True, vocab_size=None, 
					pos_vocab_size=None, emb_dim=100, dropout=0.3,
					use_rnn=True):
		super(AttentionRM, self).__init__()
		if init_emb is not None:
			self.emb = nn.Embedding.from_pretrained(init_emb, 
										freeze=(not emb_trainable))
		else:
			self.emb = nn.Embedding(vocab_size, emb_dim)
		self.pos_emb = nn.Embedding(pos_vocab_size, emb_dim)
		self.emb_proj = nn.Linear(2*emb_dim, emb_dim)

		self.use_rnn = use_rnn
		if use_rnn:
			self.evidence_collector = nn.LSTM(emb_dim, emb_dim, 1,
                                        batch_first=True, bidirectional=True)
			self.evidence_proj = nn.Linear(2*emb_dim, emb_dim, bias=False)
		else:
			self.evidence_collector = nn.Sequential(
				nn.Conv1d(emb_dim, emb_dim, 5, padding=2),
				nn.Tanh(),
				nn.Dropout(dropout, inplace=False),
				nn.Conv1d(emb_dim, emb_dim, 3, padding=1),
				nn.Dropout(dropout, inplace=False),
				)		

		self.mlp1 = nn.Sequential(
                        nn.Linear(4*emb_dim, emb_dim, bias=False),
                        nn.Tanh(),
                        nn.Dropout(dropout, inplace=False),
                        nn.Linear(emb_dim, 1, bias=False)
                    )

		self.interactive_aligner = InteractiveAligner(emb_dim)
		self.self_aligner = SelfAligner(emb_dim)
		self.summarizer = Summarizer(emb_dim)
		self.emb_dropout = nn.Dropout(dropout, inplace=False)

	def embed(self, x):
		xw_emb = self.emb_dropout(self.emb(x[:,:,0]))
		xp_emb = self.emb_dropout(self.pos_emb(x[:,:,1]))
		x_emb = self.emb_proj(torch.cat((xw_emb, xp_emb), dim=2))
		# print('x size:',x.nelement() * x.storage().element_size()/(1024**3))
		# print('total_mem_used', torch.cuda.memory_allocated(0) / (1024)**3)

		return x_emb

	def score(self, q_emb, d_emb, qlen, dlen):
		# print('d_emb size:',d_emb.storage().size() * d_emb.storage().element_size()/(1024**3))
		# print('total_mem_used', torch.cuda.memory_allocated(0) / (1024)**3)

		# d_emb = self.emb_dropout(d_emb)
		# print(q_emb.shape, d_emb.shape)
		# print('d_emb size:',d_emb.storage().size() * d_emb.storage().element_size()/(1024**3))
		# print('total_mem_used', torch.cuda.memory_allocated(0) / (1024)**3)

		# print('q_emb size:',q_emb.storage().size() * q_emb.storage().element_size()/(1024**3))
		# print('total_mem_used', torch.cuda.memory_allocated(0) / (1024)**3)

		qng, qng_len, _ = self.interactive_aligner(d_emb, q_emb,
												dlen, qlen)
		del d_emb
		# print('qng size:',qng.storage().size() * qng.storage().element_size())
		# print('total_mem_used', torch.cuda.memory_allocated(0) / (1024)**3)

		attendedD, attendedD_len, _ = self.self_aligner(qng, qng_len)
		# print('attendedD size:',attendedD.storage().size() * attendedD.storage().element_size()/(1024**3))
		# print('total_mem_used', torch.cuda.memory_allocated(0) / (1024)**3)
		del qng
		# print(attendedD.shape)
		if self.use_rnn:
			encoded,_ = self.evidence_collector(attendedD)
			# print(encoded.shape)
			encoded = self.evidence_proj(encoded)
		else:
			encoded = self.evidence_collector(attendedD.transpose(1,2))			
			# print('encoded size:',encoded.storage().size() * encoded.storage().element_size()/(1024**3))
			# print('total_mem_used', torch.cuda.memory_allocated(0) / (1024)**3)
			# print ('encoded.shape:', encoded.shape)						
			encoded = encoded.transpose(1,2)

		s = self.summarizer(q_emb, qlen)
		s = s.expand(s.shape[0], encoded.shape[1], s.shape[2])	

		catted = torch.cat((encoded, s, encoded*s, encoded-s), dim=2)
		del encoded

		probs = self.mlp1(catted)
		score = torch.mean(probs.view(probs.shape[0], -1), dim=1)
		return score		

	def forward(self, q, d, qlen, dlen):
		# print('d size:',d.nelement() * d.storage().element_size()/(1024**3))
		# print('total_mem_used', torch.cuda.memory_allocated(0) / (1024)**3)
		# qw_emb = self.emb_dropout(self.emb(q[:,:,0]))
		# qp_emb = self.emb_dropout(self.pos_emb(q[:,:,1]))
		# q_emb = self.emb_proj(torch.cat((qw_emb, qp_emb), dim=2))
		
		# dw_emb = self.emb_dropout(self.emb(d[:,:,0]))
		# # print('dw_emb size:',dw_emb.storage().size() * dw_emb.storage().element_size()/(1024**3))
		# # print('total_mem_used', torch.cuda.memory_allocated(0) / (1024)**3)
		# dp_emb = self.emb_dropout(self.pos_emb(d[:,:,1]))
		# # print('dp_emb size:',dp_emb.storage().size() * dp_emb.storage().element_size()/(1024**3))
		# # print('total_mem_used', torch.cuda.memory_allocated(0) / (1024)**3)			
		
		# d_emb = self.emb_proj(torch.cat((dw_emb, dp_emb), dim=2))		

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