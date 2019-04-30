import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ReadingComprehension.IterativeReattentionAligner.modules import IterativeAligner

class GaussianKernel(object):
	"""docstring for GaussianKernel"""
	def __init__(self, mean, std):
		super(GaussianKernel, self).__init__()
		self.mean = mean
		self.std = std

	def __call__(self, x):
		sim = torch.exp(-0.5 * (x-self.mean)**2 / self.std**2)		
		return sim
		

class RMRRM(nn.Module):
	"""docstring for ConvKNRM"""
	def __init__(self, emb_layer=None, pos_emb_layer=None, init_emb=None, emb_trainable=True, vocab_size=None, 
					pos_vocab_size=None, emb_dim=100, pos_emb_dim=100, dropout=0.3,
					use_rnn=True):
		super(RMRRM, self).__init__()
		if emb_layer is not None:
			self.emb = emb_layer
			emb_dim = self.emb.weight.data.shape[1]

		elif init_emb is not None:
			self.emb = nn.Embedding.from_pretrained(init_emb, 
										freeze=(not emb_trainable))
		else:
			self.emb = nn.Embedding(vocab_size, emb_dim)

		if pos_emb_layer is not None:
			self.pos_emb = pos_emb_layer
			pos_emb_dim = self.pos_emb.weight.data.shape[1]
		else:
			self.pos_emb = nn.Embedding(pos_vocab_size, pos_emb_dim)

		
		self.dropout = nn.Dropout(dropout, inplace=False)

	def embed(self, x):
		xw_emb = self.dropout(self.emb(x[:,:,0]))
		xp_emb = self.dropout(self.pos_emb(x[:,:,1]))
		x_emb = torch.cat((xw_emb, xp_emb), dim=2)

		# print('x size:',x.nelement() * x.storage().element_size()/(1024**3))
		# print('total_mem_used', torch.cuda.memory_allocated(0) / (1024)**3)

		return x_emb

	def score(self, q_emb, d_emb, qlen, dlen):

		qng, qng_len, _ = self.interactive_aligner(d_emb, q_emb,
												dlen, qlen)		
		del d_emb

		attendedD, attendedD_len, _ = self.self_aligner(qng, qng_len)
		attendedD = self.dropout(attendedD)
		del qng

		if self.use_rnn:
			# sorted_len, sorted_idx = torch.sort(attendedD_len, descending=True)
			# _, rev_sorted_idx = torch.sort(sorted_idx)

			# attendedD = torch.nn.utils.rnn.pack_padded_sequence(attendedD[sorted_idx], sorted_len, batch_first=True)
			encoded,_ = self.evidence_collector(attendedD)
			# encoded,_ = torch.nn.utils.rnn.pad_packed_sequence(attendedD, batch_first=True)
			# encoded = encoded[rev_sorted_idx]

			# encoded = self.evidence_proj(encoded)
		else:
			encoded = self.evidence_collector(attendedD.transpose(1,2))		
			encoded = encoded.transpose(1,2)

		encoded = self.dropout(encoded)

		s = self.answer_pointer.summarizer(q_emb, qlen)
		s = s.expand(s.shape[0], encoded.shape[1], s.shape[2])	

		probs = self.answer_pointer.computeP(s, encoded, attendedD_len, self.answer_pointer.mlp1)
		
		# catted = torch.cat((encoded, s, encoded*s, encoded-s), dim=2)
		# del encoded

		# probs = self.mlp1(catted)
		score = torch.mean(probs.view(probs.shape[0], -1), dim=1)
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