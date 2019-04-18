import sys
sys.path.append('/home/mshah1/narrativeQA/NN4NLP-Project/src')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from utils.utils import *
from IterativeReattentionAligner.modules import (InteractiveAligner, 
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
					pos_vocab_size=None, emb_dim=100, dropout=0.3):
		super(AttentionRM, self).__init__()
		if init_emb is not None:
			self.emb = nn.Embedding.from_pretrained(init_emb, 
										freeze=(not emb_trainable))
		else:
			self.emb = nn.Embedding(vocab_size, emb_dim)
		self.pos_emb = nn.Embedding(pos_vocab_size, emb_dim)
		self.emb_proj = nn.Linear(2*emb_dim, emb_dim)

		self.evidence_collector = nn.Sequential(
			nn.Conv1d(emb_dim, emb_dim, 5, padding=2),
			nn.Tanh(),
			nn.Dropout(dropout),
			nn.Conv1d(emb_dim, emb_dim, 3, padding=1),
			nn.Dropout(dropout),
			)	

		self.mlp1 = nn.Sequential(
                        nn.Linear(4*emb_dim, emb_dim, bias=False),
                        nn.Tanh(),
                        nn.Dropout(dropout),
                        nn.Linear(emb_dim, 1, bias=False)
                    )

		self.interactive_aligner = InteractiveAligner(emb_dim)
		self.self_aligner = SelfAligner(emb_dim)
		self.summarizer = Summarizer(emb_dim)
		self.emb_dropout = nn.Dropout(dropout)
	def forward(self, q, d, qlen, dlen):
		q_emb = self.emb_dropout(self.emb(q))
		dw_emb = self.emb_dropout(self.emb(d[:,:,0]))
		dp_emb = self.emb_dropout(self.pos_emb(d[:,:,1]))
		# print(d.shape, dw_emb.shape, dp_emb.shape)
		d_emb = self.emb_proj(torch.cat((dw_emb, dp_emb), dim=2))
		d_emb = self.emb_dropout(d_emb)

		qng, qng_len, _ = self.interactive_aligner(d_emb, q_emb,
												dlen, qlen)

		attendedD, attendedD_len, B = self.self_aligner(qng, qng_len)

		# with torch.no_grad():
		# 	for i in range(len(attendedD_len)):
		# 		attendedD[i, attendedD_len[i]:,:] = 0

		encoded = self.evidence_collector(attendedD.transpose(1,2))

		# encoded = torch.mean(encoded, dim=2)
		encoded = encoded.transpose(1,2)		
		s = self.summarizer(q_emb, qlen)
		s = s.expand(s.shape[0], encoded.shape[1], s.shape[2])	

		catted = torch.cat((encoded, s, encoded*s, encoded-s), dim=2)
		# with torch.no_grad():
		# 	for i in range(len(attendedD_len)):
		# 		catted[i, attendedD_len[i]:,:] = 0

		score = self.mlp1(catted)					
		score = torch.mean(score.view(score.shape[0], -1), dim=1)
		return score