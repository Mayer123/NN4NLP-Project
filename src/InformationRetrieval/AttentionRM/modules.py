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
	def __init__(self, emb_layer, pos_emb_layer, emb_trainable=True, vocab_size=None, 
					pos_vocab_size=None, emb_dim=100, pos_emb_dim=20, out_channels=50, dropout=0.3):
		super(AttentionRM, self).__init__()
		# if init_emb is not None:
		# 	self.emb = nn.Embedding.from_pretrained(init_emb, 
		# 								freeze=(not emb_trainable))
		# else:
		# 	self.emb = nn.Embedding(vocab_size, emb_dim)
		# self.pos_emb = nn.Embedding(pos_vocab_size, pos_emb_dim)
		self.emb = emb_layer
		self.pos_emb = pos_emb_layer
		self.emb_proj = nn.Linear(emb_dim+pos_emb_dim, emb_dim)

		self.evidence_collector = nn.Sequential(
			nn.Conv1d(emb_dim, out_channels, 5, padding=2),
			nn.Tanh(),
			nn.Dropout(dropout, inplace=True),
			nn.Conv1d(out_channels, out_channels, 3, padding=1),
			nn.Dropout(dropout, inplace=True),
			)	

		self.mlp1 = nn.Sequential(
                        nn.Linear(2*emb_dim, emb_dim, bias=False),
                        nn.Tanh(),
                        nn.Dropout(dropout, inplace=True),
                        nn.Linear(emb_dim, 1, bias=False)
                    )

		self.interactive_aligner = InteractiveAligner(emb_dim)
		self.self_aligner = SelfAligner(emb_dim)
		self.summarizer = Summarizer(emb_dim)
		self.reduction = nn.Linear(emb_dim, out_channels)
		#self.emb_dropout = nn.Dropout(dropout, inplace=True)
		
	def forward(self, q, d, qlen, dlen):
		#print('d size:',d.nelement() * d.storage().element_size()/(1024**3))
		#print('total_mem_used', torch.cuda.memory_allocated(0) / (1024)**3)
		qw_emb = self.emb(q[:,:,0])
		qp_emb = self.pos_emb(q[:,:,1])
		q_emb = self.emb_proj(torch.cat((qw_emb, qp_emb), dim=2))
		del qw_emb, qp_emb

		dw_emb = self.emb(d[:,:,0])
		#print('dw_emb size:',dw_emb.storage().size() * dw_emb.storage().element_size()/(1024**3))
		#print('total_mem_used', torch.cuda.memory_allocated(0) / (1024)**3)
		dp_emb = self.pos_emb(d[:,:,1])
		#print('dp_emb size:',dp_emb.storage().size() * dp_emb.storage().element_size()/(1024**3))
		#print('total_mem_used', torch.cuda.memory_allocated(0) / (1024)**3)			
		
		d_emb = self.emb_proj(torch.cat((dw_emb, dp_emb), dim=2))		
		del dw_emb, dp_emb		

		#print('d_emb size:',d_emb.storage().size() * d_emb.storage().element_size()/(1024**3))
		#print('total_mem_used', torch.cuda.memory_allocated(0) / (1024)**3)

		#d_emb = self.emb_dropout(d_emb)
		#print('d_emb size:',d_emb.storage().size() * d_emb.storage().element_size()/(1024**3))
		#print('total_mem_used', torch.cuda.memory_allocated(0) / (1024)**3)

		qng, qng_len, _ = self.interactive_aligner(d_emb, q_emb,
												dlen, qlen)
		del d_emb
		#print('qng size:',qng.storage().size() * qng.storage().element_size())
		#print('total_mem_used', torch.cuda.memory_allocated(0) / (1024)**3)

		attendedD, attendedD_len, _ = self.self_aligner(qng, qng_len)
		#print('attendedD size:',attendedD.storage().size() * attendedD.storage().element_size()/(1024**3))
		#print('total_mem_used', torch.cuda.memory_allocated(0) / (1024)**3)
		del qng
		# with torch.no_grad():
		# 	for i in range(len(attendedD_len)):
		# 		attendedD[i, attendedD_len[i]:,:] = 0

		encoded = self.evidence_collector(attendedD.transpose(1,2))
		#print('encoded size:',encoded.storage().size() * encoded.storage().element_size()/(1024**3))
		#print('total_mem_used', torch.cuda.memory_allocated(0) / (1024)**3)
		# encoded = torch.mean(encoded, dim=2)
		encoded = encoded.transpose(1,2)		
		s = self.summarizer(q_emb, qlen)
		s = self.reduction(s)
		s = s.expand(s.shape[0], encoded.shape[1], s.shape[2])	

		catted = torch.cat((encoded, s, encoded*s, encoded-s), dim=2)
		del encoded
		# with torch.no_grad():
		# 	for i in range(len(attendedD_len)):
		# 		catted[i, attendedD_len[i]:,:] = 0

		score = self.mlp1(catted)					
		score = torch.mean(score.view(score.shape[0], -1), dim=1)
		return score