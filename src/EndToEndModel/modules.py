import sys
sys.path.append('../')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from utils import utils

class EndToEndModel(nn.Module):
	"""docstring for End2EndModel"""
	def __init__(self, ir_model, rc_model, n_ctx_sents=50):
		super(EndToEndModel, self).__init__()
		self.ir_model = ir_model
		self.rc_model = rc_model
		self.n_ctx_sents = n_ctx_sents	

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
			scores[i*batch_size:(i+1)*batch_size] = self.ir_model(q_batch, c_batch, 
															qlen_batch, clen_batch)
		return scores

	# def getSentScores(self, q, c, qlen, clen, batch_size):
	# 	return self.ir_model(q, c, qlen, clen)

	def forward(self, q, c, a, qlen, clen, alen, c_batch_size=512):
		print(q.shape, c.shape, a.shape)
		selected_sents = []
		scores = []
		
		print(torch.cuda.memory_allocated(0) / (1024)**3)

		for i in range(len(q)):
			q_ = q[i, :qlen[i]]
			q_ = q_.expand(c.shape[0], -1, -1)

			qlen_ = qlen[i:i+1].expand(q_.shape[0])	
			with torch.no_grad():	
				c_scores = self.getSentScores(q_, c, qlen_, clen, c_batch_size)
			print(c_scores.shape)			

			_, topk_idx = torch.topk(c_scores, self.n_ctx_sents, dim=0)

			sents = c[topk_idx]
			sent_lens = clen[topk_idx]
			sents = [sents[j,:sent_lens[j]] for j in range(self.n_ctx_sents)]

			ctx = torch.cat(sents, dim=0)

			selected_sents.append(ctx)
			scores.append(c_scores)

		ctx_len = torch.tensor([len(s) for s in selected_sents]).long().to(c.device)
		max_ctx_len = max(ctx_len)

		ctx = torch.zeros(len(selected_sents), max_ctx_len, c.shape[2]).long().to(c.device)
		for (i, sents) in enumerate(selected_sents):
			ctx[i,:len(sents)] = sents

		scores = torch.stack(scores, dim=0)

		c_mask = utils.output_mask(ctx_len)
		q_mask = utils.output_mask(qlen)
		loss1, loss2, sidx, eidx = self.rc_model(ctx[:,:,0], ctx[:,:,1], c_mask, 
												q[:,:,0], q[:,:,1], q_mask, 
												None, a, alen)
		return loss1

