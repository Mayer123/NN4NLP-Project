import sys
sys.path.append('../')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from utils import utils
from InformationRetrieval.NoveltyNTN.modules import NoveltyNTN
from utils.utils import output_mask

class GaussianKernel(object):
    """docstring for GaussianKernel"""
    def __init__(self, mean, std):
        super(GaussianKernel, self).__init__()
        self.mean = mean
        self.std = std

    def __call__(self, x):
        sim = torch.exp(-0.5 * (x-self.mean)**2 / self.std**2)      
        return sim

class WordOverlapLoss(nn.Module):
    """docstring for StringOverlapLoss"""
    def __init__(self):
        super(WordOverlapLoss, self).__init__()
        self.G = GaussianKernel(1.0, 0.001)

    def forward(self, s1, s2, s1_len, s2_len):
        # s1.shape = (batch, seq_len, emb_dim)
        # s1 should be the true answer

        normed_s1 = s1 / torch.norm(s1, dim=2, keepdim=True)
        normed_s2 = s2 / torch.norm(s2, dim=2, keepdim=True)

        cosine = torch.bmm(normed_s1, normed_s2.transpose(1,2)) # (batch, len(s1), len(s2))
        cosine_em = self.G(cosine)
        cosine_sm = cosine_em/(torch.sum(cosine_em, dim=2, keepdim=True) + 1e-10)
        cosine_scaled = cosine_em * cosine_sm

        mask = output_mask(s1_len)
        max_match = torch.sum(cosine_scaled, dim=2)

        max_match = max_match * mask
        tot_match = torch.sum(max_match, dim=1)
        ovrl = tot_match/s1_len.float()

        loss = 1 - ovrl
        return loss

class EndToEndModel(nn.Module):
	"""docstring for End2EndModel"""
	def __init__(self, ir_model, rc_model, n_ctx_sents=50, include_novelty_score=False):
		super(EndToEndModel, self).__init__()
		self.ir_model1 = ir_model.eval()
		self.ir_model2 = ir_model
		self.rc_model = rc_model
		self.n_ctx_sents = n_ctx_sents	

		self.loss = WordOverlapLoss()

	# def getSentScores(self, q, c, qlen, clen, batch_size):
	# 	scores = torch.zeros(c.shape[0]).float().to(c.device)
	# 	n_batches = (c.shape[0] + batch_size - 1) // batch_size
	# 	for i in range(n_batches):
	# 		c_batch = c[i*batch_size:(i+1)*batch_size]			
	# 		clen_batch = clen[i*batch_size:(i+1)*batch_size]

	# 		q_batch = q[i*batch_size:(i+1)*batch_size]
	# 		qlen_batch = qlen[i*batch_size:(i+1)*batch_size]

	# 		# print(c_batch.shape, clen_batch.shape)
	# 		# print(q_batch.shape, qlen_batch.shape)
	# 		scores[i*batch_size:(i+1)*batch_size] = self.ir_model(q_batch, c_batch, 
	# 														qlen_batch, clen_batch)
	# 	return scores

	def getSentScores(self, q, c, qlen, clen, batch_size):
		return self.ir_model(q, c, qlen, clen)

	def forward(self, q, c, a, qlen, clen, alen, c_batch_size=512):
		# print(q.shape, c.shape, a.shape)
		selected_sents = []		
		top_sents = []
		top_sents_len = []
		# print(torch.cuda.memory_allocated(0) / (1024)**3)
		with torch.no_grad():
			c_scores = self.ir_model1.forward_singleContext(q, c, qlen, clen,
														batch_size=c_batch_size)
			
			top_scores, topk_idx = torch.topk(c_scores, self.n_ctx_sents*2, dim=1, sorted=False)
			
			ctx1 = [c[topk_idx[i]] for i in range(len(c_scores))]
			ctx_len1 = [clen[topk_idx[i]] for i in range(len(c_scores))]
			
			ctx1 = torch.stack(ctx1, dim=0)
			ctx_len1 = torch.stack(ctx_len1, dim=0)			
		
		for i in range(len(ctx1)):
			c_scores = self.ir_model2.forward_singleContext(q[[i]], ctx1[i], qlen[[i]], ctx_len1[i],
															batch_size=c_batch_size)			
			_, topk_idx = torch.topk(c_scores[0], self.n_ctx_sents, dim=0)

			top_sents.append(ctx1[i, topk_idx[0]])
			top_sents_len.append(ctx_len1[i, topk_idx[0]])

			topk_idx, _ = torch.sort(topk_idx, descending=False)

			sents = ctx1[i, topk_idx]			
			sent_lens = ctx_len1[i, topk_idx]
			sents = [sents[j,:sent_lens[j]] for j in range(self.n_ctx_sents)]

			ctx2 = torch.cat(sents, dim=0)

			selected_sents.append(ctx2)

		top_sents = torch.stack(top_sents, dim=0)
		top_sents_len = torch.stack(top_sents_len, dim=0)

		a_embed = self.ir_model2.emb(a)
		c_embed = self.ir_model2.emb(top_sents[:,:,0])

		loss = self.loss(a_embed, c_embed, alen, top_sents_len)
		if not torch.isfinite(loss).all():
			print(top_sents)
			print(top_sents_len)			

		loss = torch.mean(loss)

		return loss
		# for i in range(len(c_scores)):
		# 	_, topk_idx = torch.topk(c_scores[i], self.n_ctx_sents, dim=0)

		# 	sents = c[topk_idx]
		# 	sent_lens = clen[topk_idx]
		# 	sents = [sents[j,:sent_lens[j]] for j in range(self.n_ctx_sents)]

		# 	ctx = torch.cat(sents, dim=0)

		# 	selected_sents.append(ctx)

		ctx_len = torch.tensor([len(s) for s in selected_sents]).long().to(c.device)
		max_ctx_len = max(ctx_len)

		ctx = torch.zeros(len(selected_sents), max_ctx_len, c.shape[2]).long().to(c.device)
		for (i, sents) in enumerate(selected_sents):
			ctx[i,:len(sents)] = sents

		# print(type(ctx_len), type(qlen))
		loss1, loss2, sidx, eidx = self.rc_model(ctx[:,:,0], ctx[:,:,1], ctx_len, 
												q[:,:,0], q[:,:,1], qlen, 
												None, a, alen)
		return loss1

