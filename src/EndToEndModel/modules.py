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
<<<<<<< HEAD
	def __init__(self, ir_model, rc_model, ag_model, n_ctx_sents=3):
=======
	def __init__(self, ir_model, rc_model, n_ctx_sents=50, include_novelty_score=False):
>>>>>>> ahmed
		super(EndToEndModel, self).__init__()
		self.ir_model1 = ir_model.eval()
		self.ir_model2 = ir_model
		self.rc_model = rc_model
		self.ag_model = ag_model
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


	# def forward(self, q, c, avec1, avec2, qlen, clen, alen, p_words, c_batch_size=512):
	# 	print(q.shape, c.shape, avec1.shape)
	# 	selected_sents = []
	# 	scores = []
		
	# 	print(torch.cuda.memory_allocated(0) / (1024)**3)
	# 	string_sents = []
	# 	for i in range(len(q)):
	# 		q_ = q[i, :qlen[i]]
	# 		#print (q_.shape)
	# 		q_ = q_.expand(c.shape[0], -1, -1)

	# 		qlen_ = qlen[i:i+1].expand(q_.shape[0])	
	# 		#c_scores = self.ir_model(q_, c, qlen_, clen)
	# 		with torch.no_grad():	
	# 			c_scores = self.ir_model(q_, c, qlen_, clen)
	# 		#print(c_scores.shape)			

	def forward(self, q, c, avec1, avec2, qlen, clen, alen, p_words, c_batch_size=512):
		# print(q.shape, c.shape, a.shape)
		selected_sents = []		
<<<<<<< HEAD
		string_sents = []
=======
		top_sents = []
		top_sents_len = []
>>>>>>> ahmed
		# print(torch.cuda.memory_allocated(0) / (1024)**3)
		
		# if q.shape[0] == 1:
		# 	q = q.expand(2, -1, -1)
		# 	qlen = qlen.expand(2)

		# 	avec1 = avec1.exp
		
		with torch.no_grad():
			c_scores = self.ir_model1.forward_singleContext(q, c, qlen, clen,
														batch_size=c_batch_size)
			
<<<<<<< HEAD
			_, topk_idx_ir1 = torch.topk(c_scores, self.n_ctx_sents*2, dim=1, sorted=False)
=======
			top_scores, topk_idx = torch.topk(c_scores, self.n_ctx_sents*2, dim=1, sorted=False)
>>>>>>> ahmed
			
			ctx1 = [c[topk_idx_ir1[i]] for i in range(len(c_scores))]
			ctx_len1 = [clen[topk_idx_ir1[i]] for i in range(len(c_scores))]
			
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

<<<<<<< HEAD

			# sents = c[topk_idx]
			# sent_lens = clen[topk_idx]
			# sents = [sents[j,:sent_lens[j]] for j in range(self.n_ctx_sents)]
			string_sent = [p_words[_idx] for _idx in topk_idx_ir1[topk_idx]]
			string_sent = [w for s in string_sent for w in s]
			string_sents.append(string_sent)

=======
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
>>>>>>> ahmed
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
		
		print(ctx.shape, q.shape)
		loss1, loss2, sidx, eidx, extracted_span = self.rc_model(ctx[:,:,0], ctx[:,:,1], ctx_len, 
												q[:,:,0], q[:,:,1], qlen, 
												None, avec1, alen)
		print (extracted_span.shape)
		print (sidx, eidx)
		raw_span = []
		for i in range(len(string_sents)):
			raw_span.append(['<sos>'] + string_sents[i][sidx[i]:eidx[i]+1] + ['<eos>'])
			print (len(raw_span[-1]))

		gen_loss, output = self.ag_model(extracted_span, avec2, raw_span)
		return loss1+gen_loss

	def evaluate(self, q, c, qlen, clen, p_words, c_batch_size=512):
		selected_sents = []
		scores = []
		
		string_sents = []
		for i in range(len(q)):
			q_ = q[i, :qlen[i]]
			q_ = q_.expand(c.shape[0], -1, -1)

			qlen_ = qlen[i:i+1].expand(q_.shape[0])	
			#c_scores = self.ir_model(q_, c, qlen_, clen)
			
			c_scores = self.ir_model(q_, c, qlen_, clen)
			#print(c_scores.shape)			

			_, topk_idx = torch.topk(c_scores, self.n_ctx_sents, dim=0)

			sents = c[topk_idx]
			sent_lens = clen[topk_idx]
			sents = [sents[j,:sent_lens[j]] for j in range(self.n_ctx_sents)]
			string_sent = [p_words[_idx] for _idx in topk_idx]
			string_sent = [w for s in string_sent for w in s]
			string_sents.append(string_sent)

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
		sidx, eidx, extracted_span = self.rc_model.evaluate(ctx[:,:,0], ctx[:,:,1], c_mask, 
												q[:,:,0], q[:,:,1], q_mask)
		raw_span = []
		for i in range(len(string_sents)):
			raw_span.append(['<sos>'] + string_sents[i][sidx[i]:eidx[i]+1] + ['<eos>'])

		gen_loss, output = self.ag_model(extracted_span, avec2, raw_span)
		return gen_loss

		# # print(type(ctx_len), type(qlen))
		# loss1, loss2, sidx, eidx = self.rc_model(ctx[:,:,0], ctx[:,:,1], ctx_len, 
		# 										q[:,:,0], q[:,:,1], qlen, 
		# 										None, a, alen)
		# return loss1


