import sys
sys.path.append('../')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from utils import utils

class GRUDecoder(nn.Module):
	"""docstring for GRUDecoder"""
	def __init__(self, emb_dim, hidden_size, n_hidden, vocab_size, word_embeddings=None,
					tf_rate=0.9, dropout=0.3):
		super(GRUDecoder, self).__init__()
		
		if word_embeddings is None:
			self.word_embeddings = nn.Embedding(vocab_size, emb_dim)
		else:
			self.word_embeddings = word_embeddings

		self.rnn_cells = nn.ModuleList([nn.GRUCell(emb_dim, hidden_size)] +
										[nn.GRUCell(hidden_size, hidden_size) for i in range(n_hidden-1)])
		
		self.output = nn.Linear(hidden_size, vocab_size)
		self.tf_rate = tf_rate

	def getNext(self, logits, n_next=1, gumbel=True, argmax=False, sample_dist=False):
	    if gumbel:
	        logit = F.gumbel_softmax(logits[-1])
	    else:
	        logit = logits[-1]
	        logit = F.softmax(logit, dim=1)
	    if argmax:
	        pred_seq = torch.max(logit, 1)[1]                
	        pred_seq = pred_seq.squeeze().long()
	    elif sample_dist:
	        pred_seq = torch.multinomial(logit, n_next).squeeze().long()
	    return pred_seq

	def forward(self, encoder_out, x, xlen, sos_idx, generated_seqlen=None, gumbel=False):
		T = x.shape[0]
		nbatches = x.shape[1]

		hidden_states = [encoder_out.contiguous() 
											for cell in self.rnn_cells]
		h0 = hidden_states[-1]

		logits = []

		if generated_seqlen is not None:
			T = generated_seqlen

		for t in range(T):
			if t == 0:
				seq = torch.zeros(nbatches).to(encoder_out.device).long() + sos_idx
			else:
				pred_seq = self.getNext(logits, gumbel=gumbel, sample_dist=True).to(seq.device)
				if self.training:
					seq = x[t-1]
					if self.tf_rate > 0:
						flip = torch.bernoulli(torch.zeros(seq.shape[0]) + self.tf_rate).long().to(seq.device)

						seq = seq * (1-flip) + pred_seq * flip
				else:
					seq = pred_seq

			embed = self.word_embeddings(seq)
			h = embed
			for i in range(len(self.rnn_cells)):
				h = self.rnn_cells[i](h, hidden_states[i])
				hidden_states[i] = h

			logit = self.output(h)
			logits.append(logit)
		logits = torch.stack(logits, dim=0)
		return logits

if __name__ == '__main__':
	enc_out = torch.rand(2, 3)
	x = torch.randint(6, size=(4, 2)).long()

	decoder = GRUDecoder(5, 3, 2, 6)

	out = decoder(enc_out, x, torch.tensor([4, 2]).long(), 2, gumbel=True)
	print(out.shape)
	print(out)