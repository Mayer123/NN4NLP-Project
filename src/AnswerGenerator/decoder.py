import sys
sys.path.append('../')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from utils import utils
import matplotlib.pyplot as plt

def plotAttn(attn):
    attn = np.array([x.detach().cpu().numpy() for x in attn])
    plt.pcolormesh(attn)
    plt.savefig('attn.png')
    plt.clf()
    plt.close()

def getNext(logits, n_next=1, gumbel=True, sample_dist=False):
	    if gumbel:
	        logit = F.gumbel_softmax(logits[-1])
	    else:
	        logit = logits[-1]
	        logit = F.softmax(logit, dim=1)

	    if sample_dist:
	    	pred_seq = torch.multinomial(logit, n_next).squeeze().long()
	    else:	        
	        pred_seq = torch.max(logit, 1)[1]                
	        pred_seq = pred_seq.squeeze().long()
	    return pred_seq

class Attention(nn.Module):
	"""docstring for Attention"""
	def __init__(self, input_dim, proj_dim, att_dropout=0.3, proj_dropout=0.1):
		super(Attention, self).__init__()
		self.k_proj = nn.Linear(input_dim, proj_dim, bias=False)
		self.q_proj = nn.Linear(input_dim, proj_dim, bias=False)
		self.v_proj = nn.Linear(input_dim, proj_dim, bias=False)
		self.proj_dropout = nn.Dropout(proj_dropout)
		self.att_dropout = nn.Dropout(att_dropout)

	def forward(self, k, q, v, k_lens):
		enc_k = self.proj_dropout(self.k_proj(k))
		enc_q = self.proj_dropout(self.k_proj(q)).unsqueeze(1)
		enc_v = self.proj_dropout(self.k_proj(v))
		
		mask = utils.output_mask(k_lens, maxlen=k.shape[1]).unsqueeze(2)
		E = utils.getAlignmentMatrix(enc_k, enc_q, mask=mask)
		attended = torch.bmm(enc_v.transpose(1,2), E).squeeze(2)
		attended = self.att_dropout(attended)

		return attended, E	
		
class RNNEncoder(nn.Module):
	"""docstring for GRUEncoder"""
	def __init__(self, input_dim, hidden_size, n_hidden):
		super(RNNEncoder, self).__init__()
		self.input_dim = input_dim
		self.hidden_size = hidden_size

	def forward(self, x, xlen):
		xlen, sorted_idx = torch.sort(xlen, descending=True)
		_, rev_sorted_idx = torch.sort(sorted_idx)
		
		x = x[sorted_idx]
		x = nn.utils.rnn.pack_padded_sequence(x, xlen, batch_first=True)
		x, hidden = self.rnn(x)
		
		x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

		x = x[rev_sorted_idx]
		hidden = hidden.transpose(0,1)
		hidden = hidden.contiguous().view(hidden.shape[0], -1)

		return x, hidden

class GRUEncoder(RNNEncoder):
	"""docstring for GRUEncoder"""
	def __init__(self, input_dim, hidden_size, n_hidden, bidirectional=False):
		super(GRUEncoder, self).__init__(input_dim, hidden_size, n_hidden)
		self.rnn = nn.GRU(input_dim, hidden_size, n_hidden, batch_first=True, 
							bidirectional=bidirectional)

class seq2seqAG(nn.Module):
	"""docstring for seq2seq"""
	def __init__(self, encoder, decoder):
		super(seq2seqAG, self).__init__()
		self.encoder = encoder
		self.decoder = decoder

	def forward(self, c, clen, a, sos_idx):
		encoded, h = self.encoder(c, clen)
		logits = self.decoder(encoded, clen, a, sos_idx, initial_hidden=h)
		return logits

class GRUDecoder(nn.Module):
	"""docstring for GRUDecoder"""
	def __init__(self, emb_dim, hidden_size, n_hidden, vocab_size, word_embeddings=None,
					tf_rate=0.9, dropout=0.3, use_attention=False):
		super(GRUDecoder, self).__init__()
		
		if word_embeddings is None:
			self.word_embeddings = nn.Embedding(vocab_size, emb_dim)
		else:
			self.word_embeddings = word_embeddings
		self.emb_dropout = nn.Dropout(dropout)

		self.hidden_size = hidden_size												
		self.output = nn.Linear(hidden_size, vocab_size)

		self.use_attention = use_attention
		if use_attention:			
			self.attention = Attention(hidden_size, hidden_size)
			self.rnn_cells = [nn.GRUCell(emb_dim+hidden_size, hidden_size)]
		else:
			self.rnn_cells = [nn.GRUCell(emb_dim, hidden_size)]
		self.dropout = nn.Dropout(dropout)

		self.rnn_cells = nn.ModuleList(self.rnn_cells + 
								[nn.GRUCell(hidden_size, hidden_size) for i in range(n_hidden-1)])

		self.tf_rate = tf_rate	

	def forward(self, encoded, encoded_len, x, sos_idx, generated_seqlen=10, gumbel=False,
				initial_hidden=None):
		# x.shape = [seq_len, batch]
		# encoded.shape = [batch, seq_len', fdim]

		nbatches = encoded.shape[0]

		if initial_hidden is None:
			initial_hidden = torch.zeros(1, self.hidden_size)

		hidden_states = [initial_hidden.contiguous() 
											for cell in self.rnn_cells]
		h0 = hidden_states[-1]
		logits = []

		if x is not None:			
			T = x.shape[0]
		else:
			T = generated_seqlen

		for t in range(T):
			if t == 0:
				seq = torch.zeros(nbatches).to(encoded.device).long() + sos_idx
			else:
				pred_seq = getNext(logits, gumbel=gumbel, sample_dist=False).to(seq.device)
				if self.training:
					seq = x[t-1]
					if self.tf_rate > 0:
						flip = torch.bernoulli(torch.zeros(seq.shape[0]) + self.tf_rate).long().to(seq.device)

						seq = seq * (flip) + pred_seq * (1-flip)
				else:
					seq = pred_seq

			h = self.word_embeddings(seq)
			if self.use_attention:
				ctx,_ = self.attention(encoded, h0, encoded, encoded_len)
				h = torch.cat((h, ctx), dim=1)
			h = self.dropout(h)

			for i in range(len(self.rnn_cells)):
				h = self.rnn_cells[i](self.dropout(h), hidden_states[i])
				hidden_states[i] = h

			logit = self.output(h)
			logits.append(logit)
		logits = torch.stack(logits, dim=0)

		return logits

class LSTMDecoder(nn.Module):
	"""docstring for GRUDecoder"""
	def __init__(self, emb_dim, hidden_size, n_hidden, vocab_size, word_embeddings=None,
					tf_rate=0.9, dropout=0.3):
		super(GRUDecoder, self).__init__()
		
		if word_embeddings is None:
			self.word_embeddings = nn.Embedding(vocab_size, emb_dim)
		else:
			self.word_embeddings = word_embeddings

		self.rnn_cells = nn.ModuleList([nn.LSTMCell(emb_dim, hidden_size)] +
										[nn.LSTMCell(hidden_size, hidden_size) for i in range(n_hidden-1)])
		self.initial_c = nn.ModuleList([nn.Parameter(torch.zeros(1, hidden_size)) for i in range(n_hidden)])
		self.output = nn.Linear(hidden_size, vocab_size)
		self.tf_rate = tf_rate	

	def forward(self, encoder_out, x, xlen, sos_idx, generated_seqlen=None, gumbel=False):
		T = x.shape[0]
		nbatches = x.shape[1]

		hidden_states = [(encoder_out.contiguous(), self.initial_c[i].expand(nbatches,-1).contiguous()) 
								for i in range(len(self.rnn_cells))]
		h0, c0 = hidden_states[-1]
		logits = []

		if generated_seqlen is not None:
			T = generated_seqlen

		for t in range(T):
			if t == 0:
				seq = torch.zeros(nbatches).to(encoder_out.device).long() + sos_idx
			else:
				pred_seq = getNext(logits, gumbel=gumbel, sample_dist=False).to(seq.device)
				if self.training:
					seq = x[t-1]
					if self.tf_rate > 0:
						flip = torch.bernoulli(torch.zeros(seq.shape[0]) + self.tf_rate).long().to(seq.device)

						seq = seq * (flip) + pred_seq * (1-flip)
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