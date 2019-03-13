import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import rnn 

from modules import *

def test_wrapper(test_name, test_outcome):	
	if int(test_outcome) != 1:
		print "%s...%s"%(test_name, "FAIL")
	return int(test_outcome)

def test_interactive_aligner(t1, t2, t1_lens, t2_lens):
	print 'test_interactive_aligner\n================='
	success = 1
	ia = InteractiveAligner(t1.shape[-1])
	H, h_lens, E = ia(t1, t2, t1_lens, t2_lens)

	m = t1.shape[1]
	n = t2.shape[1]
	success &= test_wrapper("checking size of H", t1.shape == H.shape)
	success &= test_wrapper("checking h_lens", torch.all(t1_lens == h_lens))
	success &= test_wrapper('checking shape of E', E.shape[0] == t1.shape[0] and
										E.shape[1] == t2.shape[1] and
										E.shape[2] == t1.shape[1])	

	for i in range(len(t1_lens)):
		success &= test_wrapper("checking masking of cols in batch %d of E" % i, 
			torch.all(E[i, :, t1_lens[i]:] == 0))
	
	for i in range(len(h_lens)):
		success &= test_wrapper("checking masking of rows in batch %d of H" % i, 
			torch.all(H[i, h_lens[i]:] == 0))

	print "test_interactive_aligner...%s\n" % bool(success)
def test_self_aligner(t, t_lens):
	print 'test_self_aligner\n================='
	sa = SelfAligner(t.shape[-1])
	Ht, ht_lens, B = sa(t, t_lens)

	success = 1

	success &= test_wrapper("checking size of H", t.shape == Ht.shape)
	success &= test_wrapper("checking h_lens", torch.all(t_lens == ht_lens))
	success &= test_wrapper('checking shape of E', B.shape[0] == t.shape[0] and
										B.shape[1] == t.shape[1] and
										B.shape[2] == t.shape[1])	

	for i in range(len(t_lens)):
		success &= test_wrapper("checking masking of cols in batch %d of B" % i, 
			torch.all(B[i, :, t_lens[i]:] == 0))
	
	for i in range(B.shape[0]):
		success &= test_wrapper('checking diagonals of batch %d of B'%i, 
			torch.all(torch.diag(B[i] == 0)))

	for i in range(len(ht_lens)):
		success &= test_wrapper("checking masking of rows in batch %d of Ht" % i, 
			torch.all(Ht[i, ht_lens[i]:] == 0))
	
	print "test_self_aligner...%s\n" % bool(success)

if __name__ == '__main__':
	ts1 = [torch.arange(0,2*(i+2)).float().view(-1,2) + 1 for i in range(4,-1, -1)]	
	ts1_lens = torch.tensor([len(ts) for ts in ts1])
	ts1 = rnn.pad_sequence(ts1, batch_first=True)
	ts1_mask = output_mask(ts1_lens).unsqueeze(2)	

	ts2 = [torch.arange(0,2*(i)).float().view(-1,2) + 1 for i in range(4,-1, -1)]
	ts2_lens = torch.tensor([len(ts) for ts in ts2])
	ts2 = rnn.pad_sequence(ts2, batch_first=True)
	ts2_mask = output_mask(ts2_lens).unsqueeze(2)	

	print ts1.shape, ts2.shape	
	test_interactive_aligner(ts1, ts2, ts1_lens, ts2_lens)
	test_self_aligner(ts1, ts1_lens)