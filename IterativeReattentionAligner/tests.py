import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import rnn 

from modules import *

def test_simple_aligning_block():
	ts1 = [torch.arange(0,2*(i+2)).float().view(-1,2) for i in range(4,-1, -1)]
	ts1_lens = torch.tensor([len(ts) for ts in ts1])
	padded_ts1 = rnn.pad_sequence(ts1, batch_first=True)
	ts1_mask = output_mask(ts1_lens).unsqueeze(2)
	ts1 = (padded_ts1+1)

	ts2 = [torch.arange(0,2*(i)).float().view(-1,2) for i in range(4,-1, -1)]
	ts2_lens = torch.tensor([len(ts) for ts in ts2])
	padded_ts2 = rnn.pad_sequence(ts2, batch_first=True)
	ts2_mask = output_mask(ts2_lens).unsqueeze(2)
	ts2 = (padded_ts2+1)		
	
	alignB = SimpleAligningBlock(2)
	attented_v, lens = alignB(ts1, ts2, ts1_lens, ts2_lens)