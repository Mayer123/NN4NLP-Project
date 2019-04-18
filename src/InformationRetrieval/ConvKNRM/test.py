import numpy as np
import torch
from ConvKNRM.modules import *

if __name__ == '__main__':
	np.random.seed(0)
	torch.manual_seed(0)

	model = ConvKNRM(vocab_size=10, emb_dim=5, nfilters=4)
	q = torch.randint(10, size=(2,6)).long()
	d = torch.randint(10, size=(2,7)).long()
	print(model(q, d, torch.tensor([6,4]).long(), torch.tensor([7,5]).long()))
	
	G = GaussianKernel(0,1)
	print(G(torch.tensor([[[0,0,0.5],[0,0,0.5]]])))