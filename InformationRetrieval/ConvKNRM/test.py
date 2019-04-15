import numpy as np
import torch
from modules import *

if __name__ == '__main__':
	np.random.seed(0)
	torch.manual_seed(0)

	model = ConvKNRM(vocab_size=10, emb_dim=5, nfilters=4)
	q = torch.randint(10, size=(2,6))
	d = torch.randint(10, size=(2,7))
	print(model(q,d))
