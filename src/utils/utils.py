import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import rnn 

def output_mask(lengths, maxlen=None):
    if maxlen is None:
        maxlen = torch.max(lengths)
    lens = lengths.unsqueeze(0)
    ran = torch.arange(0, maxlen, out=lengths.new()).unsqueeze(1)
    mask = (ran < lens).float()   
    return mask.transpose(0,1) # (batch_size, maxlen)

def masked_softmax(E, mask, dim):
    E = (E - torch.max(E, dim, keepdim=True)[0])    
    E = torch.exp(E) 
    if mask is not None:
        E = E * mask
    E = E / torch.sum(E, dim=dim, keepdim=True)
    return E

def getAlignmentMatrix(v, u, mask=None, prev=None):
    E = torch.bmm(v,u.transpose(1,2))
    if prev is not None:
        E = E + prev
    E = masked_softmax(E, mask, 1)
    return E
    
def reset_embeddings(word_embeddings, fixed_embeddings, trained_idx):
    word_embeddings.weight.data[trained_idx] = torch.FloatTensor(fixed_embeddings[trained_idx]).to(word_embeddings.weight.data.device)
    return 