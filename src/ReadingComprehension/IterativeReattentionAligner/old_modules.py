import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import rnn 

def output_mask(lengths, maxlen=None):
    #print (lengths)
    if maxlen is None:
        maxlen = torch.max(lengths)
    lens = lengths.unsqueeze(0)
    ran = torch.arange(0, maxlen, out=lengths.new()).unsqueeze(1)
    mask = (ran < lens).float()   
    return mask.transpose(0,1) # (batch_size, maxlen)

def masked_softmax(E, mask, dim):
    E = (E - torch.max(E, dim, keepdim=True)[0])
    E = torch.exp(E) 
    #print (E.device, mask.device)
    if mask is not None:
        E = E * mask
    E = E / torch.sum(E, dim=dim, keepdim=True)
    return E

def getAligmentMatrix(v, u, mask=None):
    E = torch.bmm(v,u.transpose(1,2))
    #E = masked_softmax(E, mask.transpose(1,2), 2) 
    E = torch.nn.functional.softmax(E, dim=1)  
    E = E * mask.unsqueeze(2)
    return E        

class InteractiveAligner(nn.Module):
    """docstring for SimpleAligningBlock"""
    def __init__(self, enc_dim):
        super(InteractiveAligner, self).__init__()
        
        self.W_u = nn.Linear(enc_dim, enc_dim, bias=False) # enc_dim x n
        self.W_v = nn.Linear(enc_dim, enc_dim, bias=False) # enc_dim x m
        self.fusion = Fusion(enc_dim)

    def forward(self, u, v, u_mask, v_mask):
        # u and v should be padded sequences
        # u.shape = B x m x enc_dim
        # v.shape = B x n x end_dim
        
        #print "batch_size=%d, m=%d, n=%d, dim=%d" % (u.shape[0], u.shape[1], v.shape[1], v.shape[2])
        
        v_proj = F.relu(self.W_v(v))
        #v_mask = output_mask(v_lens).unsqueeze(2)
        v_proj = v_proj * v_mask.unsqueeze(2)

        u_proj = F.relu(self.W_u(u))
        #u_mask = output_mask(u_lens).unsqueeze(2)
        u_proj = u_proj * u_mask.unsqueeze(2)

        # E.shape = B x n x m
        E = getAligmentMatrix(v_proj, u_proj, mask=v_mask)        # should we use v_mask instead? Also the softmax dimension may be wrong? 
        attended_v = torch.bmm(v.transpose(1,2), E).transpose(1,2)
        #print "batch_size=%d, m=%d, dim=%d" % (attended_v.shape[0], attended_v.shape[1], attended_v.shape[2])

        fused_u = self.fusion(u, attended_v)
        #print "batch_size=%d, n=%d, dim=%d" % (fused_u.shape[0], fused_u.shape[1], fused_u.shape[2])       
<<<<<<< HEAD
        return fused_u, u_lens, E
=======
        return fused_u
>>>>>>> fc951534425d65ce4ded608830b923e743ec7ccc

class Fusion(nn.Module):
    """docstring for Fusion"""
    def __init__(self, enc_dim):
        super(Fusion, self).__init__()
        
        self.projection = nn.Linear(4*enc_dim, enc_dim, bias=False)
        self.gate = nn.Linear(4*enc_dim, enc_dim, bias=False)

    def forward(self, x, y):
        catted = torch.cat((x,y,x*y,x-y), dim=2)
        xt = F.relu(self.projection(catted))
        g = torch.sigmoid(self.gate(catted))
        o = g*xt + (1-g)*x
        return o

class SelfAligner(nn.Module):
    """docstring for SelfAlign"""
    def __init__(self, enc_dim):
        super(SelfAligner, self).__init__()
        self.fusion = Fusion(enc_dim)

    def forward(self, x, x_mask):
        #x_mask = output_mask(x_lens).unsqueeze(2)
        x = x * x_mask.unsqueeze(2)
        E = getAligmentMatrix(x, x, x_mask)
        attended_x = torch.bmm(x.transpose(1,2), E).transpose(1,2)
        fused_x = self.fusion(x, attended_x)        
<<<<<<< HEAD
        return fused_x, x_lens, E
=======
        return fused_x
>>>>>>> fc951534425d65ce4ded608830b923e743ec7ccc

class Summarizer(nn.Module):
    """docstring for Summarizer"""
    def __init__(self, enc_dim):
        super(Summarizer, self).__init__()
        self.W = nn.Linear(enc_dim, 1, bias=False)

    def forward(self, v, v_mask):
        alpha = self.W(v)
        #v_mask = output_mask(v_lens).unsqueeze(2)  
        
        alpha = torch.nn.functional.softmax(alpha, dim=1) 
        alpha = alpha * v_mask.unsqueeze(2)    
        #alpha = masked_softmax(alpha, v_mask, 1)
        #print "alpha.shape=",alpha.shape

        s = torch.sum(alpha*v, dim=1, keepdim=True)
        #print 's.shape=',s.shape

        return s

class AnswerPointer(nn.Module):
    """docstring for AnswerPointer"""
    def __init__(self, enc_dim):
        super(AnswerPointer, self).__init__()
        self.summarizer = Summarizer(enc_dim)
        self.W1 = nn.Linear(4*enc_dim, enc_dim, bias=False)
        self.w1 = nn.Linear(enc_dim, 1, bias=False)
        self.W2 = nn.Linear(4*enc_dim, enc_dim, bias=False)
        self.w2 = nn.Linear(enc_dim, 1, bias=False)
        self.fusion = Fusion(enc_dim)

    def computeP(self, s, r, u_mask):
        #print "r.shape", r.shape
        catted = torch.cat((r, s, r*s, r-s), dim=2)
        score1 = self.w1(torch.tanh(self.W1(catted)))
        #r_mask = output_mask(r_lens).unsqueeze(2)       
        #p = masked_softmax(score1, r_mask, 1)
        #p = score1 * u_mask.unsqueeze(2)
        return score1

    def forward(self, v, v_mask, R, u_mask):
        s = self.summarizer(v, v_mask)      
        s = s.expand(s.shape[0], R.shape[1], s.shape[2])        
        p1 = self.computeP(s, R, u_mask)

        l = p1*R
        st = self.fusion(s, l)
        p2 = self.computeP(st, R, u_mask)
        return p1, p2


class AligningBlock(nn.Module):
    """docstring for AligningBlock"""
    def __init__(self, enc_dim, hidden_size, n_hidden):
        super(AligningBlock, self).__init__()
        self.interactive_aligner = InteractiveAligner(enc_dim)
        self.self_aligner = SelfAligner(enc_dim)
        self.evidence_collector = nn.LSTM(enc_dim, hidden_size, n_hidden, 
                            batch_first=True, bidirectional=True)        

    def forward(self, u, v, u_lens, v_lens, Et=None, Bt=None, prev_Zs=None):
        H, h_lens, E = self.interactive_aligner(u, v, u_lens, v_lens, prev=Et)
        Z, z_lens, B = self.self_aligner(H, h_lens, prev=Bt)
        
        if prev_Zs is not None:
            for z in prev_Zs:            
                Z += z
                            
        z_lens, sorted_idxs = torch.sort(z_lens, descending=True)
        rev_sorted_idxs = sorted(range(len(sorted_idxs)), key=lambda i: sorted_idxs[i])

        Z = Z[sorted_idxs]
        packed_Z = rnn.pack_padded_sequence(Z, z_lens, batch_first=True)
        R, _ = self.evidence_collector(packed_Z)
        R, r_lens = rnn.pad_packed_sequence(R, batch_first=True)    

        R = R[rev_sorted_idxs]
        r_lens = r_lens[rev_sorted_idxs]
        r_lens = r_lens.to(R.device)

        Z = Z[rev_sorted_idxs]
        z_lens = z_lens[rev_sorted_idxs]

        return R, Z, E, B, r_lens, z_lens, h_lens        

class IterativeAligner(nn.Module):
    """docstring for IterativeAligner"""
    def __init__(self, enc_dim, hidden_size, n_hidden, niters):
        super(IterativeAligner, self).__init__()
        self.aligning_block = AligningBlock(enc_dim, hidden_size, n_hidden)
        self.y = nn.Parameter(torch.rand(1))
        self.answer_pointer = AnswerPointer(enc_dim)
        assert (niters >= 1)
        self.niters = niters

    def forward(self, u, v, u_mask, v_mask):
        u_lens = torch.sum(u_mask, 1)
        v_lens = torch.sum(v_mask, 1)

        R, Z, E, B, r_lens, z_lens, h_lens = self.aligning_block(u, v, u_lens, v_lens)
        if self.niters > 1:
            Zs = [Z]
            Et = self.y * torch.bmm(E, B)
            Bt = self.y * torch.bmm(B, B)

            for i in range(self.niters-2):
                R, Z, E, B, r_lens, z_lens, h_lens = self.aligning_block(R, v,
                                                         r_lens, v_lens, Et=Et,
                                                         Bt=Bt)
                Zs.append(Z)
                Et = self.y * torch.bmm(E, B)
                Bt = self.y * torch.bmm(B, B)
            R, Z, E, B, r_lens, z_lens, h_lens = self.aligning_block(R, v,
                                                         r_lens, v_lens, Et=Et,
                                                         Bt=Bt, prev_Zs=Zs)
        
        p1, p2 = self.answer_pointer(v, v_lens, R, r_lens)
        # print "p1.shape", p1.shape
        # print 'p2.shape', p2.shape
        return p1, p2

if __name__ == '__main__':
    ts1 = [torch.arange(0,2*(i+2)).float().view(-1,2) for i in range(4)]
    ts1_lens = torch.tensor([len(ts) for ts in ts1])
    padded_ts1 = rnn.pad_sequence(ts1, batch_first=True)
    ts1_mask = output_mask(ts1_lens).unsqueeze(2)
    ts1 = (padded_ts1+1)

    ts2 = [torch.arange(0,2*(i)).float().view(-1,2) for i in range(4)]
    ts2_lens = torch.tensor([len(ts) for ts in ts2])
    padded_ts2 = rnn.pad_sequence(ts2, batch_first=True)
    ts2_mask = output_mask(ts2_lens).unsqueeze(2)
    ts2 = (padded_ts2+1)

    #print padded_ts1.shape, ts1_mask.shape
    #print padded_ts2.shape, ts2_mask.shape      
    
    alignB = AligningBlock(2, 1, 1)
    p1,p2 = alignB(ts1, ts2, ts1_lens, ts2_lens)        
    #print p1.shape, p2.shape
