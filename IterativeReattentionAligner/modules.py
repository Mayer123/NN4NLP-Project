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
    # inv_mask = mask.clone()   
    # print E[2] 
    # E = E * mask
    # print E[2]
    # E[E == 0] = -float('inf')
    # print E[2], mask[2]
    # E = F.softmax(E, dim=dim)
    # return E
    E = (E - torch.max(E, dim, keepdim=True)[0])    
    E = torch.exp(E) 
    #print (E.device, mask.device)
    if mask is not None:
        E = E * mask
    E = E / torch.sum(E, dim=dim, keepdim=True)
    return E

def getAligmentMatrix(v, u, mask=None, prev=None):
    E = torch.bmm(v,u.transpose(1,2))
    if prev is not None:
        E = E + prev
    E = masked_softmax(E, mask, 1)
    # E[E == 0] = -float('inf')
    # E = F.softmax(E, 1)    
    return E        

class InteractiveAligner(nn.Module):
    """docstring for SimpleAligningBlock"""
    def __init__(self, enc_dim):
        super(InteractiveAligner, self).__init__()
        
        self.W_u = nn.Linear(enc_dim, enc_dim, bias=False) # enc_dim x n
        self.W_v = nn.Linear(enc_dim, enc_dim, bias=False) # enc_dim x m
        self.fusion = Fusion(enc_dim)

    def forward(self, u, v, u_lens, v_lens, prev=None):
        # u and v should be padded sequences
        # u.shape = B x m x enc_dim
        # v.shape = B x n x end_dim
        
        # print "batch_size=%d, m=%d, n=%d, dim=%d" % (u.shape[0], u.shape[1], v.shape[1], v.shape[2])        
        v_proj = F.relu(self.W_v(v))
        v_mask = output_mask(v_lens).unsqueeze(2)
        
        v_proj = v_proj * v_mask

        u_proj = F.relu(self.W_u(u))
        u_mask = output_mask(u_lens).unsqueeze(2)
        u_proj = u_proj * u_mask

        # E.shape = B x n x m
        E = getAligmentMatrix(v_proj, u_proj, mask=v_mask, prev=prev)  #q_tilde 

        attended_v = torch.bmm(v.transpose(1,2), E).transpose(1,2)
        attended_v = attended_v * u_mask
        # print E[4], v[4], attended_v[4]
        # print "batch_size=%d, m=%d, dim=%d" % (attended_v.shape[0], attended_v.shape[1], attended_v.shape[2])
        fused_u = self.fusion(u, attended_v) # c_bar  
        # print u_lens[1], u[1], attended_v[1], fused_u[1]
        #print "batch_size=%d, n=%d, dim=%d" % (fused_u.shape[0], fused_u.shape[1], fused_u.shape[2])       
        return fused_u, u_lens, E

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

    def forward(self, x, x_lens, prev=None):
        x_mask = output_mask(x_lens).unsqueeze(2)
        x = x * x_mask
        
        I = torch.eye(x_mask.shape[1], device=x_mask.device).unsqueeze(0)        
        x_mask_ = torch.abs(I-1) * x_mask

        E = getAligmentMatrix(x, x, x_mask_, prev=prev)                      
        attended_x = torch.bmm(x.transpose(1,2), E).transpose(1,2)        
        attended_x = attended_x * x_mask      #c_tilde  

        fused_x = self.fusion(x, attended_x)    # c_hat     
        return fused_x, x_lens, E

class Summarizer(nn.Module):
    """docstring for Summarizer"""
    def __init__(self, enc_dim):
        super(Summarizer, self).__init__()
        self.W = nn.Linear(enc_dim, 1, bias=False)

    def forward(self, v, v_lens):
        alpha = self.W(v)
        v_mask = output_mask(v_lens).unsqueeze(2)       
        alpha = masked_softmax(alpha, v_mask, 1)
        #print "alpha.shape=",alpha.shape

        s = torch.sum(alpha*v, dim=1, keepdim=True)
        #print 's.shape=',s.shape

        return s

class AnswerPointer(nn.Module):
    """docstring for AnswerPointer"""
    def __init__(self, enc_dim, dropout=0):
        super(AnswerPointer, self).__init__()
        self.summarizer = Summarizer(enc_dim)
        self.mlp1 = nn.Sequential(
                        nn.Linear(4*enc_dim, enc_dim, bias=False),
                        nn.Tanh(),
                        nn.Dropout(dropout),
                        nn.Linear(enc_dim, 1, bias=False)      
                    )
        self.mlp2 = nn.Sequential(
                        nn.Linear(4*enc_dim, enc_dim, bias=False),
                        nn.Tanh(),
                        nn.Dropout(dropout),
                        nn.Linear(enc_dim, 1, bias=False)      
                    )
        # self.W1 = nn.Linear(4*enc_dim, enc_dim, bias=False)
        # self.w1 = nn.Linear(enc_dim, 1, bias=False)
        # self.W2 = nn.Linear(4*enc_dim, enc_dim, bias=False)
        # self.w2 = nn.Linear(enc_dim, 1, bias=False)
        self.fusion = Fusion(enc_dim)

    def computeP(self, s, r, r_lens, mlp):
        #print "r.shape", r.shape
        catted = torch.cat((r, s, r*s, r-s), dim=2)
        score1 = mlp(catted)
        r_mask = output_mask(r_lens).unsqueeze(2)               
        p = masked_softmax(score1, r_mask, 1)
        # p = torch.exp(score1) * r_mask
        return p

    def forward(self, v, v_lens, R, r_lens):
        s = self.summarizer(v, v_lens)      
        s = s.expand(s.shape[0], R.shape[1], s.shape[2])        
        p1 = self.computeP(s, R, r_lens, self.mlp1)

        l = p1*R
        st = self.fusion(s, l)
        p2 = self.computeP(st, R, r_lens, self.mlp2)        

        p = torch.bmm(p1, p2.transpose(1,2))

        eps = 1e-8
        p = (1-eps)*p + eps*torch.min(p[p != 0])
        p = torch.log(p)
        p = p.view(p.shape[0], -1)

        return p

        
class AligningBlock(nn.Module):
    """docstring for AligningBlock"""
    def __init__(self, enc_dim, hidden_size, n_hidden, dropout=0):
        super(AligningBlock, self).__init__()
        self.interactive_aligner = InteractiveAligner(enc_dim)
        self.self_aligner = SelfAligner(enc_dim)
        self.evidence_collector = nn.Sequential(
                                    nn.Dropout(dropout),
                                    nn.LSTM(enc_dim, hidden_size, n_hidden,
                                        batch_first=True, bidirectional=True),                                    
                                  )
        self.dropout = nn.Dropout(dropout)
    def forward(self, u, v, u_lens, v_lens, Et=None, Bt=None, prev_Zs=None):
        H, h_lens, E = self.interactive_aligner(u, v, u_lens, v_lens, prev=Et)
        Z, z_lens, B = self.self_aligner(H, h_lens, prev=Bt) #c_hat
        
        if prev_Zs is not None:
            for z in prev_Zs:            
                Z += z
                            
        # z_lens, sorted_idxs = torch.sort(z_lens, descending=True)
        # rev_sorted_idxs = sorted(range(len(sorted_idxs)), key=lambda i: sorted_idxs[i])

        # Z = Z[sorted_idxs]
        # packed_Z = rnn.pack_padded_sequence(Z, z_lens, batch_first=True)
        R, _ = self.evidence_collector(Z)
        R = self.dropout(R) # c_check
        # R, r_lens = rnn.pad_packed_sequence(R, batch_first=True)    

        # R = R[rev_sorted_idxs]
        # r_lens = r_lens[rev_sorted_idxs]
        # r_lens = r_lens.to(R.device)

        # Z = Z[rev_sorted_idxs]
        # z_lens = z_lens[rev_sorted_idxs]

        return R, Z, E, B, z_lens, z_lens, h_lens        

class IterativeAligner(nn.Module):
    """docstring for IterativeAligner"""
    def __init__(self, enc_dim, hidden_size, n_hidden, niters, dropout=0):
        super(IterativeAligner, self).__init__()
        self.aligning_block = AligningBlock(enc_dim, hidden_size, n_hidden)
        self.y = nn.Parameter(torch.rand(1))
        self.answer_pointer = AnswerPointer(enc_dim, dropout=dropout)
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
        
        p = self.answer_pointer(v, v_lens, R, r_lens)        
        # print "p1.shape", p1.shape
        # print 'p2.shape', p2.shape
        return p
        

if __name__ == '__main__':
    ts1 = [torch.arange(0,2*(i+2)).float().view(-1,2)+1 for i in range(4)]
    ts1_lens = torch.tensor([len(ts) for ts in ts1])
    padded_ts1 = rnn.pad_sequence(ts1, batch_first=True)
    ts1_mask = output_mask(ts1_lens)
    ts1 = (padded_ts1)

    ts2 = [torch.arange(0,2*(i+1)).float().view(-1,2)+1 for i in range(4)]
    ts2_lens = torch.tensor([len(ts) for ts in ts2])
    padded_ts2 = rnn.pad_sequence(ts2, batch_first=True)
    ts2_mask = output_mask(ts2_lens)
    ts2 = (padded_ts2)

    print(padded_ts1.shape, ts1_mask.shape)
    print(padded_ts2.shape, ts2_mask.shape)
    
    alignB = IterativeAligner(2, 1, 1, 1)
    p = alignB(ts1, ts2, ts1_mask, ts2_mask)        
    print(p[1], ts1[1], p.shape)    
    max_idxs = torch.argmax(p, dim=1)
    
    # print(p1[1], ts1[1], p1.shape) 
    # print(p2[1], ts1[1], p2.shape)
