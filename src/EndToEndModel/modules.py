import sys
sys.path.append('../')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from utils import utils
from ReadingComprehension.IterativeReattentionAligner.CSMrouge import RRRouge

class EndToEndModel(nn.Module):
    """docstring for End2EndModel"""
    def __init__(self, ir_model1, ir_model2, rc_model, ag_model, w2i, c2i, 
                    n_ctx1_sents=6, n_ctx2_sents=5, chunk_size=15, use_ir2=False):
        super(EndToEndModel, self).__init__()
        self.ir_model1 = ir_model1
        self.ir_model2 = ir_model2
        self.rc_model = rc_model
        self.ag_model = ag_model
        self.n_ctx1_sents = n_ctx1_sents
        self.n_ctx2_sents = n_ctx2_sents
        self.chunk_size = chunk_size
        self.i2w = {v:k for k,v in w2i.items()}
        self.c2i = c2i
        self.ir_loss = nn.NLLLoss()
        self.rrrouge = RRRouge()
        self.use_ir2 = use_ir2

    def idx2words(self, idxs):
        return [self.i2w[i.data] for i in idxs]

    def getRouge(self, spans, a1, a2):        
        return torch.Tensor([self.rrrouge.calc_score([" ".join(span)], [a1, a2]) for span in spans])

    def getSpans(self, q, c, c_chars, qlen, clen, p_words, c_rouge=None, a1=None, a2=None, c_batch_size=512):
        # print(q.shape, c.shape, a.shape)
        selected_sents = []
        selected_sents_chars = []
        string_sents = []                    

        c_scores = self.ir_model1.forward_singleContext(q, c, qlen, clen,
                                                    batch_size=c_batch_size)
        c_scores = c_scores
        c_scores = torch.log(F.gumbel_softmax(c_scores))

        if self.ir_model1.training:
            best_sent_idx = torch.argmax(c_rouge, dim=1).to(c.device)
            ir1_loss = self.ir_loss(c_scores, best_sent_idx)
        else:
            ir1_loss = 0

        _, topk_idx_ir1 = torch.topk(c_scores, self.n_ctx1_sents, dim=1, sorted=False)
        
        ctx1 = [c[topk_idx_ir1[i]] for i in range(len(c_scores))]
        ctx1 = torch.stack(ctx1, dim=0)

        ctx1_chars = [c_chars[topk_idx_ir1[i]] for i in range(len(c_scores))]
        ctx1_chars = torch.stack(ctx1_chars, dim=0)

        ctx_len1 = [clen[topk_idx_ir1[i]] for i in range(len(c_scores))]        
        ctx_len1 = torch.stack(ctx_len1, dim=0)
        
        ir2_loss = 0
        for i in range(len(ctx1)):
            new_ctx = ctx1[i]
            new_ctx = [new_ctx[j, :ctx_len1[i,j]] for j in range(len(new_ctx))]
            new_ctx = torch.cat(new_ctx, dim=0) 

            new_ctx_chars = ctx1_chars[i]
            new_ctx_chars = [new_ctx_chars[j, :ctx_len1[i,j]] for j in range(len(new_ctx_chars))]
            new_ctx_chars = torch.cat(new_ctx_chars, dim=0) 

            if self.use_ir2 and new_ctx.shape[0] // self.chunk_size > self.n_ctx2_sents:                
                new_pwords = []
                for _idx in topk_idx_ir1[i]:
                    new_pwords += p_words[_idx]
                new_pwords = np.array(new_pwords)

                clip_idx = new_pwords.shape[0] % self.chunk_size
                if clip_idx > 0:
                    new_pwords = new_pwords[:-clip_idx]
                new_pwords = new_pwords.reshape(-1, self.chunk_size)                

                clip_idx = new_ctx.shape[0] % self.chunk_size                
                if clip_idx > 0:
                    new_ctx = new_ctx[:-clip_idx]
                    new_ctx_chars = new_ctx_chars[:-clip_idx]

                new_ctx_chars = new_ctx_chars.view(-1, self.chunk_size, new_ctx_chars.shape[1])    
                new_ctx = new_ctx.view(-1, self.chunk_size, new_ctx.shape[1])
                new_ctx_lens = torch.LongTensor([self.chunk_size]*new_ctx.shape[0]).to(ctx_len1.device)                          

                c_scores = self.ir_model2.forward_singleContext(q[[i]], new_ctx, qlen[[i]], new_ctx_lens,
                                                                batch_size=c_batch_size) 
                c_scores = c_scores.view(1,-1)
                c_scores = torch.log(F.gumbel_softmax(c_scores)).squeeze(0)                
                                
                _, topk_idx = torch.topk(c_scores, self.n_ctx2_sents, dim=0)  

                sents = new_ctx[topk_idx]
                sent_lens = new_ctx_lens[topk_idx]
                sents = [sents[j,:sent_lens[j]] for j in range(self.n_ctx2_sents)]

                sents_chars = new_ctx_chars[topk_idx]
                sents_chars = [sents_chars[j,:sent_lens[j]] for j in range(self.n_ctx2_sents)]

                ctx2 = torch.cat(sents, dim=0)
                ctx2_chars = torch.cat(sents_chars, dim=0)

                string_sent = []
                for _idx in topk_idx:
                    string_sent.append(new_pwords[_idx])
                string_sent = np.concatenate(string_sent, axis=0)

                if self.ir_model2.training:
                    p_words_rouge = self.getRouge(new_pwords, a1[i], a2[i])
                    best_sent_idx = torch.argmax(p_words_rouge, dim=0).to(c_scores.device)                    
                    ir2_loss += self.ir_loss(c_scores.unsqueeze(0), best_sent_idx.unsqueeze(0))
            else:
                ctx2 = new_ctx
                ctx2_chars = new_ctx_chars

                string_sent = []
                for _idx in topk_idx_ir1[i]:
                    string_sent.append(p_words[_idx])
                string_sent = np.concatenate(string_sent, axis=0)                

            selected_sents.append(ctx2)
            selected_sents_chars.append(ctx2_chars)

            
            #string_sent = [w for s in string_sent for w in s]            
            string_sents.append(string_sent)

        ctx_len = torch.tensor([len(s) for s in selected_sents]).long().to(c.device)
        max_ctx_len = max(ctx_len)

        ctx = torch.zeros(len(selected_sents), max_ctx_len, c.shape[2]).long().to(c.device)
        ctx_chars = torch.zeros(len(selected_sents_chars), max_ctx_len, c_chars.shape[2]).long().to(c.device)     
        for (i, sents) in enumerate(selected_sents):
            ctx[i,:len(sents)] = sents
            ctx_chars[i,:len(sents)] = selected_sents_chars[i]

        ir2_loss /= q.shape[0]
        return ctx, ctx_chars, ctx_len, string_sents, ir1_loss, ir2_loss

    def forward(self, q, q_chars, c, c_chars, c_rouge, avec1, avec2, qlen, clen, alen, p_words, a1, a2, c_batch_size=512):        
        ctx, c_chars, ctx_len, string_sents, ir1_loss, ir2_loss = self.getSpans(q, c, c_chars, qlen, clen, p_words, c_rouge=c_rouge, a1=a1, a2=a2, c_batch_size=c_batch_size)

        # max_word_len = 16
        # words2charIds = lambda W: [[self.c2i.get(w[i], self.c2i['<unk>']) if i < len(w) else self.c2i['<pad>'] for i in range(max_word_len)] for w in W]
        # c_chars = [words2charIds(sent) for sent in string_sents]
        # c_chars = torch.LongTensor(c_chars).to(c.device)
        # print(c_chars.shape)

        # print(ctx.shape, q.shape) # batch, seq_len, [word_index, pos_index]
        loss1, sidx, eidx = self.rc_model(ctx[:,:,0], ctx[:,:,1], ctx[:,:,2], c_chars, ctx_len, 
                                                q[:,:,0], q[:,:,1], q[:,:,2], q_chars, qlen, 
                                                string_sents, avec1, alen, a1, a2)

        # print (loss1)
        return loss1, ir1_loss, ir2_loss, sidx, eidx
        
        # print (sidx, eidx)
        # raw_span = []
        # for i in range(len(string_sents)):
        #     print("*************")
        #     print(len(string_sents[i]))
        #     print( sidx[i] , eidx[i], eidx[i] - sidx[i])
        #     print(len(string_sents[i][sidx[i]:eidx[i]+1]))
        #     raw_span.append(['<sos>'] + string_sents[i][sidx[i]:eidx[i]+1] + ['<eos>'])
        #     print (len(raw_span[-1]))

        # gen_loss, output = self.ag_model(extracted_span, avec2, raw_span)

        # return loss1+gen_loss

    def evaluate(self, q, q_chars, c, c_chars, qlen, clen, p_words, c_batch_size=512):
        ctx, c_chars, ctx_len, string_sents, _, _ = self.getSpans(q, c, c_chars, qlen, clen, 
                                                            p_words, a1=None, a2=None, c_batch_size=c_batch_size)
        sidx, eidx = self.rc_model.evaluate(ctx[:,:,0], ctx[:,:,1], ctx[:,:,2], c_chars, ctx_len, 
                                                q[:,:,0], q[:,:,1], q[:,:,2], q_chars, qlen)
        # raw_span = []
        # for i in range(len(string_sents)):
        #     raw_span.append(['<sos>'] + string_sents[i][sidx[i]:eidx[i]+1] + ['<eos>'])

        # gen_loss, output = self.ag_model(extracted_span, avec2, raw_span)
        return sidx, eidx, string_sents
        # return gen_loss

        # # print(type(ctx_len), type(qlen))
        # loss1, loss2, sidx, eidx = self.rc_model(ctx[:,:,0], ctx[:,:,1], ctx_len, 
        #                                       q[:,:,0], q[:,:,1], qlen, 
        #                                       None, a, alen)
        # return loss1
