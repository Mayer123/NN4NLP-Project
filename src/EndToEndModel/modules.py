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
                    n_ctx1_sents=6, n_ctx2_sents=5, span_size=15, use_ir2=False,
                    ir_pretrain=False, rc_pretrain=False):
        super(EndToEndModel, self).__init__()
        self.ir_model1 = ir_model1
        self.ir_model2 = ir_model2
        self.rc_model = rc_model
        self.ag_model = ag_model
        self.n_ctx1_sents = n_ctx1_sents
        self.n_ctx2_sents = n_ctx2_sents
        self.chunk_size = span_size
        self.i2w = {v:k for k,v in w2i.items()}
        self.c2i = c2i
        self.ir_loss = nn.NLLLoss()
        self.rrrouge = RRRouge()
        self.use_ir2 = use_ir2
        self.ir_pretrain = ir_pretrain
        self.rc_pretrain = rc_pretrain

    def idx2words(self, idxs):
        return [self.i2w[i.data] for i in idxs]

    def getRouge(self, spans, a1, a2):        
        return torch.Tensor([self.rrrouge.calc_score([" ".join(span)], [a1, a2]) for span in spans])

    def getSpans(self, q, c, c_chars, qlen, clen, p_words, bsi=None, bss=None, bslen=None, c_rouge=None, a1=None, a2=None, 
                    c_batch_size=512):
        # print(q.shape, c.shape, a.shape)
        selected_sents = []
        selected_sents_chars = []
        string_sents = []                    

        if self.rc_pretrain:            
            topk_idx_ir1 = [bsi[i, :min(self.n_ctx1_sents, bslen[i]), 0] for i in range(q.shape[0])]
            # print(topk_idx_ir1)
            topk_idx_ir1 = torch.stack(topk_idx_ir1, dim=0).to(c.device)
        else:
            c_scores = self.ir_model1.forward_singleContext(q, c, qlen, clen,
                                                    batch_size=c_batch_size)        
            if not torch.isfinite(c_scores).all():
                print('bad c_scores')
                print(c_scores)        
            c_scores = torch.log(F.gumbel_softmax(c_scores))
            _, topk_idx_ir1 = torch.topk(c_scores, self.n_ctx1_sents, 
                                        dim=1, sorted=False)

        ir1_loss = 0
        misses = 0
        if self.ir_model1.training:
            # max_scores, best_sent_idxs = torch.max(c_rouge, dim=1)
            # best_sent_idxs = best_sent_idxs.cuda()
            # ir1_loss = self.ir_loss(c_scores, best_sent_idxs)
            if bsi is None:
                _, topk_rouge_idx = torch.topk(c_rouge, self.n_ctx1_sents, dim=1)
                best_start_idx = 0
                best_end_idx = 0
            else:
                best_start_idx = []
                best_end_idx = []

                topk_rouge_idx = [bsi[i, :min(self.n_ctx1_sents, bslen[i]), 0] for i in range(q.shape[0])]

                for i in range(len(topk_idx_ir1)):
                    top_selected_idx = [k for k,j in enumerate(topk_rouge_idx[i]) if j in topk_idx_ir1[i]]
                    if top_selected_idx == []:
                        worst_idx = torch.argmin(bss[i], dim=0)
                        topk_idx_ir1[i][-1] = bsi[i, worst_idx, 0]
                        top_selected_idx.append(worst_idx)
                        misses += 1

                                        
                    best_span_idx = torch.argmax(bss[i, top_selected_idx], dim=0)
                    best_span_idx = top_selected_idx[best_span_idx]                
                    prev_chunk_idx = topk_idx_ir1[i, :torch.nonzero(topk_idx_ir1[i] == bsi[i, best_span_idx, 0]).view(-1)[0]]
                    # print(best_span_idx, bsi[i, best_span_idx], bss[i, best_span_idx], topk_idx_ir1[i], prev_chunk_idx)

                    # span = c[bsi[i, best_span_idx, 0], bsi[i, best_span_idx, 1]:bsi[i, best_span_idx, 2]+1, 0]
                    # print(bsi[i, best_span_idx], span)
                    # print([self.i2w[int(x)] for x in span])

                    best_start_idx.append(bsi[i, best_span_idx, 1] + torch.sum(clen[prev_chunk_idx]))
                    best_end_idx.append(bsi[i, best_span_idx, 2] + torch.sum(clen[prev_chunk_idx]))
                
                best_start_idx = torch.LongTensor(best_start_idx).to(q.device)
                best_end_idx = torch.LongTensor(best_end_idx).to(q.device)                      

            rest_rouge_idx = [[j for j in range(c.shape[0]) if j not in topk_rouge_idx[i]] for i in range(q.shape[0])]  

            if not self.rc_pretrain:          
                best_score = [c_scores[i, topk_rouge_idx[i]] for i in range(len(topk_rouge_idx))]
                rest_score = [c_scores[i, rest_rouge_idx[i]] for i in range(len(rest_rouge_idx))]

                for i in range(len(topk_rouge_idx)):                   
                    _rest_score = rest_score[i].view(1,-1)
                    _best_score = best_score[i].view(-1,1).expand(-1, _rest_score.shape[1])
                    diff = F.relu(1 - (_best_score - _rest_score))
                    ir1_loss += torch.mean(diff)
                ir1_loss /= len(topk_rouge_idx)
        
        ctx1 = [c[topk_idx_ir1[i]] for i in range(len(topk_idx_ir1))]
        ctx1 = torch.stack(ctx1, dim=0)        

        ctx1_chars = [c_chars[topk_idx_ir1[i]] for i in range(len(topk_idx_ir1))]
        ctx1_chars = torch.stack(ctx1_chars, dim=0)

        ctx_len1 = [clen[topk_idx_ir1[i]] for i in range(len(topk_idx_ir1))]        
        ctx_len1 = torch.stack(ctx_len1, dim=0)        
        
        ir2_loss = 0
        for i in range(len(ctx1)):
            new_ctx = ctx1[i]
            new_ctx = [new_ctx[j, :ctx_len1[i,j]] for j in range(len(new_ctx))]
            new_ctx = torch.cat(new_ctx, dim=0)             
            # print([self.i2w[int(j)] for j in new_ctx[best_start_idx[i]:best_end_idx[i]+1,0]])

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
                if self.ir_model2.training:
                    p_words_rouge = self.getRouge(new_pwords, a1[i], a2[i])                    
                    # best_sent_idx = torch.argmax(p_words_rouge, dim=0).to(c_scores.device)                    
                    # ir2_loss += self.ir_loss(c_scores.unsqueeze(0), best_sent_idx.unsqueeze(0))                   

                    _, topk_rouge_idx = torch.topk(p_words_rouge, self.n_ctx2_sents, dim=0)
                    rest_rouge_idx = [j for j in range(p_words_rouge.shape[0]) if j not in topk_rouge_idx]
                    
                    best_score = c_scores[topk_rouge_idx]
                    rest_score = c_scores[rest_rouge_idx]

                    _rest_score = rest_score.view(1,-1)
                    _best_score = best_score.view(-1,1).expand(-1, _rest_score.shape[1])                                       
                    diff = F.relu(1 - (_best_score - _rest_score))
                    ir2_loss += torch.mean(diff)

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

        ctx = nn.utils.rnn.pad_sequence(selected_sents, batch_first=True)
        ctx_chars = nn.utils.rnn.pad_sequence(selected_sents_chars, batch_first=True)

        # ctx = torch.zeros(len(selected_sents), max_ctx_len, c.shape[2]).long().to(c.device)
        # ctx_chars = torch.zeros(len(selected_sents_chars), max_ctx_len, c_chars.shape[2]).long().to(c.device)     
        # for (i, sents) in enumerate(selected_sents):
        #     ctx[i,:len(sents)] = sents
        #     ctx_chars[i,:len(sents)] = selected_sents_chars[i]

        ir2_loss /= q.shape[0]
        if self.ir_model1.training:
            # for i in range(len(string_sents)):
                # print(best_start_idx[i], best_end_idx[i])
                # print(string_sents[i][best_start_idx[i]:best_end_idx[i]+1])
                # print([self.i2w[int(j)] for j in ctx[i, best_start_idx[i]:best_end_idx[i]+1,0]])
            return ctx, ctx_chars, ctx_len, string_sents, ir1_loss, ir2_loss, best_start_idx, best_end_idx, misses/q.shape[0]
        return ctx, ctx_chars, ctx_len, string_sents

    def forward(self, q, q_chars, c, c_chars, c_rouge, avec1, avec2, qlen, clen, alen, p_words, a1, a2, bsi=None, bss=None, bslen=None, c_batch_size=512):        
        ctx, c_chars, ctx_len, string_sents, ir1_loss, ir2_loss, best_start_idx, best_end_idx, miss_rate = self.getSpans(q, c, c_chars, qlen, clen, p_words, 
                                                                                c_rouge=c_rouge, a1=a1, a2=a2, bsi=bsi, bss=bss,
                                                                                bslen=bslen, c_batch_size=c_batch_size)

        # max_word_len = 16
        # words2charIds = lambda W: [[self.c2i.get(w[i], self.c2i['<unk>']) if i < len(w) else self.c2i['<pad>'] for i in range(max_word_len)] for w in W]
        # c_chars = [words2charIds(sent) for sent in string_sents]
        # c_chars = torch.LongTensor(c_chars).to(c.device)
        # print(c_chars.shape)

        if self.ir_pretrain:
            return ir1_loss, ir1_loss, ir2_loss, miss_rate, None, None
        
        loss1, sidx, eidx = self.rc_model(ctx[:,:,0], ctx[:,:,1], ctx[:,:,2], c_chars, ctx_len, 
                                                q[:,:,0], q[:,:,1], q[:,:,2], q_chars, qlen, 
                                                string_sents, avec1, alen, a1, a2, best_start_idx, 
                                                best_end_idx)

        # print (loss1)
        return loss1, ir1_loss, ir2_loss, miss_rate, sidx, eidx
        
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

    def evaluate(self, q, q_chars, c, c_chars, qlen, clen, p_words, bsi=None, bss=None, bslen=None, c_batch_size=512):
        ctx, c_chars, ctx_len, string_sents = self.getSpans(q, c, c_chars, qlen, clen, 
                                                            p_words, a1=None, a2=None, bsi=bsi, bss=bss,
                                                            bslen=bslen, c_batch_size=c_batch_size)

        if self.ir_pretrain:
            return None, None, string_sents

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
