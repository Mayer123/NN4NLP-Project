import sys
sys.path.append('../../')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from ReadingComprehension.IterativeReattentionAligner.modules import IterativeAligner
from ReadingComprehension.IterativeReattentionAligner.loss import DCRLLoss
from ReadingComprehension.IterativeReattentionAligner.decoder import Decoder
from utils.utils import output_mask

class GaussianKernel(object):
    """docstring for GaussianKernel"""
    def __init__(self, mean, std):
        super(GaussianKernel, self).__init__()
        self.mean = mean
        self.std = std

    def __call__(self, x):
        sim = torch.exp(-0.5 * (x-self.mean)**2 / self.std**2)      
        return sim

class WordOverlapLoss(nn.Module):
    """docstring for StringOverlapLoss"""
    def __init__(self):
        super(WordOverlapLoss, self).__init__()
        self.G = GaussianKernel(1.0, 0.001)

    def forward(self, s1, s2, s1_len, s2_len):
        # s1.shape = (batch, seq_len, emb_dim)
        # s1 should be the true answer

        normed_s1 = s1 / torch.norm(s1, dim=2, keepdim=True)
        normed_s2 = s2 / torch.norm(s2, dim=2, keepdim=True)

        cosine = torch.bmm(normed_s1, normed_s2.transpose(1,2)) # (batch, len(s1), len(s2))
        cosine_em = self.G(cosine)
        cosine_sm = cosine_em/(torch.sum(cosine_em, dim=2, keepdim=True) + 1e-10)
        cosine_scaled = cosine_em * cosine_sm

        mask = output_mask(s1_len)
        max_match = torch.sum(cosine_scaled, dim=2)

        max_match = max_match * mask
        tot_match = torch.sum(max_match, dim=1)
        ovrl = tot_match/s1_len.float()

        loss = 1 - ovrl
        return loss

class MnemicReader(nn.Module):
    ## model(c_vec, c_pos, c_ner, c_em, c_mask, q_vec, q_pos, q_ner, q_em, q_mask, start, end)
    def __init__(self, input_size, hidden_size, num_layers, pos_emb_dim, word_embeddings,
                    num_pos, vocab_size, emb_dropout=0.0, rnn_dropout=0.0):
        super(MnemicReader, self).__init__()
        self.num_layers = num_layers
        self.rnn = nn.ModuleList()

        self.pos_emb = nn.Sequential(
            nn.Embedding(num_pos, pos_emb_dim, padding_idx=0),
            nn.Dropout(emb_dropout)
            )
        self.vocab_size = vocab_size
        self.emb_size = word_embeddings.shape[1]
        self.word_embeddings = torch.nn.Embedding(word_embeddings.shape[0], word_embeddings.shape[1], 
                                                    padding_idx=0)
        self.word_embeddings.weight.data.copy_(word_embeddings)
        self.word_embeddings = nn.Sequential(
            self.word_embeddings,
            nn.Dropout(emb_dropout)
            )
        #self.answerPointerModel = answerPointerModel()    
        self.aligningBlock = IterativeAligner( 2 * hidden_size, hidden_size, 1, 3, dropout=rnn_dropout)

        self.loss = WordOverlapLoss()
        self.DCRL_loss = DCRLLoss(4)
        #self.weight_a = torch.pow(torch.randn(1, requires_grad=True), 2)
        #self.weight_b = torch.pow(torch.randn(1, requires_grad=True), 2)
        self.weight_a = torch.randn(1, requires_grad=True)
        self.weight_b = torch.randn(1, requires_grad=True)
        self.half = torch.tensor(0.5)

        self.rnn_dropout = nn.Dropout(rnn_dropout)

        for i in range(num_layers):
            lstm = nn.LSTM(input_size, hidden_size, num_layers=1, bidirectional=True)
            self.rnn.append(lstm)

        self.use_RLLoss = False
        # self.generative_decoder = Decoder(self.emb_size, hidden_size, self.word_embeddings, self.emb_size, self.vocab_size, 15, 0.4)
        # self.gen_loss = nn.NLLLoss(ignore_index=0)
        # self.fc_in = nn.Linear(word_embeddings.shape[1], hidden_size*2)

    def prepare_decoder_input(self, s_index, e_index, context):
        batch_size, seq_len = context.shape        
        cut = e_index - s_index
        max_len = torch.max(cut)
        max_len += 3
        decoder_input = torch.zeros(batch_size, max_len).to(context.device).long()
        for i in range(batch_size):
            decoder_input[i,0] = torch.Tensor([2]).long().to(context.device)
            decoder_input[i,1:e_index[i]-s_index[i]+2] = context[i,s_index[i]:e_index[i]+1]
            decoder_input[i,e_index[i]-s_index[i]+2] = torch.Tensor([3]).long().to(context.device)
        return decoder_input

    def char_lstm_forward(self, char, char_lens):
        # c_char is 3d (batch, words, chars)        
        char_squeezed = char.view(-1, char.size()[2])        
        char_e = self.char_emb(char_squeezed)

        # char_len_squeezed = char_lens.view(-1,)
        # char_len_squeezed, sorted_idx = torch.sort(char_len_squeezed, descending=True)
        # _,rev_sorted_idx = torch.sort(sorted_idx)
        
        # char_e = char_e[sorted_idx]
        char_e = char_e.transpose(0,1)        
        
        # char_e = nn.utils.rnn.pack_padded_sequence(char_e, char_len_squeezed)       
        con_char_lstm = self.char_lstm(char_e)[1][0]
        
        # con_char_lstm = con_char_lstm[:,rev_sorted_idx,:]        
        con_char_lstm = torch.cat((con_char_lstm[0,:,:],con_char_lstm[1,:,:]), 1)        
        con_char = con_char_lstm.view(char.size()[0], char.size()[1], -1)

        return con_char

    def getAnswerSpanProbs(self, c_vec, c_pos, c_lens, q_vec, q_pos, q_lens):
        con_vec = self.word_embeddings(c_vec)
        con_pos = self.pos_emb(c_pos)     

        que_vec = self.word_embeddings(q_vec)
        que_pos = self.pos_emb(q_pos)

        con_input = torch.cat([con_vec, con_pos], 2)
        que_input = torch.cat([que_vec, que_pos], 2)
        x1 = con_input.transpose(0, 1)
        # x1_len, x1_sorted_idx = torch.sort(torch.sum(c_mask, dim=1), descending=True)
        # _, x1_rev_sorted_idx = torch.sort(x1_sorted_idx)
        # packed_x1 = nn.utils.rnn.pack_padded_sequence(x1[:,x1_sorted_idx,:], x1_len)

        x2 = que_input.transpose(0, 1)
        # x2_len, x2_sorted_idx = torch.sort(torch.sum(q_mask, dim=1), descending=True)
        # _, x2_rev_sorted_idx = torch.sort(x2_sorted_idx)
        # packed_x2 = nn.utils.rnn.pack_padded_sequence(x2[:,x2_sorted_idx,:], x2_len)
        

        enc_con = []
        enc_que = []
        for i in range(self.num_layers):
            x1 = self.rnn[i](x1)[0]
            # packed_x1 = self.rnn[i](packed_x1)[0]
            # x1, x1_len = nn.utils.rnn.pad_packed_sequence(packed_x1)
            x1 = self.rnn_dropout(x1)      
            enc_con.append(x1)      
            # enc_con.append(x1[:,x1_rev_sorted_idx,:])            

            x2 = self.rnn[i](x2)[0]
            # packed_x2 = self.rnn[i](packed_x2)[0]
            # x2, x2_len = nn.utils.rnn.pad_packed_sequence(packed_x2)
            x2 = self.rnn_dropout(x2)
            enc_que.append(x2)
            # enc_que.append(x2[:,x2_rev_sorted_idx,:])
            
            # if i < self.num_layers -1:
            #     packed_x1 = nn.utils.rnn.pack_padded_sequence(x1, x1_len)
            #     packed_x2 = nn.utils.rnn.pack_padded_sequence(x2, x2_len)

        enc_con = torch.cat(enc_con, 2).transpose(0, 1) # (batch_size, seq_len, enc_con_dim)
        enc_que = torch.cat(enc_que, 2).transpose(0, 1) # (batch_size, seq_len, enc_que_dim)            
        # print(enc_con.shape, enc_que.shape)
        s_prob, e_prob, probs, final_context = self.aligningBlock(enc_con, enc_que, u_lens=c_lens,  v_lens=q_lens)
        
        return s_prob, e_prob, probs, final_context

    def forward(self, c_vec, c_pos, c_lens, q_vec, q_pos, q_lens, context, a_vec, alen):
        s_prob, e_prob, probs, final_context = self.getAnswerSpanProbs(c_vec, c_pos,c_lens, 
                                                                        q_vec, q_pos, q_lens)
        #print (s_prob.shape, e_prob.shape)
        #print (start, end)
        #print (torch.gather(s_prob.squeeze(), 1, start.unsqueeze(1)))
        #print (start)
        #print (torch.gather(e_prob.squeeze(), 1, end.unsqueeze(1)))
        #print (end)
        #s_prob = torch.log(s_prob)
        #e_prob = torch.log(e_prob)
        context_len = s_prob.shape[1]
        max_idx = torch.argmax(probs, dim=1)
        start = max_idx // context_len
        end = max_idx % context_len

        pred_a = [c_vec[i, start[i]:end[i]] for i in range(len(start))]
        
        pred_a_len = [len(a) for a in pred_a]
        padded_pred_a = torch.zeros(len(pred_a), max(pred_a_len), 
                                   dtype=c_vec.dtype).to(c_vec.device)
        for (i,x) in enumerate(pred_a):
           padded_pred_a[i,:pred_a_len[i]] = x
            
        pred_a_emb = self.word_embeddings(padded_pred_a)
        decoder_input = self.prepare_decoder_input(start, end, c_vec)
        a_emb = self.word_embeddings(a_vec)

        loss = self.loss(a_emb, pred_a_emb, alen, start-end)
        loss = torch.mean(loss)

        if not self.use_RLLoss:
            return loss, loss, start, end, decoder_input

        #return loss
        #s_prob = torch.exp(s_prob)
        #e_prob = torch.exp(e_prob)
        # loss1 = self.loss(s_prob, start)
        # loss2 = self.loss(e_prob, end)

        #s_prob = torch.nn.functional.softmax(s_prob, dim=1)
        #e_prob = torch.nn.functional.softmax(e_prob, dim=1)
        
        #s_prob = s_prob * c_mask.float()
        #e_prob = e_prob * c_mask.float()

        probs = torch.exp(probs)
        rl_loss = self.DCRL_loss(probs, s_prob, e_prob, context_len, start, end, context, a1, a2)
        #_, s_index = torch.max(torch.squeeze(s_prob), dim=1)
        #_, e_index = torch.max(torch.squeeze(e_prob), dim=1)
        #print (loss1, loss2)
        #loss = (start - s_index)**2 + (end - e_index)**2
        #loss = (loss1+loss2)*self.weight_a.pow(-1)*self.half+rl_loss*self.weight_b.pow(-1)*self.half+torch.log(self.weight_a)+torch.log(self.weight_b)
        #return loss
        self.weight_a = self.weight_a.to(rl_loss.device)
        self.weight_b = self.weight_b.to(rl_loss.device)
        self.half = self.half.to(rl_loss.device)
        a1 = torch.pow(self.weight_a, -2) * self.half
        a2 = torch.pow(self.weight_b, -2) * self.half
        b1 = torch.log(torch.pow(self.weight_a, 2))
        b2 = torch.log(torch.pow(self.weight_b, 2))
        #total_loss = (loss1+loss2)*self.weight_a+rl_loss*self.weight_b
        return loss * a1 + rl_loss * a2 + b1 + b2, loss, s_index, e_index

    def evaluate(self, c_vec, c_pos, c_mask, q_vec, q_pos, q_mask):
        s_prob, e_prob, probs, final_context = self.getAnswerSpanProbs(c_vec, c_pos, c_mask, q_vec, 
                                                                        q_pos, q_mask)

        #s_prob = torch.squeeze(s_prob)
        #e_prob = torch.squeeze(e_prob)

        context_len = s_prob.shape[1]
        max_idx = torch.argmax(probs, dim=1)
        s_index = max_idx // context_len
        e_index = max_idx % context_len
        decoder_input = self.prepare_decoder_input(s_index, e_index, c_vec)
        # decode_input = self.prepare_decoder_input(s_index, e_index, con_vec)
        # generate_output = self.generative_decoder.generate(decode_input)
        return s_index, e_index, decoder_input

if __name__ == '__main__':
    loss = WordOverlapLoss()
    s1 = torch.tensor([
            [
                [1, 2],
                [3, 4],
                [7, 8],
                [9, 0.],
            ],
            [
                [1, 2],
                [3, 4],
                [5, 6],
                [5, 6],
            ]
        ], requires_grad=False)

    s2 = torch.tensor([
            [
                [1, 2],
                [3, 4.09],
                [5, 6],
                [5, 6],
            ],
            [
                [1, 2],
                [3, 4],
                [5, 6],
                [5, 6],
            ]
        ], requires_grad=True)
    # w = nn.Linear(s1.shape[-1], s1.shape[-1], bias=False)
    # s1 = w(s1)
    # s2 = w(s2)

    l = loss(s1.float(), s2.float(), torch.tensor([4,2]), torch.tensor([2,2]))
    print(l)
    l = torch.mean(l)
    l.backward()
    print(s2.grad)
    # seq_len = 60
    # seq_len2 = 40
    # batch = 2

    # input_size = 100
    # vocab_size = 200

    # char_embedding_dim = 300
    # word_embedding_dim = 100

    # char_hidden_size = 200
    # encoder_hidden_size = 200

    # encoder_input_dim = word_embedding_dim + (2 * char_embedding_dim) 

    # word_embeddings = np.random.uniform(-0.25, 0.25, (vocab_size, word_embedding_dim))
    # char_embeddings = np.random.uniform(-0.25, 0.25, (vocab_size, char_embedding_dim))
   
    # encoder_input_dim = 64 + 64 + word_embedding_dim + (2*10) + 1
    #           #pos_emb_dim + ner_emb_dim + word_embeddings.shape[1] + (2*hidden_size) + 1
 
    # mnemonicReader = MnemicReader(input_size=encoder_input_dim, hidden_size=10, num_layers=1, char_emb_dim =64, \
    #         pos_emb_dim=64, ner_emb_dim=64, word_embeddings=word_embeddings, num_char=40, num_pos=40, num_ner=40)

    # c_vec = torch.Tensor(np.random.randint(40, size=(batch, seq_len))).long()
    # c_pos = torch.Tensor(np.random.randint(40, size=(batch, seq_len))).long()
    # c_ner = torch.Tensor(np.random.randint(40, size=(batch, seq_len))).long()
    # c_em = torch.Tensor(np.random.randint(2, size=(batch, seq_len))).float()

    # c_char = torch.Tensor(np.random.randint(40, size=(batch, seq_len, 10))).long()
    # c_mask = torch.Tensor(np.random.randint(40, size=(batch, seq_len))).long()
    
    # q_vec = torch.Tensor(np.random.randint(40, size=(batch, seq_len2))).long()
    # q_pos = torch.Tensor(np.random.randint(40, size=(batch, seq_len2))).long()
    # q_ner = torch.Tensor(np.random.randint(40, size=(batch, seq_len2))).long()
    # q_em = torch.Tensor(np.random.randint(2, size=(batch, seq_len2))).float()

    # q_char = torch.Tensor(np.random.randint(40, size=(batch, seq_len2, 10))).long()
    # q_mask = torch.Tensor(np.random.randint(40, size=(batch, seq_len2))).long()

    # mnemonicReader(c_vec, c_pos, c_ner, c_char, c_em, c_mask, q_vec, q_pos, q_ner, q_char, q_em, q_mask, start=0, end=4)