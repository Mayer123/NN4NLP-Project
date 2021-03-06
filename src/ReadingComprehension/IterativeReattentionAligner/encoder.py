import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from ReadingComprehension.IterativeReattentionAligner.modules import IterativeAligner
from ReadingComprehension.IterativeReattentionAligner.loss import DCRLLoss
from ReadingComprehension.IterativeReattentionAligner.decoder import Decoder
from AnswerGenerator.decoder import GRUDecoder, GRUEncoder, seq2seqAG

class MnemicReader(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, char_emb_dim, 
                    pos_emb_dim, ner_emb_dim, word_embeddings, num_char, 
                    num_pos, num_ner, vocab_size, emb_dropout=0.0, rnn_dropout=0.0,
                    use_generator=False):
        super(MnemicReader, self).__init__()
        self.num_layers = num_layers
        self.rnn = nn.ModuleList()            

        self.char_emb = nn.Sequential(
            nn.Embedding(num_char, char_emb_dim, padding_idx=0),
            nn.Dropout(emb_dropout)
            )
        self.pos_emb = nn.Sequential(
            nn.Embedding(num_pos, pos_emb_dim, padding_idx=0),
            nn.Dropout(emb_dropout)
            )
        self.ner_emb = nn.Sequential(
            nn.Embedding(num_ner, ner_emb_dim, padding_idx=0),
            nn.Dropout(emb_dropout)
            )
        self.vocab_size = vocab_size
        self.emb_size = word_embeddings.shape[1]
        self.full_vocab = word_embeddings.shape[0]
        self.word_embeddings = torch.nn.Embedding(word_embeddings.shape[0], word_embeddings.shape[1], 
                                                    padding_idx=0)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(word_embeddings))
        self.word_embeddings = nn.Sequential(
            self.word_embeddings,
            nn.Dropout(emb_dropout)
            )

        self.char_lstm = nn.LSTM(char_emb_dim, char_emb_dim, num_layers=1, bidirectional=True)
        self.aligningBlock = IterativeAligner( 2 * hidden_size, hidden_size, 1, 3, dropout=rnn_dropout)

        self.loss = nn.NLLLoss()
        self.DCRL_loss = DCRLLoss(4)

        self.weight_a = torch.randn(1, requires_grad=True)
        self.weight_b = torch.randn(1, requires_grad=True)
        self.half = torch.tensor(0.5)

        self.rnn_dropout = nn.Dropout(rnn_dropout)

        for i in range(num_layers):
            lstm = nn.LSTM(input_size, hidden_size, num_layers=1, bidirectional=True)
            self.rnn.append(lstm)

        self.use_RLLoss = False

        print(hidden_size)        
        self.use_generator = use_generator
        if self.use_generator:
            # self.decoder = GRUDecoder(word_embeddings.shape[1], 2*hidden_size, 1, word_embeddings.shape[0], 
            #                             word_embeddings=self.word_embeddings, use_attention=False)
            encoder = GRUEncoder(2*word_embeddings.shape[1], 1024, 1, bidirectional=False)
            decoder = GRUDecoder(word_embeddings.shape[1], 1024, 1, word_embeddings.shape[0], 
                                        word_embeddings=self.word_embeddings, use_attention=True,
                                        tf_rate=0.9)
            self.answer_generator = seq2seqAG(encoder, decoder)

            self.decoder_loss = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
        # self.generative_decoder = Decoder(self.emb_size, hidden_size, self.word_embeddings, self.emb_size, self.vocab_size, 15, 0.4)
        # self.gen_loss = nn.NLLLoss(ignore_index=0)
        # self.fc_in = nn.Linear(word_embeddings.shape[1], hidden_size*2)
        #self.word_loss = WordOverlapLoss()

    def prepare_decoder_input(self, s_index, e_index, context):
        batch_size, seq_len, hidden_size = context.shape        
        cut = e_index - s_index
        max_len = torch.max(cut)
        max_len += 3
        decoder_input = torch.zeros(batch_size, max_len, hidden_size).to(context.device)
        for i in range(batch_size):
            decoder_input[i,0,:] = self.word_embeddings(torch.Tensor([2]).long().to(context.device))
            decoder_input[i,1:e_index[i]-s_index[i]+2,:] = context[i,s_index[i]:e_index[i]+1,:]
            decoder_input[i,e_index[i]-s_index[i]+2,:] = self.word_embeddings(torch.Tensor([3]).long().to(context.device))
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

    def getAnswerSpanProbs(self, c_vec, c_pos, c_ner, c_char, c_em, c_char_lens, c_mask, q_vec, q_pos, q_ner, q_char, q_em, q_char_lens, q_mask):
        con_vec = self.word_embeddings(c_vec)
        con_pos = self.pos_emb(c_pos)
        con_ner = self.ner_emb(c_ner) 

        
        con_char = self.char_lstm_forward(c_char, c_char_lens)
        con_char *= c_mask.unsqueeze(2).float()        

        que_vec = self.word_embeddings(q_vec)
        que_pos = self.pos_emb(q_pos)
        que_ner = self.ner_emb(q_ner)

        que_char = self.char_lstm_forward(q_char, q_char_lens)
        que_char *= q_mask.unsqueeze(2).float()

        con_input = torch.cat([con_vec, con_char, con_pos, con_ner, c_em.unsqueeze(2)], 2)
        que_input = torch.cat([que_vec, que_char, que_pos, que_ner, q_em.unsqueeze(2)], 2)
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
        s_prob, e_prob, probs, final_context, hidden_R = self.aligningBlock(enc_con, enc_que, c_mask.float(),  q_mask.float())
        
        return s_prob, e_prob, probs, final_context, hidden_R

    def getDecoderLoss(self, pred_logits, a_vec):
        a_vec = a_vec.contiguous().view(-1)
        pred_logits = pred_logits.contiguous().view(-1, pred_logits.shape[2])        
        decoder_loss = self.decoder_loss(pred_logits, a_vec)
        return decoder_loss

    def extractSpan(self, context, s_index, e_index):
        ctx_span_len = e_index - s_index
        ctx_span = [context[i, s_index[i]:e_index[i]] for i in range(context.shape[0])]        
        ctx_span = nn.utils.rnn.pad_sequence(ctx_span, batch_first=True) 
        return ctx_span, ctx_span_len

    def getTopKSpans(self, context, probs, k):
        context_len = context.shape[1]
        _, top_spans = torch.topk(probs, 4, dim=1)
        ctx_span = []
        for i in range(top_spans.shape[0]):
            si = top_spans[i] // context_len
            ei = top_spans[i] % context_len
            span = [context[i, si[j] : ei[j]].detach() for j in range(len(si))]
            span = torch.cat(span, dim=0)
            ctx_span.append(span)

        ctx_span_len = torch.tensor([s.shape[0] for s in ctx_span]).long().to(context.device)
        ctx_span = nn.utils.rnn.pad_sequence(ctx_span, batch_first=True)        
        return ctx_span, ctx_span_len

    def getPredsFromGenerator(self, hidden_R, ctx_span, ctx_span_len, a_vec):        
        hidden_R = hidden_R.transpose(0,1)
        hidden_R = hidden_R.contiguous().view(hidden_R.shape[0], -1)       

        # pred_logits = self.decoder(ctx_span, ctx_span_len, a_vec, 2, 
        #                             gumbel=True, initial_hidden=hidden_R)
        pred_logits = self.answer_generator(ctx_span, ctx_span_len, a_vec, 2)
        pred_logits = pred_logits.transpose(0,1)

        _, pred_ans = torch.max(pred_logits, dim=2)        

        return pred_logits, pred_ans

    def forward(self, c_vec, c_pos, c_ner, c_char, c_em, c_char_lens, c_mask, q_vec, q_pos, q_ner, q_char, q_em, q_char_lens, q_mask, start, end, context, a1, a2, a_vec):        
        s_prob, e_prob, probs, final_context, hidden_R = self.getAnswerSpanProbs(c_vec, c_pos, c_ner, c_char, 
                                                                        c_em, c_char_lens, c_mask, q_vec, 
                                                                        q_pos, q_ner, q_char, q_em, 
                                                                        q_char_lens, q_mask)
        #print (s_prob.shape, e_prob.shape)
        #print (start, end)
        #print (torch.gather(s_prob.squeeze(), 1, start.unsqueeze(1)))
        #print (start)
        #print (torch.gather(e_prob.squeeze(), 1, end.unsqueeze(1)))
        #print (end)
        #s_prob = torch.log(s_prob)
        #e_prob = torch.log(e_prob)
        s_prob = torch.squeeze(s_prob)
        e_prob = torch.squeeze(e_prob)
        #s_prob = torch.log(s_prob)
        #e_prob = torch.log(e_prob)
        loss1 = self.loss(torch.log(s_prob), start)
        loss2 = self.loss(torch.log(e_prob), end)
        loss = loss1 + loss2

        context_len = final_context.shape[1]
        
        if not torch.isfinite(probs).all():
            print('bad probs')
            return

        s_index = torch.argmax(s_prob, dim=1)
        e_index = torch.argmax(e_prob, dim=1)
        # max_idx = torch.argmax(probs, dim=1)
        # s_index = max_idx // context_len
        # e_index = max_idx % context_len

        if self.use_generator:
            # ctx_span, ctx_span_len = self.getTopKSpans(final_context, probs, 2)
            ctx_span, ctx_span_len = self.extractSpan(final_context, s_index, e_index)
            pred_logits, pred_ans = self.getPredsFromGenerator(hidden_R, ctx_span, ctx_span_len, a_vec.transpose(0,1))
            decoder_loss = self.getDecoderLoss(pred_logits, a_vec)
            loss += decoder_loss
        pred_ans = None

        #pred_a = [c_vec[i, s_index[i]:e_index[i]] for i in range(len(s_index))]
        
        #pred_a_len = [len(a) for a in pred_a]
        #padded_pred_a = torch.zeros(len(pred_a), max(pred_a_len), 
        #                           dtype=c_vec.dtype).to(c_vec.device)
        #for (i,x) in enumerate(pred_a):
        #   padded_pred_a[i,:pred_a_len[i]] = x
            
        #pred_a_emb = self.word_embeddings(padded_pred_a)
        #a_emb = self.word_embeddings(a_vec)

        #sim_loss = self.word_loss(a_emb, pred_a_emb, alen, s_index-e_index)
        #sim_loss = torch.mean(sim_loss)
        #character_ids = batch_to_ids(context).to(c_vec.device)

        #elmo_embeddings = self.elmo(character_ids)
        #final_context = torch.cat([final_context]+elmo_embeddings['elmo_representations'], dim=2)
        # generate_output = self.generative_decoder(final_context, a_vec, c_mask, c_vec)
        # batch_size, target_iter = a_vec.shape
        # gen_out = torch.zeros(batch_size, target_iter).to(generate_output.device)
        # for i in range(batch_size):
        #     gen_out[i,:] = generate_output[i,:,:].max(1)[1]
        # generate_output = generate_output[:,1:,:].contiguous().view(-1, generate_output.shape[-1])
        # generate_output = F.softmax(generate_output, dim=1)
        # eps = 1e-8
        # generate_output = (1-eps)*generate_output + eps*torch.min(generate_output[generate_output != 0])
        # loss = self.gen_loss(torch.log(generate_output), a_vec[:,1:].contiguous().view(-1))
        if not self.use_RLLoss:
            return loss, loss, s_index, e_index, pred_ans

        probs = torch.exp(probs)
        rl_loss = self.DCRL_loss(probs, s_prob, e_prob, context_len, start, end, context, a1, a2)

        self.weight_a = self.weight_a.to(rl_loss.device)
        self.weight_b = self.weight_b.to(rl_loss.device)
        self.half = self.half.to(rl_loss.device)
        a1 = torch.pow(self.weight_a, -2) * self.half
        a2 = torch.pow(self.weight_b, -2) * self.half
        b1 = torch.log(torch.pow(self.weight_a, 2))
        b2 = torch.log(torch.pow(self.weight_b, 2))
        #total_loss = (loss1+loss2)*self.weight_a+rl_loss*self.weight_b
        return loss * a1 + rl_loss * a2 + b1 + b2, loss, s_index, e_index, pred_ans


    def evaluate(self, c_vec, c_pos, c_ner, c_char, c_em, c_char_lens, c_mask, q_vec, q_pos, q_ner, q_char, q_em, q_char_lens, q_mask):
        s_prob, e_prob, probs, final_context, hidden_R = self.getAnswerSpanProbs(c_vec, c_pos, c_ner, c_char, 
                                                                        c_em, c_char_lens, c_mask, q_vec, 
                                                                        q_pos, q_ner, q_char, q_em, 
                                                                        q_char_lens, q_mask)

        s_prob = torch.squeeze(s_prob)
        e_prob = torch.squeeze(e_prob)

        context_len = final_context.shape[1]
        
        s_index = torch.argmax(s_prob, dim=1)
        e_index = torch.argmax(e_prob, dim=1)
        
        # max_idx = torch.argmax(probs, dim=1)
        # s_index = max_idx // context_len
        # e_index = max_idx % context_len

        if self.use_generator:
            ctx_span, ctx_span_len = self.extractSpan(final_context, s_index, e_index)
            _, pred_ans = self.getPredsFromGenerator(hidden_R, ctx_span, ctx_span_len, None)
        else:
            pred_ans = None

        return s_index, e_index, torch.log(s_prob), torch.log(e_prob), pred_ans

if __name__ == '__main__':
    seq_len = 60
    seq_len2 = 40
    batch = 2

    input_size = 100
    vocab_size = 200

    char_embedding_dim = 300
    word_embedding_dim = 100

    char_hidden_size = 200
    encoder_hidden_size = 200

    encoder_input_dim = word_embedding_dim + (2 * char_embedding_dim) 

    word_embeddings = np.random.uniform(-0.25, 0.25, (vocab_size, word_embedding_dim))
    char_embeddings = np.random.uniform(-0.25, 0.25, (vocab_size, char_embedding_dim))
   
    encoder_input_dim = 64 + 64 + word_embedding_dim + (2*10) + 1
              #pos_emb_dim + ner_emb_dim + word_embeddings.shape[1] + (2*hidden_size) + 1
 
    mnemonicReader = MnemicReader(input_size=encoder_input_dim, hidden_size=10, num_layers=1, char_emb_dim =64, \
            pos_emb_dim=64, ner_emb_dim=64, word_embeddings=word_embeddings, num_char=40, num_pos=40, num_ner=40)

    c_vec = torch.Tensor(np.random.randint(40, size=(batch, seq_len))).long()
    c_pos = torch.Tensor(np.random.randint(40, size=(batch, seq_len))).long()
    c_ner = torch.Tensor(np.random.randint(40, size=(batch, seq_len))).long()
    c_em = torch.Tensor(np.random.randint(2, size=(batch, seq_len))).float()

    c_char = torch.Tensor(np.random.randint(40, size=(batch, seq_len, 10))).long()
    c_mask = torch.Tensor(np.random.randint(40, size=(batch, seq_len))).long()
    
    q_vec = torch.Tensor(np.random.randint(40, size=(batch, seq_len2))).long()
    q_pos = torch.Tensor(np.random.randint(40, size=(batch, seq_len2))).long()
    q_ner = torch.Tensor(np.random.randint(40, size=(batch, seq_len2))).long()
    q_em = torch.Tensor(np.random.randint(2, size=(batch, seq_len2))).float()

    q_char = torch.Tensor(np.random.randint(40, size=(batch, seq_len2, 10))).long()
    q_mask = torch.Tensor(np.random.randint(40, size=(batch, seq_len2))).long()

    mnemonicReader(c_vec, c_pos, c_ner, c_char, c_em, c_mask, q_vec, q_pos, q_ner, q_char, q_em, q_mask, start=0, end=4)
