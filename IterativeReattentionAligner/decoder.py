import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random

class Decoder(nn.Module):

    def __init__(self, input_size, hidden_size, word_embeddings, decode_dim, max_iterations, sampling_rate):
        super(Decoder, self).__init__()

        self.answer_encoder = nn.LSTM(input_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.answer_generator = nn.GRUCell(hidden_size*2+word_embeddings.shape[1], hidden_size)

        self.fc = nn.Linear(hidden_size*2, hidden_size)

        self.word_embeddings = torch.nn.Embedding(word_embeddings.shape[0], word_embeddings.shape[1], 
                                                    padding_idx=0)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(word_embeddings))

        self.embed_dim = word_embeddings.shape[1]

        self.decode_dim = decode_dim

        self.max_iterations = max_iterations

        self.attention = Attention(hidden_size, hidden_size)

        self.out_fc = nn.Linear(hidden_size+word_embeddings.shape[1]+hidden_size*2, decode_dim)

        self.sampling_rate = sampling_rate

    def forward(self, span, true_answer):
        batch_size = span.shape[0]
        encoded_answer = self.answer_encoder(span)     # batch * seq * hidden*2 
        last_state = encoded_answer[1][0]
        hidden = torch.cat((last_state[0,:,:],last_state[1,:,:]), 1)
        init_state = torch.tanh(self.fc(hidden))       # batch * hidden
        target_iterations = true_answer.shape[1]
        results = torch.zeros(batch_size, target_iterations, self.decode_dim).to(span.device)
        output = self.word_embeddings(true_answer[:,0])
        for i in range(1, target_iterations):
            output, hidden = self.decode(output, init_state, encoded_answer[0])
            results[:,i,:] = output
            top1 = output.max(1)[1]
            if random.random() < self.sampling_rate:
                output = self.word_embeddings(true_answer[:,i])
            else:
                output = self.word_embeddings(top1)

        return results

    def decode(self, prev_output, current_state, encoded_span):
        attn_vector = self.attention(encoded_span, current_state).unsqueeze(1)
        attented_span = torch.bmm(attn_vector, encoded_span).squeeze()
        rnn_input = torch.cat([attented_span, prev_output], dim=1)        #  
        hidden = self.answer_generator(rnn_input, current_state)
        output = self.out_fc(torch.cat([hidden, prev_output, attented_span], dim=1))

        return output, hidden

    def generate(self, span):
        batch_size = span.shape[0]
        encoded_answer = self.answer_encoder(span)     # batch * seq * hidden*2 
        last_state = encoded_answer[1][0]
        hidden = torch.cat((last_state[0,:,:],last_state[1,:,:]), 1)
        init_state = torch.tanh(self.fc(hidden))       # batch * hidden
        results = torch.zeros(batch_size, self.max_iterations).to(span.device)
        output = self.word_embeddings(torch.Tensor([2]*batch_size).long().to(span.device))
        for i in range(1, self.max_iterations):
            output, hidden = self.decode(output, init_state, encoded_answer[0])
            top1 = output.max(1)[1]
            results[:,i] = top1
            output = self.word_embeddings(top1)
        return results

class Attention(nn.Module):

    def __init__(self, encode_dim, decode_dim):
        super(Attention, self).__init__()

        self.linear1 = nn.Linear(encode_dim*2+decode_dim, decode_dim)
        self.linear2 = nn.Linear(decode_dim, 1)

    def forward(self, encoded_span, prev_state):

        seq_len = encoded_span.shape[1]
        prev_state = prev_state.unsqueeze(1).repeat(1, seq_len, 1)        # batch * seq * hidden

        energy = torch.tanh(self.linear1(torch.cat([encoded_span, prev_state], 2)))   # batch * seq * decode_hidden
        attn = F.softmax(self.linear2(energy).squeeze(), dim=1)

        return attn



