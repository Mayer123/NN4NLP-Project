import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random

class Decoder(nn.Module):

    def __init__(self, input_size, hidden_size, word_embeddings, embed_dim, decode_dim, max_iterations, sampling_rate):
        super(Decoder, self).__init__()
        self.embed_dim = embed_dim
        self.answer_encoder = nn.GRU(input_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.answer_generator = nn.GRUCell(hidden_size*2+self.embed_dim, hidden_size)

        self.fc = nn.Linear(hidden_size*2, hidden_size)
        self.word_embeddings = word_embeddings
        
        self.decode_dim = decode_dim

        self.max_iterations = max_iterations

        self.attention = Attention(hidden_size)

        self.out_fc = nn.Linear(hidden_size+self.embed_dim+hidden_size*2, decode_dim)

        self.sampling_rate = sampling_rate

    def forward(self, span, true_answer):
        batch_size = span.shape[0]
        encoded_span, last_state = self.answer_encoder(span)     # batch * seq * hidden*2 
        hidden = torch.cat((last_state[0,:,:],last_state[1,:,:]), 1)
        hidden = torch.tanh(self.fc(hidden))       # batch * hidden
        target_iterations = true_answer.shape[1]
        results = torch.zeros(batch_size, target_iterations, self.decode_dim).to(span.device)
        output = self.word_embeddings(true_answer[:,0])
        for i in range(1, target_iterations):
            output, hidden = self.decode(output, hidden, encoded_span)
            results[:,i,:] = output
            top1 = output.max(1)[1]
            output = self.word_embeddings(true_answer[:,i])
            # if random.random() < self.sampling_rate:
            #     output = self.word_embeddings(true_answer[:,i])
            # else:
            #     output = self.word_embeddings(top1)
        return results

    def decode(self, prev_output, hidden, encoded_span):
        attn_vector = self.attention(encoded_span, hidden).unsqueeze(1)
        attented_span = torch.bmm(attn_vector, encoded_span).squeeze()
        rnn_input = torch.cat([attented_span, prev_output], dim=1)        #  
        hidden = self.answer_generator(rnn_input, hidden)
        output = self.out_fc(torch.cat([hidden, prev_output, attented_span], dim=1))

        return output, hidden

    def generate(self, span):
        batch_size = span.shape[0]
        encoded_span, last_state = self.answer_encoder(span)     # batch * seq * hidden*2 
        hidden = torch.cat((last_state[0,:,:],last_state[1,:,:]), 1)
        hidden = torch.tanh(self.fc(hidden))       # batch * hidden
        results = torch.zeros(batch_size, self.max_iterations).to(span.device)
        output = self.word_embeddings(torch.Tensor([2]*batch_size).long().to(span.device))
        for i in range(1, self.max_iterations):
            output, hidden = self.decode(output, hidden, encoded_span)
            top1 = output.max(1)[1]
            results[:,i] = top1
            output = self.word_embeddings(top1)
        return results

class Attention(nn.Module):

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.key_layer = nn.Linear(hidden_size*2, hidden_size)
        self.query_layer = nn.Linear(hidden_size, hidden_size)
        self.energy_layer = nn.Linear(hidden_size, 1)

    def forward(self, encoded_span, hidden):

        #seq_len = encoded_span.shape[1]
        key = self.key_layer(encoded_span)             # batch * seq * hidden
        hidden = hidden.unsqueeze(1)        # batch * 1 * hidden
        query = self.query_layer(hidden)        # batch * 1 * hidden 

        energy = torch.tanh(key + query)   # batch * seq * decode_hidden
        attn = F.softmax(self.energy_layer(energy).squeeze(), dim=1)

        return attn



