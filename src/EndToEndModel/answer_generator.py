import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from decoder import Decoder
from allennlp.modules.elmo import Elmo, batch_to_ids

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

class Languagemodel(nn.Module):
        ## model(c_vec, c_pos, c_ner, c_em, c_mask, q_vec, q_pos, q_ner, q_em, q_mask, start, end)
        def __init__(self, input_size, hidden_size, num_layers, word_embeddings, vocab_size, emb_dropout=0.0, rnn_dropout=0.0):
                super(Languagemodel, self).__init__()
                self.num_layers = num_layers
                self.rnn = nn.ModuleList()

                self.vocab_size = vocab_size
                self.emb_size = word_embeddings.shape[1]
                self.word_embeddings = torch.nn.Embedding(word_embeddings.shape[0], word_embeddings.shape[1],
                                                                                                        padding_idx=0)
                self.full_vocab_size = word_embeddings.shape[0]
                self.word_embeddings.weight.data.copy_(torch.from_numpy(word_embeddings))
                self.word_embeddings = nn.Sequential(
                        self.word_embeddings,
                        nn.Dropout(emb_dropout)
                        )

                self.rnn_dropout = nn.Dropout(rnn_dropout)

                for i in range(num_layers):
                        lstm = nn.LSTM(input_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
                        self.rnn.append(lstm)

                self.generative_decoder = Decoder(self.emb_size+2048, hidden_size, self.word_embeddings, self.emb_size, self.vocab_size, self.full_vocab_size, 15, 0.4)
                self.gen_loss = nn.NLLLoss(ignore_index=0)
                self.elmo = Elmo(options_file, weight_file, 2, dropout=0)

        def forward(self, span, a_vec, raw_span):
                span_vec = self.word_embeddings(span)
                character_ids = batch_to_ids(raw_span).cuda()

                elmo_embeddings = self.elmo(character_ids)
                elmo_representations = torch.cat([span_vec]+elmo_embeddings['elmo_representations'], dim=2)
                generate_output = self.generative_decoder(elmo_representations, a_vec, elmo_embeddings['mask'], span)
                batch_size, target_iter = a_vec.shape
                gen_out = torch.zeros(batch_size, target_iter).to(generate_output.device)
                for i in range(batch_size):
                        gen_out[i,:] = generate_output[i,:,:].max(1)[1]
                generate_output = generate_output[:,1:,:].contiguous().view(-1, generate_output.shape[-1])

                generate_output = F.softmax(generate_output, dim=1)
                eps = 1e-8
                generate_output = (1-eps)*generate_output + eps*torch.min(generate_output[generate_output != 0])
                generate_loss = self.gen_loss(torch.log(generate_output), a_vec[:,1:].contiguous().view(-1))
                #print (generate_output.max(1)[1])
                #print (a_vec[:,1:].contiguous().view(-1))
                loss = generate_loss
                return loss, gen_out

        def evaluate(self, span, raw_span):
                span_vec = self.word_embeddings(span)
                character_ids = batch_to_ids(raw_span).cuda()

                elmo_embeddings = self.elmo(character_ids)
                elmo_representations = torch.cat([span_vec]+elmo_embeddings['elmo_representations'], dim=2)
                generate_output = self.generative_decoder.generate(elmo_representations, elmo_embeddings['mask'], span)
                return generate_output
