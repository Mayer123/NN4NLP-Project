### Adding wiki test dataset

import numpy as np
import time
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tests import test_prediction, test_generation
import torch.nn.utils as utils
from torch.nn import functional as F


# data_directory = "/home/hyd/workhorse2/narrativeQA/NN4NLP-Project/wikitext-103/"

# train_map_data = data_directory + "wiki.train.tokens.mapped"
# valid_map_data = data_directory + "wiki.valid.tokens.mapped"
# test_map_data = data_directory + "wiki.test.tokens.mapped"

# train_data_vocab = data_directory + "wiki.train.vocab"
# valid_data_vocab = data_directory + "wiki.valid.vocab"
# test_data_vocab = data_directory + "wiki.test.vocab"

############# Penn tree bank ####################
data_directory = "/home/hyd/workhorse2/narrativeQA/NN4NLP-Project/LM_data/ptb/"
train_new_data = data_directory + "ptb.train.txt"
valid_new_data = data_directory + "ptb.valid.txt"
test_new_data = data_directory + "ptb.test.txt"

train_map_data = data_directory + "ptb.train.mapped"
valid_map_data = data_directory + "ptb.valid.mapped"
test_map_data = data_directory + "ptb.test.mapped"

train_data_vocab = data_directory + "ptb.train.vocab"
valid_data_vocab = data_directory + "ptb.valid.vocab"
test_data_vocab = data_directory + "ptb.test.vocab"
#################################################


train_dataset = open(train_map_data, 'r').read().split()
train_vocab = list(map(lambda x: tuple(x.split()), open(train_data_vocab, 'r').readlines()))

train_string_2_int_dict = {k:int(v) for k,v in train_vocab} # 0 is <sos>, 1 is <eos>
train_int_2_string_dict = {int(v):k for k,v in train_vocab}
# train_int_2_string_dict[0] = "<sos>"
# train_int_2_string_dict[1] = "<eos>"

print("CHANGING THE TEST DATA TO TRAIN")

test_dataset = open(test_map_data, 'r').read().split()
test_vocab = list(map(lambda x: tuple(x.split()), open(test_data_vocab, 'r').readlines()))
test_string_2_int_dict = {k:int(v) for k,v in test_vocab}
test_int_2_string_dict = {int(v):k for k,v in test_vocab}

class LanguageModelDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seq_len = 70 # two for sos and eos

    def __iter__(self):
        # concatenate your articles and build into batches
        # np.random.shuffle(self.dataset)
        x = np.array(self.dataset).astype(np.int32)  # start and end index 
        # print(x.shape)
        noWords = len(x)
        expLength = noWords // self.batch_size
        x = x[:-(noWords % ( self.batch_size * expLength) )] # 2075648
        x = x.reshape(self.batch_size, expLength).transpose() # (32432, 64)
        indices = np.array(range(0, expLength - self.seq_len, self.seq_len))
        np.random.shuffle(indices)

        for i in indices:
            feat = torch.LongTensor(x[i : i+self.seq_len, : ])  # convert to float? torch.Tensor
            # feat = torch.cat([torch.zeros(1, feat.shape[1]).long(), feat], dim=0) 
            label = torch.LongTensor( x[i+1 : i+self.seq_len+1, :] ) 
            # label = torch.LongTensor( x[i+self.seq_len+1, :] ) 
            yield feat.cuda(), label.cuda()
     
class LanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super(LanguageModel, self).__init__()
        # 3 lstm layers, projection layer, 
        print("VOCAB SIZE: ", vocab_size)
        self.embedding = nn.Embedding(vocab_size, 400)
        self.rnn1 = nn.LSTM(400, 1024)
        self.rnn2 = nn.LSTM(1024, 1024)
        self.rnn3 = nn.LSTM(1024, 400)
        self.fc1 = nn.Linear(400, vocab_size)
        self.fc1.weight = self.embedding.weight
        
    def forward(self, inp, future):
        # print("INPUT TO FORARD")
        # print(inp.shape)
        out = self.embedding(inp)
        out, _ = self.rnn1(out)
        out, _ = self.rnn2(out)
        out, _ = self.rnn3(out)
        out = self.fc1(out)
        logits = out # + Variable(sample_gumbel(out.size(), out=out.data.new())) # Generally done in generation 

        if future > 0:
            futures = []
            logits += Variable(sample_gumbel(out.size(), out=out.data.new())).cuda()
            out = torch.max(logits[-1:, :, :], 2)[1] + 1 #logits = (seq_len, batch_size, vocab_size )

            for _ in range(future - 1):
                out = self.embedding(out)
                out, _ = self.rnn1(out)
                out, _ = self.rnn2(out)
                out, _ = self.rnn3(out)
                out = self.fc1(out)
                h = out + Variable(sample_gumbel(out.size(), out=out.data.new())).cuda()
                futures.append(h) 
                out = torch.max(h, 2)[1] + 1

            logits = [logits] + futures
        return logits

class CrossEntropyLoss3D(nn.CrossEntropyLoss):
    def forward(self, inputVector, targetVector):
        # print("inside the loss func")
        # print(inputVector.shape)
        # print(targetVector.shape)
        # print('========================')
        criterion = super(CrossEntropyLoss3D, self)
        criterion.__init__(reduction='sum')
        inputReshaped = inputVector.view(-1, inputVector.size()[2])
        targetVector = targetVector.contiguous()
        targetReshaped = targetVector.view(-1)
        
        # print(inputReshaped.shape)
        # print(targetReshaped.shape)

        return criterion.forward(inputReshaped, targetReshaped)


def sample_gumbel(shape, eps=1e-10, out=None):
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    return - torch.log(eps - torch.log(U + eps))


class LanguageModelTrainer:
    def __init__(self, model, loader, valid_loader, max_epochs=1, run_id='exp'):
        """
            Use this class to train your model
        """
        # feel free to add any other parameters here
        self.model = model
        self.loader = loader
        self.train_losses = []
        self.val_losses = []
        self.predictions = []
        self.predictions_test = []
        self.generated_logits = []
        self.generated = []
        self.generated_logits_test = []
        self.generated_test = []
        self.epochs = 0
        self.max_epochs = max_epochs
        self.run_id = run_id
        self.valid_loader = valid_loader
        
        # TODO: Define your optimizer and criterion here
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion =  CrossEntropyLoss3D()

    def train(self):
        
        self.model.train() # set to training mode
        epoch_loss = 0
        num_batches = 0

        for batch_num, (inputs, targets) in enumerate(self.loader):
            self.optimizer.zero_grad()
            # print("input to the network ")
            # print(inputs.shape, targets.shape)
            targets = Variable(targets)
            inputs = Variable(inputs)
            out = self.model(inputs, future=0)
            preds = torch.max(out, 2)[1]  # (L, BS)
            preds = preds.data.cpu().numpy()    
            n = preds.shape[1]
            for i in range(5):
                transcript = self.decode_output(preds[:, i], train_int_2_string_dict)
                # print(transcript)
            loss = self.criterion(out, targets)
            epoch_loss += loss.item()
            loss.backward()
            self.optimizer.step()

        epoch_loss = epoch_loss / (batch_num + 1)
        print('[TRAIN]  Epoch [%d/%d]   Loss: %.4f'
                      % (self.epochs , self.max_epochs, epoch_loss))
        self.epochs += 1
        self.train_losses.append(epoch_loss)

    def decode_output(self, output, train_int_2_string_dict):
        words = []
        # print(output.shape)
        for o in output:
            # if o == 1:
            #     break
            words.append(train_int_2_string_dict[o])
        return " ".join(words)

    def test(self):
        # don't change these
        self.model.eval() # set to eval mode
        loss = 0
        c = 0
        for batch_num, (inputs, targets) in enumerate(self.valid_loader):
            predictions = self.model(inputs.cuda(), future=0) # get predictions #70, 20, 18838
            preds = torch.max(predictions, 2)[1]  # (L, BS)
            # print(preds.shape, targets.shape)
            # exit()
            preds = preds.data.cpu().numpy()    
            # print(preds)
            # print(preds.shape)
            # exit() 
            n = preds.shape[1]
            # for i in range(n):
                # transcript = self.decode_output(preds[:, i], train_int_2_string_dict)
                # print(transcript)
                # yield transcript

            loss += float(self.criterion(predictions, targets))
            c += 1

        print('[VAL]  Epoch [%d/%d]   Loss: %.4f'
                    % (self.epochs, self.max_epochs, loss/c))
        return loss

    def save(self):
        # don't change these
        model_path = os.path.join('experiments', self.run_id, 'model-{}.pkl'.format(self.epochs))
        torch.save({'state_dict': self.model.state_dict()},
            model_path)



NUM_EPOCHS = 30
BATCH_SIZE = 20

run_id = str(int(time.time()))
if not os.path.exists('./experiments'):
    os.mkdir('./experiments')
os.mkdir('./experiments/%s' % run_id)
print("Saving models to ./experiments/%s" % run_id)

model = LanguageModel(len(train_vocab)).cuda()
model = nn.DataParallel(model, device_ids=[0])

train_loader = LanguageModelDataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = LanguageModelDataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)
trainer = LanguageModelTrainer(model=model, loader=train_loader, valid_loader = valid_loader, max_epochs=NUM_EPOCHS, run_id=run_id)



best_nll = 1e30 
for epoch in range(NUM_EPOCHS):
    trainer.train()
    # trainer.test()
    nll = trainer.test()
    if nll < best_nll:
        best_nll = nll
        trainer.save()
