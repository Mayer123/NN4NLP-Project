import json
from collections import defaultdict, Counter
import numpy as np
import tqdm
import random
import torch
import torch.utils.data
import time
from rouge import Rouge
import argparse
import logging
from languagemodel import Languagemodel
import re
from CSMrouge import RRRouge
from bleu import Bleu
from allennlp.modules.elmo import batch_to_ids

stoplist = set(['.',',', '...', '..'])
bleu_obj = Bleu(4)

def word_tokenize(text):
  """Split on whitespace and punctuation."""
  return re.findall(r'\w+|[^\w\s]', text, re.UNICODE)

class TextDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        data.sort(key=lambda x: len(x[0]))
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample[0], sample[1], sample[2]

def add_arguments(parser):
    parser.add_argument("train_file", help="File that contains training data")
    parser.add_argument("dev_file", help="File that contains dev data")
    parser.add_argument("embedding_file", help="File that contains pre-trained embeddings")
    parser.add_argument('--dicts_dir', type=str, default=None, help='Directory containing the word dictionaries')
    parser.add_argument('--seed', type=int, default=6, help='Random seed for the experiment')
    parser.add_argument('--epochs', type=int, default=20, help='Train data iterations')
    parser.add_argument('--train_batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--dev_batch_size', type=int, default=32, help='Batch size for dev')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size for LSTM')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers for LSTM')
    parser.add_argument('--char_emb_size', type=int, default=50, help='Embedding size for characters')
    parser.add_argument('--pos_emb_size', type=int, default=50, help='Embedding size for pos tags')
    parser.add_argument('--ner_emb_size', type=int, default=50, help='Embedding size for ner')
    parser.add_argument('--emb_dropout', type=float, default=0.0, help='Dropout rate for embedding layers')
    parser.add_argument('--rnn_dropout', type=float, default=0.0, help='Dropout rate for RNN layers')
    parser.add_argument('--log_file', type=str, default="RMR.log", help='path to the log file')

def build_dicts(data):    
    w2i = Counter()
    tag2i = Counter()
    ner2i = Counter()
    c2i = Counter()
    for i, sample in enumerate(data):
        for w in sample['context_tokens']:
            w2i[w] += 1
            for c in w:
                c2i[c] += 1
        # for t in sample['context_pos']:
        #     tag2i[t] += 1
        # for e in sample['context_ner']:
        #     ner2i[e] += 1
        for w in sample['question_tokens']:
            w2i[w] += 1
            for c in w:
                c2i[c] += 1
        # for t in sample['question_pos']:
        #     tag2i[t] += 1
        # for e in sample['question_ner']:
        #     ner2i[e] += 1
    word_dict = {}
    tag_dict = {}
    ner_dict = {}
    char_dict = {}
    common_vocab = {}
    for i, (k, v) in enumerate(w2i.most_common()):
        if v >= 20:
            common_vocab[k] = i + 4                         # <SOS> for 2 <EOS> for 3
    for i, (k, v) in enumerate(w2i.most_common()):
        word_dict[k] = i + 4                         # <SOS> for 2 <EOS> for 3
    # for i, (k, v) in enumerate(tag2i.most_common()):
    #     tag_dict[k] = i + 2
    # for i, (k, v) in enumerate(ner2i.most_common()):
    #     ner_dict[k] = i + 2
    # for i, (k, v) in enumerate(c2i.most_common()):
    #     char_dict[k] = i + 2
    # count = 0
    # count1 = 0
    # count2 = 0
    # for i, sample in enumerate(data):
    #     for w in sample['answers'][0].lower().split():
    #         count += 1
    #         if w in common_vocab:
    #             count2 += 1
    #         if w in word_dict:
    #             count1 += 1
    # print (count, count1, count2)
    # exit(0)
    for k,v in common_vocab.items():
        assert v == word_dict[k]
    # 0 for padding and 1 for unk
    # for d in ['word_dict', 'tag_dict', 'ner_dict', 'char_dict']:
    #     with open('../prepro/dicts/%s.json'%d, 'w') as f:
    #         json.dump(locals()[d], f)
    return word_dict, tag_dict, ner_dict, char_dict, common_vocab

def convert_data(data, w2i, tag2i, ner2i, c2i, common_vocab, max_len=-1):
    for i, sample in enumerate(data):
        answer1 = sample['answers'][0].lower()
        answer2 = sample['answers'][1].lower()
        answer_tokens = word_tokenize(answer1) if random.random() < 0.5 else word_tokenize(answer2)
        answer_vector = [2]+[common_vocab[w] if w in common_vocab else 1 for w in answer_tokens]+[3]
        chosen_span = sample['chosen_answer']
        span_vector = [2] + [common_vocab[w] if w in common_vocab else 1 for w in chosen_span]+[3]
        chosen_span = ['<sos>'] + chosen_span + ['<eos>']
        yield (span_vector, answer_vector, chosen_span)

def generate_embeddings(filename, word_dict):
    embeddings = np.random.uniform(-0.25, 0.25, (len(word_dict)+4, 100))
    count = 0
    trained_idx = []
    with open(filename, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            tokens = line.split()
            if tokens[0] in word_dict:
                trained_idx.append(word_dict[tokens[0]])
                embeddings[word_dict[tokens[0]]] = np.array(list(map(float, tokens[1:])))
                count += 1
    #logger.info('Total vocab size %s pre-trained words %s' % (len(word_dict), count))
    #np.save("../prepro/embeddings.npy", embeddings)
    return embeddings, trained_idx

def pad_answer(answers):
    max_len = max([len(ans) for ans in answers])
    ans_batch = np.zeros((len(answers), max_len), dtype=int)
    for i, ans in enumerate(answers):
        ans_batch[i, :len(ans)] = np.array(ans)
    return torch.as_tensor(ans_batch)

def generate_answer(indices, id2words):
    words = []
    skip = [0, 1, 2]
    # print (indices)
    for idx in indices:
        if idx in skip:
            continue
        if idx == 3:
            return words
        words.append(id2words[idx])
    return words

def generate_scores(generate_output, id2words, a):
    rouge = Rouge()
    rouge_score = 0.0
    for i in range(0, len(generate_output)):
        pred = generate_answer(generate_output[i], id2words)
        pred_ans = ' '.join(pred)
        if pred_ans in stoplist or pred_ans == '':
            pred_ans = 'NO-ANSWER-FOUND'
        ans = generate_answer(a[i], id2words)
        real_ans = ' '.join(ans)
        if real_ans == '':
            real_ans = 'UNKNOWN-WORD'
        rouge_score += rouge.get_scores(pred_ans, real_ans)[0]['rouge-l']['f']
        print ('Generated Output %s' % pred_ans)
        print (ans)
    return rouge_score

def reset_embeddings(word_embeddings, fixed_embeddings, trained_idx):
    word_embeddings.weight.data[trained_idx] = torch.FloatTensor(fixed_embeddings[trained_idx]).cuda()
    return 

def main(args):
    global logger
    
    logger = logging.getLogger()
    fh = logging.FileHandler(args.log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info('-' * 100)
    logger.info('Loading data')
    with open(args.train_file, 'r') as f:
        training_data = json.load(f)
    with open(args.dev_file, 'r') as f:
        dev_data = json.load(f)

    logger.info('Converting to index')
    if args.dicts_dir is not None:
        dicts = []
        for d in ['word_dict', 'tag_dict', 'ner_dict', 'char_dict']:
            with open('%s/%s.json'%(args.dicts_dir, d), 'r') as f:
                dicts.append(json.load(f))
        [w2i, tag2i, ner2i, c2i] = dicts
    else:
        w2i, tag2i, ner2i, c2i, common_vocab = build_dicts(training_data)
    print (len(w2i), len(tag2i), len(ner2i), len(c2i), len(common_vocab))
    train = convert_data(training_data, w2i, tag2i, ner2i, c2i, common_vocab, 800)
    dev = convert_data(dev_data, w2i, tag2i, ner2i, c2i, common_vocab)
    dev = list(dev)[0:32]
    id2words = {}
    for k, v in common_vocab.items():
        id2words[v] = k
    train = TextDataset(list(train))
    dev = TextDataset(list(dev))
    print (len(train))
    print (len(dev))
    logger.info('Generating embeddings')
    embeddings, trained_idx = generate_embeddings(args.embedding_file, w2i)
    train_loader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=args.train_batch_size, num_workers=4, collate_fn=lambda batch : zip(*batch))
    dev_loader = torch.utils.data.DataLoader(dev, batch_size=args.dev_batch_size, num_workers=4, collate_fn=lambda batch : zip(*batch))
    #print (embeddings.shape)
    use_cuda = torch.cuda.is_available()
    #use_cuda = False
    input_size = embeddings.shape[1]
    model = Languagemodel(input_size, args.hidden_size, args.num_layers, embeddings, len(common_vocab)+4, args.emb_dropout, args.rnn_dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 
                                                            factor=0.5, patience=0,
                                                            verbose=True)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        model.cuda()

    logger.info('Start training')
    logger.info('-' * 100)
    global_step = 0
    best = 0.0
    for ITER in range(args.epochs):
        train_loss = 0.0
        start_time = time.time()
        model.train()
        for batch in tqdm.tqdm(train_loader):
            global_step += 1
            span, a_vec, raw_span = batch
            span = pad_answer(span)
            a_vec = pad_answer(a_vec)
            if global_step == 1:
                print (span.shape)
                print (batch_to_ids(raw_span).shape)
            if use_cuda:
                span = span.cuda()
                a_vec = a_vec.cuda()
            
            batch_loss, gen_out = model(span, a_vec, raw_span)
            train_loss += batch_loss.cpu().item()
            optimizer.zero_grad()
            batch_loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step()
            reset_embeddings(model.word_embeddings[0], embeddings, trained_idx)
            if global_step % 500 == 0:
                print (gen_out)
                print (a_vec)
                logger.info("iter %r global_step %s : batch loss=%.4f, time=%.2fs" % (ITER, global_step, batch_loss.cpu().item(), time.time() - start_time))
        logger.info("iter %r global_step %s : train loss/batch=%.4f, time=%.2fs" % (ITER, global_step, train_loss/len(train_loader), time.time() - start_time))
        model.eval()
        with torch.no_grad():
            gen_rouge = 0.0
            dev_loss = 0.0
            for batch in dev_loader:
                span, a_vec, raw_span = batch
                span = pad_answer(span)
                if use_cuda:
                    span = span.cuda()

                generate_output = model.evaluate(span, raw_span)

                gen_rouge += generate_scores(generate_output.tolist(), id2words, a_vec)
            
            logger.info("iter %r: dev loss %.4f time=%.2fs" % (ITER, dev_loss/len(dev_loader), time.time() - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Reinforced Mnemonic Reader Model")
    add_arguments(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)

