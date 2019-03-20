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
from encoder import MnemicReader
import cProfile, pstats, io
#from bleu import compute_bleu
from nltk.translate.bleu_score import sentence_bleu

stoplist = set(['.',',', '...', '..'])



class TextDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        data.sort(key=lambda x: len(x[0]))
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], sample[6], sample[7], sample[8]\
        , sample[9], sample[10], sample[11], sample[12], sample[13], sample[14], sample[15], sample[16]

def add_arguments(parser):
    parser.add_argument("train_file", help="File that contains training data")
    parser.add_argument("dev_file", help="File that contains dev data")
    parser.add_argument("embedding_file", help="File that contains pre-trained embeddings")
    parser.add_argument('--dicts_dir', type=str, default=None, help='Directory containing the word dictionaries')
    parser.add_argument('--seed', type=int, default=6, help='Random seed for the experiment')
    parser.add_argument('--epochs', type=int, default=5, help='Train data iterations')
    parser.add_argument('--train_batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--dev_batch_size', type=int, default=32, help='Batch size for dev')
    parser.add_argument('--hidden_size', type=int, default=100, help='Hidden size for LSTM')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers for LSTM')
    parser.add_argument('--char_emb_size', type=int, default=50, help='Embedding size for characters')
    parser.add_argument('--pos_emb_size', type=int, default=50, help='Embedding size for pos tags')
    parser.add_argument('--ner_emb_size', type=int, default=50, help='Embedding size for ner')
    parser.add_argument('--emb_dropout', type=float, default=0.3, help='Dropout rate for embedding layers')
    parser.add_argument('--rnn_dropout', type=float, default=0.3, help='Dropout rate for RNN layers')
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
        for t in sample['context_pos']:
            tag2i[t] += 1
        for e in sample['context_ner']:
            ner2i[e] += 1
        for w in sample['question_tokens']:
            w2i[w] += 1
            for c in w:
                c2i[c] += 1
        for t in sample['question_pos']:
            tag2i[t] += 1
        for e in sample['question_ner']:
            ner2i[e] += 1
    word_dict = {}
    tag_dict = {}
    ner_dict = {}
    char_dict = {}
    for i, (k, v) in enumerate(w2i.most_common()):
        word_dict[k] = i + 2
    for i, (k, v) in enumerate(tag2i.most_common()):
        tag_dict[k] = i + 2
    for i, (k, v) in enumerate(ner2i.most_common()):
        ner_dict[k] = i + 2
    for i, (k, v) in enumerate(c2i.most_common()):
        char_dict[k] = i + 2
    # 0 for padding and 1 for unk
    for d in ['word_dict', 'tag_dict', 'ner_dict', 'char_dict']:
        with open('../prepro/dicts/%s.json'%d, 'w') as f:
            json.dump(locals()[d], f)
    return word_dict, tag_dict, ner_dict, char_dict

def convert_data(data, w2i, tag2i, ner2i, c2i, max_len=-1):
    for i, sample in enumerate(data):
        context_vector = [w2i[w] if w in w2i else 1 for w in sample['context_tokens']]
        context_pos_vec = [tag2i[t] if t in tag2i else 1 for t in sample['context_pos']]
        context_ner_vec = [ner2i[e] if e in ner2i else 1 for e in sample['context_ner']]
        context_character = [[c2i[c] if c in c2i else 1 for c in w] for w in sample['context_tokens']]
        question_vector = [w2i[w] if w in w2i else 1 for w in sample['question_tokens']]
        question_pos_vec = [tag2i[t] if t in tag2i else 1 for t in sample['question_pos']]
        question_ner_vec = [ner2i[e] if e in ner2i else 1 for e in sample['question_ner']]
        question_character = [[c2i[c] if c in c2i else 1 for c in w] for w in sample['question_tokens']]
        context_em = sample['context_em_feature']
        context_tokens = sample['context_tokens']
        answer1 = sample['answers'][0].lower()
        answer2 = sample['answers'][1].lower()
        ans_start = sample['start_index']
        ans_end = sample['end_index']
        if max_len != -1 and len(context_vector) > max_len:
            if sample['start_index'] >= max_len or sample['end_index'] >= max_len: 
                new_start = len(context_vector) - max_len
                if new_start > sample['start_index']:
                    print('This context is too long')
                    print (current_len)
                context_vector = context_vector[new_start:new_start+max_len]
                context_pos_vec = context_pos_vec[new_start:new_start+max_len]
                context_ner_vec = context_ner_vec[new_start:new_start+max_len]
                context_character = context_character[new_start:new_start+max_len]
                context_em = context_em[new_start:new_start+max_len]
                context_tokens = context_tokens[new_start:new_start+max_len]
                ans_start = ans_start - new_start
                ans_end = ans_end - new_start
            else:
                context_vector = context_vector[:max_len]
                context_pos_vec = context_pos_vec[:max_len]
                context_ner_vec = context_ner_vec[:max_len]
                context_character = context_character[:max_len]
                context_em = context_em[:max_len]
                context_tokens = context_tokens[:max_len]
        yield (context_vector, context_pos_vec, context_ner_vec, context_character, context_em, \
            question_vector, question_pos_vec, question_ner_vec, question_character, sample['question_em_feature'], ans_start, ans_end, \
            context_tokens, sample['question_tokens'], sample['chosen_answer'], answer1, answer2)


def generate_embeddings(filename, word_dict):
    embeddings = np.random.uniform(-0.25, 0.25, (len(word_dict)+2, 100))
    count = 0
    with open(filename, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            tokens = line.split()
            if tokens[0] in word_dict:
                embeddings[word_dict[tokens[0]]] = np.array(list(map(float, tokens[1:])))
                count += 1
    logger.info('Total vocab size %s pre-trained words %s' % (len(word_dict), count))
    np.save("../prepro/embeddings.npy", embeddings)
    return embeddings

def pad_sequence(sentences, pos, ner, char, em):
    max_len = max([len(sent) for sent in sentences])
    sent_batch = np.zeros((len(sentences), max_len), dtype=int)
    pos_batch = np.zeros((len(sentences), max_len), dtype=int)
    ner_batch = np.zeros((len(sentences), max_len), dtype=int)
    char_batch = np.zeros((len(sentences), max_len, 16), dtype=int)
    em_batch = np.zeros((len(sentences), max_len), dtype=int)
    masks = np.zeros((len(sentences), max_len), dtype=int)
    for i, sent in enumerate(sentences):
        sent_batch[i,:len(sent)] = np.array(sent)
        pos_batch[i,:len(pos[i])] = np.array(pos[i])
        ner_batch[i,:len(ner[i])] = np.array(ner[i])
        em_batch[i,:len(em[i])] = np.array(em[i])
        masks[i,:len(sent)] = 1
        for j, word in enumerate(sent):
            if len(char[i][j]) > 16:
                char_batch[i, j, :16] = np.array(char[i][j][:16])
            else:
                char_batch[i, j, :len(char[i][j])] = np.array(char[i][j])
    #print([len(sent) for sent in sentences])
    return torch.as_tensor(sent_batch), torch.as_tensor(pos_batch), torch.as_tensor(ner_batch), torch.as_tensor(em_batch), torch.as_tensor(char_batch), torch.as_tensor(masks)

def compute_scores(rouge, start, end, context, a1, a2):
    rouge_score = 0.0
    bleu1 = 0.0
    bleu4 = 0.0
    for i in range(0, len(start)):
        #print (context[i])
        #print (start[i], end[i])
        #print (len(context[i]), context[i][start[i]:end[i]+1])
        if start[i] > end[i]:
            predicted_span = 'NO-ANSWER-FOUND'
        else:
            predicted_span = ' '.join(context[i][start[i]:end[i]+1])
        if predicted_span in stoplist:
            predicted_span = 'NO-ANSWER-FOUND'
        #print ("Sample output " + predicted_span + " A1 " + a1[i] + " A2 " + a2[i])
        rouge_score += max(rouge.get_scores(predicted_span, a1[i])[0]['rouge-l']['f'], rouge.get_scores(predicted_span, a2[i])[0]['rouge-l']['f'])
        bleu1 += sentence_bleu([a1[i].split(),a2[i].split()], predicted_span.split(), weights=(1, 0, 0, 0))
        bleu4 += sentence_bleu([a1[i].split(),a2[i].split()], predicted_span.split(), weights=(0.25, 0.25, 0.25, 0.25))
        #bleu1 += compute_bleu([[a1[i],a2[i]]], [predicted_span], max_order=1)[0]
        #bleu4 += compute_bleu([[a1[i],a2[i]]], [predicted_span])[0]
    return (rouge_score, bleu1, bleu4)


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
    rouge = Rouge()
    logger.info('Converting to index')
    if args.dicts_dir is not None:
        dicts = []
        for d in ['word_dict', 'tag_dict', 'ner_dict', 'char_dict']:
            with open('%s/%s.json'%(args.dicts_dir, d), 'r') as f:
                dicts.append(json.load(f))
        [w2i, tag2i, ner2i, c2i] = dicts
    else:
        w2i, tag2i, ner2i, c2i = build_dicts(training_data)
    train = convert_data(training_data, w2i, tag2i, ner2i, c2i, 800)
    dev = convert_data(dev_data, w2i, tag2i, ner2i, c2i)
    print (len(w2i), len(tag2i), len(ner2i), len(c2i))
    train = TextDataset(list(train))
    dev = TextDataset(list(dev))
    print (len(train))
    print (len(dev))
    logger.info('Generating embeddings')
    embeddings = generate_embeddings(args.embedding_file, w2i)
    train_loader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=args.train_batch_size, num_workers=4, collate_fn=lambda batch : zip(*batch))
    dev_loader = torch.utils.data.DataLoader(dev, batch_size=args.dev_batch_size, num_workers=4, collate_fn=lambda batch : zip(*batch))
    #print (embeddings.shape)
    use_cuda = torch.cuda.is_available()
    #use_cuda = False
    input_size = embeddings.shape[1] + args.char_emb_size * 2 + args.pos_emb_size + args.ner_emb_size + 1
    model = MnemicReader(input_size, args.hidden_size, args.num_layers, 
                            args.char_emb_size, args.pos_emb_size, args.ner_emb_size, 
                            embeddings, len(c2i)+2, len(tag2i)+2, len(ner2i)+2, 
                            args.emb_dropout, args.rnn_dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0008, weight_decay=0.0001)
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
            #print (global_step)
            c_vec, c_pos, c_ner, c_char, c_em, q_vec, q_pos, q_ner, q_char, q_em, start, end, c, q, c_a, a1, a2 = batch
            c_vec, c_pos, c_ner, c_em, c_char, c_mask = pad_sequence(c_vec, c_pos, c_ner, c_char, c_em)
            q_vec, q_pos, q_ner, q_em, q_char, q_mask = pad_sequence(q_vec, q_pos, q_ner, q_char, q_em)
            start = torch.as_tensor(start)
            end = torch.as_tensor(end)
            c_em = c_em.float()
            q_em = q_em.float()
            if use_cuda:
                c_vec = c_vec.cuda()
                c_pos = c_pos.cuda()
                c_ner = c_ner.cuda()
                c_char = c_char.cuda()
                c_em = c_em.cuda()
                c_mask = c_mask.cuda()
                q_vec = q_vec.cuda()
                q_pos = q_pos.cuda()
                q_ner = q_ner.cuda()
                q_char = q_char.cuda()
                q_em = q_em.cuda()
                q_mask = q_mask.cuda()
                start = start.cuda()
                end = end.cuda()
            batch_loss = model(c_vec, c_pos, c_ner, c_char, c_em, c_mask, q_vec, q_pos, q_ner, q_char, q_em, q_mask, start, end, c)
            train_loss += batch_loss.cpu().item()
            optimizer.zero_grad()
            batch_loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(),10)
            optimizer.step()
            #if global_step % 100 == 0:
            #    logger.info("iter %r global_step %s : batch loss=%.4f, time=%.2fs" % (ITER, global_step, batch_loss.cpu().item(), time.time() - start_time))

        logger.info("iter %r global_step %s : train loss/batch=%.4f, time=%.2fs" % (ITER, global_step, train_loss/len(train_loader), time.time() - start_time))
        model.eval()
        with torch.no_grad():
            dev_start_acc = 0.0
            dev_end_acc = 0.0
            rouge_scores = 0.0
            bleu1_scores = 0.0
            bleu4_scores = 0.0
            for batch in dev_loader:
                c_vec, c_pos, c_ner, c_char, c_em, q_vec, q_pos, q_ner, q_char, q_em, start, end, c, q, c_a, a1, a2 = batch
                c_vec, c_pos, c_ner, c_em, c_char, c_mask = pad_sequence(c_vec, c_pos, c_ner, c_char, c_em)
                q_vec, q_pos, q_ner, q_em, q_char, q_mask = pad_sequence(q_vec, q_pos, q_ner, q_char, q_em)
                start = torch.as_tensor(start)
                end = torch.as_tensor(end)
                c_em = c_em.float()
                q_em = q_em.float()
                if use_cuda:
                    c_vec = c_vec.cuda()
                    c_pos = c_pos.cuda()
                    c_ner = c_ner.cuda()
                    c_char = c_char.cuda()
                    c_em = c_em.cuda()
                    c_mask = c_mask.cuda()
                    q_vec = q_vec.cuda()
                    q_pos = q_pos.cuda()
                    q_ner = q_ner.cuda()
                    q_char = q_char.cuda()
                    q_em = q_em.cuda()
                    q_mask = q_mask.cuda()

                pred_start, pred_end = model.evaluate(c_vec, c_pos, c_ner, c_char, c_em, c_mask, q_vec, q_pos, q_ner, q_char, q_em, q_mask)


                batch_score = compute_scores(rouge, pred_start.tolist(), pred_end.tolist(), c, a1, a2)
                rouge_scores += batch_score[0]
                bleu1_scores += batch_score[1]
                bleu4_scores += batch_score[2]
                dev_start_acc += torch.sum(torch.eq(pred_start.cpu(), start)).item()
                dev_end_acc += torch.sum(torch.eq(pred_end.cpu(), end)).item()
            avg_rouge = rouge_scores / len(dev)
            dev_start_acc /= len(dev)
            dev_end_acc /= len(dev)
            avg_bleu1 = bleu1_scores / len(dev)
            avg_bleu4 = bleu4_scores / len(dev)
            logger.info("iter %r: dev average rouge score %.4f, bleu1 score %.4f, bleu4 score %.4f, start acc %.4f, end acc %.4f time=%.2fs" % (ITER, avg_rouge, avg_bleu1, avg_bleu4, dev_start_acc, dev_end_acc, time.time() - start_time))
            scheduler.step(avg_rouge)
            if avg_rouge > best:
                best = avg_rouge
                torch.save(model, 'best_model2')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Reinforced Mnemonic Reader Model")
    add_arguments(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #logger.setLevel(logging.INFO)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
