import sys
sys.path.append('../')
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
from ReadingComprehension.IterativeReattentionAligner.e2e_encoder import MnemicReader as e2e_MnemicReader
import cProfile, pstats, io
from utils import *
from InformationRetrieval.AttentionRM.modules import AttentionRM
from EndToEndModel.modules import EndToEndModel
from nltk.translate.bleu_score import sentence_bleu
import re
import pickle
from CSMrouge import RRRouge
from bleu import Bleu

stoplist = set(['.',',', '...', '..'])
bleu_obj = Bleu(4)

def add_arguments(parser):
    parser.add_argument("train_file", help="File that contains training data")
    parser.add_argument("dev_file", help="File that contains dev data")
    parser.add_argument("embedding_file", help="File that contains pre-trained embeddings")
    parser.add_argument('--dicts_dir', type=str, default=None, help='Directory containing the word dictionaries')
    parser.add_argument('--seed', type=int, default=6, help='Random seed for the experiment')
    parser.add_argument('--epochs', type=int, default=20, help='Train data iterations')
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
    parser.add_argument('--load_model', type=str, default="", help='path to the log file')
    parser.add_argument('--model_name', type=str, default="rmr.pt", help='path to the log file')
    parser.add_argument('--save_results', action='store_true', help='path to the log file')
    parser.add_argument('--RL_loss_after', type=int, default=5, help='path to the log file')
    parser.add_argument('--mode', type=str, default='summary', help='path to the log file')

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
    common_vocab = {}
    for i, (k, v) in enumerate(w2i.most_common()):
        if v >= 31:
            common_vocab[k] = i + 4                         # <SOS> for 2 <EOS> for 3
    for i, (k, v) in enumerate(w2i.most_common()):
        word_dict[k] = i + 4                         # <SOS> for 2 <EOS> for 3
    for i, (k, v) in enumerate(tag2i.most_common()):
        tag_dict[k] = i + 2
    for i, (k, v) in enumerate(ner2i.most_common()):
        ner_dict[k] = i + 2
    for i, (k, v) in enumerate(c2i.most_common()):
        char_dict[k] = i + 2
    count = 0
    count1 = 0
    for sample in data:
        for w in sample['answers'][0].lower():
            count += 1
            if w in common_vocab:
                count1 += 1
        for w in sample['answers'][1].lower():
            count += 1
            if w in common_vocab:
                count1 += 1
    print (count)
    print (count1)
    return word_dict, tag_dict, ner_dict, char_dict, common_vocab

def convert_data(data, w2i, tag2i, ner2i, c2i, common_vocab, max_len=-1):
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
        answer_tokens = word_tokenize(answer1) if random.random() < 0.5 else word_tokenize(answer2)
        answer_vector = [2]+[common_vocab[w] if w in common_vocab else 1 for w in answer_tokens]+[3]
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
            context_tokens, sample['question_tokens'], sample['chosen_answer'], answer1, answer2, sample['_id'], answer_vector)


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
    
    return embeddings, trained_idx

def pad_sequence(sentences, pos, ner, char, em):
    max_len = max([len(sent) for sent in sentences])
    sent_batch = np.zeros((len(sentences), max_len), dtype=int)
    pos_batch = np.zeros((len(sentences), max_len), dtype=int)
    ner_batch = np.zeros((len(sentences), max_len), dtype=int)
    char_batch = np.zeros((len(sentences), max_len, 16), dtype=int)
    em_batch = np.zeros((len(sentences), max_len), dtype=int)
    masks = np.zeros((len(sentences), max_len), dtype=int)
    char_lens = np.ones((len(sentences), max_len), dtype=int)
    for i, sent in enumerate(sentences):
        sent_batch[i,:len(sent)] = np.array(sent)
        pos_batch[i,:len(pos[i])] = np.array(pos[i])
        ner_batch[i,:len(ner[i])] = np.array(ner[i])
        em_batch[i,:len(em[i])] = np.array(em[i])
        masks[i,:len(sent)] = 1
        for j, word in enumerate(sent):
            if len(char[i][j]) > 16:
                char_batch[i, j, :16] = np.array(char[i][j][:16])
                char_lens[i,j] = 16
            else:
                char_batch[i, j, :len(char[i][j])] = np.array(char[i][j])
                char_lens[i,j] = len(char[i][j])
    #print([len(sent) for sent in sentences])
    return torch.as_tensor(sent_batch), torch.as_tensor(pos_batch), torch.as_tensor(ner_batch), torch.as_tensor(em_batch), torch.as_tensor(char_batch), torch.as_tensor(char_lens), torch.as_tensor(masks)

def pad_answer(answers):
    max_len = max([len(ans) for ans in answers])
    ans_batch = np.zeros((len(answers), max_len), dtype=int)
    for i, ans in enumerate(answers):
        ans_batch[i, :len(ans)] = np.array(ans)
    return torch.as_tensor(ans_batch)    

def compute_scores(rouge, rrrouge, start, end, context, a1, a2):
    rouge_score = 0.0
    bleu1 = 0.0
    bleu4 = 0.0
    another_rouge = 0.0
    preds = []
    sample_scores = []
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
        # print ('Extracted Span %s' % predicted_span)
        #print ("Sample output " + str(start[i]) +" " + str(end[i]) + " " + predicted_span + " A1 " + a1[i] + " A2 " + a2[i])
        #score += max(rouge.get_scores(predicted_span, a1[i])[0]['rouge-l']['f'], rouge.get_scores(predicted_span, a2[i])[0]['rouge-l']['f'])
        #return score
        #print ("Sample output " + predicted_span + " A1 " + a1[i] + " A2 " + a2[i])
        rouge_score += max(rouge.get_scores(predicted_span, a1[i])[0]['rouge-l']['f'], rouge.get_scores(predicted_span, a2[i])[0]['rouge-l']['f'])
        bleu1 += sentence_bleu([a1[i].split(),a2[i].split()], predicted_span.split(), weights=(1, 0, 0, 0))
        bleu4 += sentence_bleu([a1[i].split(),a2[i].split()], predicted_span.split(), weights=(0.25, 0.25, 0.25, 0.25))
        another_rouge += rrrouge.calc_score([predicted_span], [a1[i], a2[i]])
        #bleu1 += compute_bleu([[a1[i],a2[i]]], [predicted_span], max_order=1)[0]
        #bleu4 += compute_bleu([[a1[i],a2[i]]], [predicted_span])[0]
        preds.append(predicted_span)
        sample_scores.append(another_rouge)
    return (rouge_score, bleu1, bleu4, another_rouge, preds, sample_scores)

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

def generate_scores(rouge, generate_output, id2words, a1, a2):
    rouge_score = 0.0
    for i in range(0, len(generate_output)):
        pred = generate_answer(generate_output[i], id2words)
        pred_ans = ' '.join(pred)
        if pred_ans in stoplist or pred_ans == '':
            pred_ans = 'NO-ANSWER-FOUND'
        rouge_score += max(rouge.get_scores(pred_ans, a1[i])[0]['rouge-l']['f'], rouge.get_scores(pred_ans, a2[i])[0]['rouge-l']['f'])
        # print ('Generated Output %s' % pred_ans)
        # print (a1[i])
        # print (a2[i])
    return rouge_score

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
    with open(args.train_file, 'rb') as f:
        training_data = pickle.load(f)
    with open(args.dev_file, 'rb') as f:
        dev_data = pickle.load(f)

    w2i = {'<pad>': 0,
            '<unk>' : 1}
    tag2i = w2i.copy()
    ner2i = w2i.copy()
    c2i = w2i.copy()
    logger.info('Converting to index')
    train = convert_fulltext(training_data, w2i, tag2i, ner2i, c2i)
    dev = convert_fulltext(dev_data, w2i, tag2i, ner2i, c2i, update_dict=False)
    train = FulltextDataset(train, args.train_batch_size)
    dev = FulltextDataset(dev, args.dev_batch_size)
    
    train_loader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=1, num_workers=4, collate_fn=mCollateFn)
    dev_loader = torch.utils.data.DataLoader(dev, batch_size=1, num_workers=4, collate_fn=mCollateFn)
    logger.info('Generating embeddings')
    embeddings, trained_idx = generate_embeddings(args.embedding_file, w2i)
    embeddings = torch.from_numpy(embeddings)
    use_cuda = torch.cuda.is_available()

    if args.load_model == '':
        rc_model = e2e_MnemicReader(input_size, args.hidden_size, args.num_layers, 
                            args.pos_emb_size, embeddings, len(tag2i)+2, len(common_vocab)+4,
                            args.emb_dropout, args.rnn_dropout)
        ir_model = AttentionRM(init_emb=embeddings, pos_vocab_size=len(tag2i)+2)
        model = EndToEndModel(ir_model, rc_model)
    else:
        model = torch.load(args.load_model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0008, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 
                                                            factor=0.5, patience=0,
                                                            verbose=True)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        model = model.cuda()

    logger.info('Start training')
    logger.info('-' * 100)
    global_step = 0
    best = 0.0
    for ITER in range(args.epochs):
        train_loss = 0.0
        start_time = time.time()
        model.train()
        if ITER >= args.RL_loss_after:
            model.use_RLLoss = True
        for batch in tqdm.tqdm(train_loader):
            global_step += 1
            #print (global_step)
            q, passage, a, pscore, qlens, plens, slens, alens, a1, a2 = batch
            print(q.shape, passage.shape, a.shape, pscore.shape)
            if use_cuda:
                q = q.cuda()
                passage = passage.cuda()
                a = a.cuda()
                pscore = pscore.cuda()
                qlens = qlens.cuda()
                plens = plens.cuda()
                slens = slens.cuda()
                alens = alens.cuda()
            
            batch_loss = model(q, passage, a, pscore, qlens, plens, alens, slens)
            train_loss += batch_loss.cpu().item()
            optimizer.zero_grad()
            batch_loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step()
            reset_embeddings(model.word_embeddings[0], embeddings, trained_idx)
            # if global_step % 100 == 0:
            #     logger.info("iter %r global_step %s : batch loss=%.4f, time=%.2fs" % (ITER, global_step, batch_loss.cpu().item(), time.time() - start_time))
        logger.info("iter %r global_step %s : train loss/batch=%.4f, time=%.2fs" % (ITER, global_step, train_loss/len(train_loader), time.time() - start_time))
        model.eval()
        with torch.no_grad():
            rouge_scores = 0.0
            bleu1_scores = 0.0
            bleu4_scores = 0.0
            another_rouge = 0.0
            dev_loss = 0.0
            for batch in dev_loader:
                q, passage, a, pscore, qlens, plens, slens, alens, a1, a2 = batch
                if use_cuda:
                    q = q.cuda()
                    passage = passage.cuda()
                    qlens = qlens.cuda()
                    plens = plens.cuda()
                    slens = slens.cuda()
                batch_loss, generate_output = model.evaluate(q, passage, qlens, plens, slens)
                dev_loss += CE_loss.cpu().item()
                batch_score = compute_scores(rouge, rrrouge, generate_output, a1, a2)
                #gen_rouge += generate_scores(rouge, generate_output.tolist(), id2words, a1, a2)
                rouge_scores += batch_score[0]
                bleu1_scores += batch_score[1]
                bleu4_scores += batch_score[2]
                another_rouge += batch_score[3]
            avg_rouge = rouge_scores / len(dev)
            avg_bleu1 = bleu1_scores / len(dev)
            avg_bleu4 = bleu4_scores / len(dev)
            another_rouge_avg = another_rouge / len(dev)
            logger.info("iter %r: dev loss %.4f dev average rouge score %.4f, another rouge %.4f, bleu1 score %.4f, bleu4 score %.4f, time=%.2fs" % (ITER, dev_loss/len(dev_loader), avg_rouge, another_rouge_avg, avg_bleu1, avg_bleu4, time.time() - start_time))
            scheduler.step(avg_rouge)
            if avg_rouge > best:
                best = avg_rouge
                if args.save_results:
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