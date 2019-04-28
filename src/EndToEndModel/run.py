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
            reset_embeddings(model.word_embeddings, embeddings, trained_idx)
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