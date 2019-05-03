import sys
sys.path.append('../../')
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
from ReadingComprehension.IterativeReattentionAligner.CSMrouge import RRRouge
from ReadingComprehension.IterativeReattentionAligner.bleu import Bleu
from ReadingComprehension.IterativeReattentionAligner.encoder import MnemicReader
# from EndToEndModel.answer_generator import AnswerGenerator
from ReadingComprehension.IterativeReattentionAligner.e2e_encoder import MnemicReader as e2e_MnemicReader
import cProfile, pstats, io
from ReadingComprehension.IterativeReattentionAligner.utils import *
from InformationRetrieval.AttentionRM.modules import AttentionRM
from InformationRetrieval.SimpleRM.modules import KNRM
from InformationRetrieval.ConvKNRM.modules import ConvKNRM
from InformationRetrieval.BOWRM.modules import BOWRM
from EndToEndModel.modules import EndToEndModel
from nltk.translate.bleu_score import sentence_bleu
import re
import pickle

stoplist = set(['.',',', '...', '..'])
bleu_obj = Bleu(4)
NUM_WORKERS = 0

def add_arguments(parser):
    parser.add_argument("train_file", help="File that contains training data")
    parser.add_argument("dev_file", help="File that contains dev data")
    parser.add_argument("embedding_file", help="File that contains pre-trained embeddings")
    parser.add_argument('--dicts_dir', type=str, default=None, help='Directory containing the word dictionaries')
    parser.add_argument('--seed', type=int, default=6, help='Random seed for the experiment')
    parser.add_argument('--epochs', type=int, default=20, help='Train data iterations')
    parser.add_argument('--train_batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--dev_batch_size', type=int, default=16, help='Batch size for dev')
    parser.add_argument('--hidden_size', type=int, default=100, help='Hidden size for LSTM')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers for LSTM')
    parser.add_argument('--char_emb_size', type=int, default=50, help='Embedding size for characters')
    parser.add_argument('--pos_emb_size', type=int, default=20, help='Embedding size for pos tags')
    parser.add_argument('--ner_emb_size', type=int, default=50, help='Embedding size for ner')
    parser.add_argument('--emb_dropout', type=float, default=0.3, help='Dropout rate for embedding layers')
    parser.add_argument('--rnn_dropout', type=float, default=0.3, help='Dropout rate for RNN layers')
    parser.add_argument('--log_file', type=str, default="RMR.log", help='path to the log file')
    parser.add_argument('--load_model', type=str, default="", help='whether to load pre-trained model')
    parser.add_argument('--model_name', type=str, default="rmr.pt", help='the name for the saved model')
    parser.add_argument('--save_results', action='store_true', help='whether to store the results')
    parser.add_argument('--RL_loss_after', type=int, default=5, help='activate RL loss after how many epochs')
    parser.add_argument('--mode', type=str, default='summary', help='mode of training') 
    parser.add_argument('--min_occur', type=int, default=100, help='minimum occurance of a word to be counted in common vocab')  
    parser.add_argument('--load_ir_model', type=str, default='', help='mode of training') 
    parser.add_argument('--eval_only', action='store_true', help='whether to store the results')
    parser.add_argument('--use_ir2', action='store_true', help='whether to store the results')
    parser.add_argument('--alt_ir_training', action='store_true', help='whether to store the results')
    parser.add_argument('--n_chunks', type=int, default=1, help='number of chunks to select with the first IR model')  
    parser.add_argument('--n_spans', type=int, default=5, help='number of spans to select using the second IR model')  
    parser.add_argument('--chunk_len', type=int, default=100, help='number of spans to select using the second IR model')  
    parser.add_argument('--span_len', type=int, default=15, help='number of spans to select using the second IR model')  




def compute_scores(rouge, rrrouge, start, end, context, a1, a2, show=False):
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
        if show:
            print ('Extracted Span %s' % predicted_span)
        #print ("Sample output " + str(start[i]) +" " + str(end[i]) + " " + predicted_span + " A1 " + a1[i] + " A2 " + a2[i])
        #score += max(rouge.get_scores(predicted_span, a1[i])[0]['rouge-l']['f'], rouge.get_scores(predicted_span, a2[i])[0]['rouge-l']['f'])
        #return score
        #print ("Sample output " + predicted_span + " A1 " + a1[i] + " A2 " + a2[i])
        # print('pred_span:', predicted_span, start[i], end[i], context[i])
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
        idx = int(idx)
        if idx in skip:
            continue
        if idx == 3:
            return words
        words.append(id2words[idx])
    return words

def generate_scores(rouge, generate_output, id2words, a1, a2, show=False):
    rouge_score = 0.0
    for i in range(0, len(generate_output)):
        pred = generate_answer(generate_output[i], id2words)
        pred_ans = ' '.join(pred)
        if pred_ans in stoplist or pred_ans == '':
            pred_ans = 'NO-ANSWER-FOUND'
        rouge_score += max(rouge.get_scores(pred_ans, a1[i])[0]['rouge-l']['f'], rouge.get_scores(pred_ans, a2[i])[0]['rouge-l']['f'])
        if show:
            print ('Generated Output %s' % pred_ans)
            print (a1[i])
            print (a2[i])
    return rouge_score

def train_full(args):
    global logger
    rouge = Rouge()
    rrrouge = RRRouge()
    logger = logging.getLogger()
    fh = logging.FileHandler(args.log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info('-' * 100)
    logger.info('Loading fulltext data')
    with open(args.train_file, 'rb') as f:
        training_data = pickle.load(f)
    with open(args.dev_file, 'rb') as f:
        dev_data = pickle.load(f)
    w2i, tag2i, ner2i, c2i, common_vocab = build_fulltext_dicts(training_data, args.min_occur)
    id2words = {}
    for k, v in common_vocab.items():
        id2words[v] = k

    w2i['<pad>'] = 0
    w2i['<unk>'] = 1
    tag2i['<pad>'] = 0 
    tag2i['<unk>'] = 1
    ner2i['<pad>'] = 0 
    ner2i['<unk>'] = 1
    c2i['<pad>'] = 0 
    c2i['<unk>'] = 1

    logger.info('Converting to index')
    train = convert_fulltext(training_data, w2i, tag2i, ner2i, c2i, common_vocab, max_len=args.chunk_len, build_chunks=True)
    dev = convert_fulltext(dev_data, w2i, tag2i, ner2i, c2i, common_vocab, max_len=args.chunk_len, build_chunks=True)
    train = FulltextDataset(train, args.train_batch_size)
    dev = FulltextDataset(dev, args.dev_batch_size)
    train_loader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=1, num_workers = NUM_WORKERS, collate_fn=mCollateFn)
    dev_loader = torch.utils.data.DataLoader(dev, batch_size=1, num_workers = NUM_WORKERS, collate_fn=mCollateFn, shuffle=False)
    logger.info('Generating embeddings')
    embeddings, trained_idx = generate_embeddings(args.embedding_file, w2i)
    embeddings = torch.from_numpy(embeddings).float()
    fixed_embeddings = embeddings[trained_idx]

    input_size = embeddings.shape[1] + args.char_emb_size * 2 + args.pos_emb_size + args.ner_emb_size
    #use_cuda = False
    use_cuda = torch.cuda.is_available()
    rc_model = e2e_MnemicReader(input_size, args.hidden_size, args.num_layers, 
                            args.char_emb_size, args.pos_emb_size, args.ner_emb_size, 
                            embeddings, len(c2i)+2, len(tag2i)+2, len(ner2i)+2, len(common_vocab)+4,
                            args.emb_dropout, args.rnn_dropout)
    # rc_model = e2e_MnemicReader(input_size, args.hidden_size, args.num_layers,
    #                         args.pos_emb_size, embeddings, len(tag2i)+2, len(w2i)+4,
    #                         args.emb_dropout, args.rnn_dropout)
    # ir_model = AttentionRM(rc_model.word_embeddings, rc_model.pos_emb, pos_vocab_size=len(tag2i))
    if args.load_ir_model == '':
        # ir_model = KNRM(init_emb=embeddings)
        ir_model = BOWRM(init_emb=embeddings)
    else:
        ir_model = torch.load(args.load_ir_model)

    ag_model = None # AnswerGenerator(input_size, args.hidden_size, args.num_layers, rc_model.word_embeddings, embeddings.shape[1], len(common_vocab)+4, embeddings.shape[0], args.emb_dropout, args.rnn_dropout)
    model = EndToEndModel(ir_model, 
                            AttentionRM(init_emb=embeddings, pos_vocab_size=len(tag2i)), 
                            rc_model, ag_model, w2i, c2i, use_ir2=args.use_ir2, 
                            n_ctx1_sents=args.n_chunks, n_ctx2_sents=args.n_spans,
                            span_size=args.span_len)

    optimizer = torch.optim.Adam([{'params':model.ir_model1.parameters(), 'lr':1e-3},
                                  {'params':model.ir_model2.parameters(), 'lr':1e-3},
                                  {'params':model.rc_model.parameters(), 'lr':0.0008}], 
                                  lr=1e-3, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 
                                                            factor=0.5, patience=1,
                                                            verbose=True)
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enable = True
        torch.cuda.manual_seed(args.seed)
        model = model.cuda()
        fixed_embeddings = fixed_embeddings.cuda()


    logger.info('Start training')
    logger.info('-' * 100)
    global_step = 0
    best = 0.0
    pr = cProfile.Profile()
    for ITER in range(args.epochs):
        train_loss = 0.0
        total_rc_loss = 0.0
        total_ir1_loss = 0.0
        total_ir2_loss = 0.0
        start_time = time.time()
        if not args.eval_only:            
            if ITER >= args.RL_loss_after:
                model.use_RLLoss = True

            model.train()
            if args.use_ir2 and args.alt_ir_training:
                if ITER % 10 < 5:
                    model.ir_model1 = model.ir_model1.train()
                    model.ir_model2 = model.ir_model2.eval()
                else:
                    model.ir_model1 = model.ir_model1.eval()
                    model.ir_model2 = model.ir_model2.train()                
            else:
                model.train()
            # pr.enable()
            t = tqdm.tqdm(train_loader)
            local_step = 0
            for batch in t:
                global_step += 1
                local_step  += 1
                #print (global_step)
                q, q_chars, passage, passage_chars, passage_rouge, avec1, avec2, qlens, slens, alens, a1, a2, p_words = batch                
                # print(q.dtype, passage.dtype, a.dtype, qlens.dtype, slens.dtype, alens.dtype)
                # if global_step == 20:
                #     pr.disable()
                #     s = io.StringIO()
                #     ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
                #     ps.print_stats(10)
                #     print(s.getvalue())
                #     return
                #q, passage, a, qlens, slens, alens, a1, a2 = batch
                if use_cuda:
                    q = q.cuda()
                    q_chars = q_chars.cuda()
                    passage = passage.cuda()
                    passage_chars = passage_chars.cuda()
                    # avec1 = avec1.cuda()
                    # avec2 = avec2.cuda()
                    qlens = qlens.cuda()
                    slens = slens.cuda()
                    alens = alens.cuda()

                rc_loss, ir1_loss, ir2_loss, sidx, eidx = model(q, q_chars, passage, passage_chars, passage_rouge, avec1, avec2, 
                                                qlens, slens, alens, p_words, a1, a2)
                batch_loss = rc_loss+ir1_loss+ir2_loss                
                optimizer.zero_grad()
                batch_loss.backward()

                train_loss += float(batch_loss)
                total_rc_loss += float(rc_loss)
                total_ir1_loss += float(ir1_loss)
                total_ir2_loss += float(ir2_loss)
                torch.nn.utils.clip_grad_norm_(model.rc_model.parameters(),1)
                optimizer.step()

                reset_embeddings(rc_model.word_embeddings, embeddings, trained_idx)  
                t.set_postfix(loss=train_loss/local_step, rc_loss=total_rc_loss/local_step, 
                                ir1_loss=total_ir1_loss/local_step, ir2_loss=total_ir2_loss/local_step)          
                # break
                #rc_model.word_embeddings[0].weight.data[trained_idx] = fixed_embeddings
                # pr.disable()
                # s = io.StringIO()
                # ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
                # ps.print_stats(10)
                # print(s.getvalue())
                # return
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
            gen_rouge = 0.
            count = 0
            for batch in dev_loader:
                q, q_chars, passage, passage_chars, _, avec1, avec2, qlens, slens, alens, a1, a2, p_words = batch
                count += q.shape[0]
                if use_cuda:
                    q = q.cuda()
                    q_chars = q_chars.cuda()
                    passage = passage.cuda()
                    passage_chars = passage_chars.cuda()
                    qlens = qlens.cuda()
                    slens = slens.cuda()
                sidx, eidx, context = model.evaluate(q, q_chars, passage, passage_chars, qlens, slens, p_words)
                # dev_loss += 0# batch_loss.cpu().item()
                # print(context)

                batch_score = compute_scores(rouge, rrrouge, sidx, eidx, context, a1, a2)
                # gen_rouge += generate_scores(rouge, generate_output.cpu().numpy(), id2words, a1, a2)
                rouge_scores += batch_score[0]
                bleu1_scores += batch_score[1]
                bleu4_scores += batch_score[2]
                another_rouge += batch_score[3]
            avg_rouge = rouge_scores / count
            avg_bleu1 = bleu1_scores / count
            avg_bleu4 = bleu4_scores / count
            another_rouge_avg = another_rouge / count
            logger.info("iter %r: dev loss %.4f dev average rouge score %.4f, another rouge %.4f, bleu1 score %.4f, bleu4 score %.4f, time=%.2fs" % (ITER, dev_loss/len(dev_loader), avg_rouge, another_rouge_avg, avg_bleu1, avg_bleu4, time.time() - start_time))
            scheduler.step(another_rouge_avg)
            if avg_rouge > best:
                best = avg_rouge
                if args.save_results:
                    torch.save(model, args.model_name)
                    

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
    rrrouge = RRRouge()
    logger.info('Converting to index')
    if args.dicts_dir is not None:
        dicts = []
        for d in ['word_dict', 'tag_dict', 'ner_dict', 'char_dict', 'common_vocab']:
            with open('%s/%s.json'%(args.dicts_dir, d), 'r') as f:
                dicts.append(json.load(f))
        [w2i, tag2i, ner2i, c2i, common_vocab] = dicts
    else:
        w2i, tag2i, ner2i, c2i, common_vocab = build_dicts(training_data)
    print (len(w2i), len(tag2i), len(ner2i), len(c2i), len(common_vocab))
    train = convert_data(training_data, w2i, tag2i, ner2i, c2i, common_vocab, 800)
    dev = convert_data(dev_data, w2i, tag2i, ner2i, c2i, common_vocab)
    #dev = list(dev)[0:32]
    id2words = {}
    for k, v in common_vocab.items():
        id2words[v] = k

    train = TextDataset(list(train))
    dev = TextDataset(list(dev))
    print (len(train))
    print (len(dev))
    logger.info('Generating embeddings')
    embeddings, trained_idx = generate_embeddings(args.embedding_file, w2i)
    train_loader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=args.train_batch_size, num_workers = NUM_WORKERS, collate_fn=lambda batch : zip(*batch))
    dev_loader = torch.utils.data.DataLoader(dev, batch_size=args.dev_batch_size, num_workers = NUM_WORKERS, collate_fn=lambda batch : zip(*batch))
    #print (embeddings.shape)
    use_cuda = torch.cuda.is_available()
    #use_cuda = False
    input_size = embeddings.shape[1] + args.char_emb_size * 2 + args.pos_emb_size + args.ner_emb_size + 1
    if args.load_model == '':
        model = MnemicReader(input_size, args.hidden_size, args.num_layers, 
                            args.char_emb_size, args.pos_emb_size, args.ner_emb_size, 
                            embeddings, len(c2i)+2, len(tag2i)+2, len(ner2i)+2, len(common_vocab)+4,
                            args.emb_dropout, args.rnn_dropout)
    else:
        model = torch.load(args.load_model)
    
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
    Train_Rouge = []
    Train_Loss = []
    Dev_Rouge = []
    Dev_Loss = []
    pr = cProfile.Profile()
    for ITER in range(args.epochs):
        train_loss = 0.0
        train_CE_loss = 0.0
        train_rouge_score = 0.0
        start_time = time.time()
        model.train()
        if ITER >= args.RL_loss_after:
            model.use_RLLoss = True
        for batch in tqdm.tqdm(train_loader):
            # pr.enable()
            global_step += 1
            #print (global_step)
            c_vec, c_pos, c_ner, c_char, c_em, q_vec, q_pos, q_ner, q_char, q_em, start, end, c, q, c_a, a1, a2, _id, a_vec = batch
            c_vec, c_pos, c_ner, c_em, c_char, c_char_lens, c_mask = pad_sequence(c_vec, c_pos, c_ner, c_char, c_em)
            q_vec, q_pos, q_ner, q_em, q_char, q_char_lens, q_mask = pad_sequence(q_vec, q_pos, q_ner, q_char, q_em)
            a_len = torch.Tensor([len(a) for a in a_vec])
            a_vec = pad_answer(a_vec)
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
                a_vec = a_vec.cuda()
                a_len = a_len.cuda()
            
            batch_loss, CE_loss, s_index, e_index, gen_out = model(c_vec, c_pos, c_ner, c_char, c_em, c_char_lens, c_mask, q_vec, q_pos, q_ner, q_char, q_em, q_char_lens, q_mask, start, end, c, a1, a2, a_vec, a_len)
            train_loss += batch_loss.cpu().item()
            train_CE_loss += CE_loss.cpu().item()
            tmp = generate_scores(rouge, gen_out.tolist(), id2words, a1, a2)
            batch_score = compute_scores(rouge, rrrouge, s_index.tolist(), e_index.tolist(), c, a1, a2)
            train_rouge_score += batch_score[0]
            optimizer.zero_grad()
            batch_loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step()
            reset_embeddings(model.word_embeddings[0], embeddings, trained_idx)
            # pr.disable()
            # s = io.StringIO()
            # ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
            # ps.print_stats()
            # print(s.getvalue())
            # return
            if global_step % 100 == 0:
                print (gen_out.tolist())
                logger.info("iter %r global_step %s : batch loss=%.4f, batch_rouge=%.4f, batch_gen=%.4f, time=%.2fs" % (ITER, global_step, batch_loss.cpu().item(), batch_score[0], tmp, time.time() - start_time))
        Train_Rouge.append(train_rouge_score/len(train))
        Train_Loss.append(train_CE_loss/len(train_loader))
        logger.info("iter %r global_step %s : train loss/batch=%.4f, train CE loss/batch %.4f, train rouge score %.4f, time=%.2fs" % (ITER, global_step, train_loss/len(train_loader), train_CE_loss/len(train_loader), train_rouge_score/len(train), time.time() - start_time))
        model.eval()
        with torch.no_grad():
            dev_start_acc = 0.0
            dev_end_acc = 0.0
            rouge_scores = 0.0
            bleu1_scores = 0.0
            bleu4_scores = 0.0
            another_rouge = 0.0
            dev_loss = 0.0
            gen_rouge = 0.0
            all_preds = []
            all_a1 = []
            all_a2 = []
            all_ids = []
            all_scores = []
            nlloss = torch.nn.NLLLoss()
            for batch in dev_loader:
                c_vec, c_pos, c_ner, c_char, c_em, q_vec, q_pos, q_ner, q_char, q_em, start, end, c, q, c_a, a1, a2, _id, a_vec = batch
                c_vec, c_pos, c_ner, c_em, c_char, c_char_lens, c_mask = pad_sequence(c_vec, c_pos, c_ner, c_char, c_em)
                q_vec, q_pos, q_ner, q_em, q_char, q_char_lens, q_mask = pad_sequence(q_vec, q_pos, q_ner, q_char, q_em)
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

                pred_start, pred_end, s_prob, e_prob, generate_output = model.evaluate(c_vec, c_pos, c_ner, c_char, c_em, c_char_lens, c_mask, q_vec, q_pos, q_ner, q_char, q_em, q_char_lens, q_mask, c)
                loss1 = nlloss(s_prob.cpu(), start)
                loss2 = nlloss(e_prob.cpu(), end)
                CE_loss = loss1 + loss2
                dev_loss += CE_loss.cpu().item()
                batch_score = compute_scores(rouge, rrrouge, pred_start.tolist(), pred_end.tolist(), c, a1, a2, show=True)
                gen_rouge += generate_scores(rouge, generate_output.tolist(), id2words, a1, a2, show=True)
                rouge_scores += batch_score[0]
                bleu1_scores += batch_score[1]
                bleu4_scores += batch_score[2]
                another_rouge += batch_score[3]
                all_preds.extend(batch_score[4])
                all_a1.extend(a1)
                all_a2.extend(a2)
                all_ids.extend(_id)
                all_scores.extend(batch_score[5])
                dev_start_acc += torch.sum(torch.eq(pred_start.cpu(), start)).item()
                dev_end_acc += torch.sum(torch.eq(pred_end.cpu(), end)).item()
            avg_rouge = rouge_scores / len(dev)
            dev_start_acc /= len(dev)
            dev_end_acc /= len(dev)
            avg_bleu1 = bleu1_scores / len(dev)
            avg_bleu4 = bleu4_scores / len(dev)
            Dev_Rouge.append(avg_rouge)
            Dev_Loss.append(dev_loss/len(dev_loader))

            another_rouge_avg = another_rouge / len(dev)
            gen_rouge_avg = gen_rouge / len(dev)
            # word_target_dict = dict(enumerate(map(lambda item: [item[0], item[1]],zip(all_a1, all_a2))))
            # word_response_dict = dict(enumerate(map(lambda item: [item],all_preds)))
            # coco_bleu, coco_bleus = bleu_obj.compute_score(word_target_dict, word_response_dict)
            # coco_bleu1, _, _, coco_bleu4 = coco_bleu
            dev_output = [{'prediction': pred, 'answer1': a1, 'answer2':a2, 'rouge_score':s, '_id':_id} for pred, a1, a2, s, _id in zip(all_preds, all_a1, all_a2, all_scores, all_ids)]
            logger.info("iter %r: dev loss %.4f dev generate rouge %.4f dev average rouge score %.4f, another rouge %.4f, bleu1 score %.4f, bleu4 score %.4f, start acc %.4f, end acc %.4f time=%.2fs" % (ITER, dev_loss/len(dev_loader), gen_rouge_avg, avg_rouge, another_rouge_avg, avg_bleu1, avg_bleu4, dev_start_acc, dev_end_acc, time.time() - start_time))
            if model.use_RLLoss == True:
                scheduler.step(avg_rouge)
            if avg_rouge > best:
                best = avg_rouge
                if args.save_results:
                    torch.save(model, args.model_name)
                    with open('Best_dev_output.json', 'w') as fout:
                        json.dump(dev_output, fout)
    exp_stats = {'training_loss':Train_Loss, 'dev_loss':Dev_Loss, 'training_rouge':Train_Rouge, 'dev_rouge':Dev_Rouge}
    with open('experiment_stats.json', 'w') as fout:
        json.dump(exp_stats, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Reinforced Mnemonic Reader Model")
    add_arguments(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #logger.setLevel(logging.INFO)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.mode == 'summary':
        main(args)
    elif args.mode == 'fulltext':
        train_full(args)
    else:
        print ("undefined training mode")