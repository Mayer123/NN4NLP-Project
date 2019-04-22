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
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import sacrebleu
from CSMrouge import RRRouge
from bleu import Bleu
from run import build_dicts,convert_data,generate_embeddings,compute_scores,TextDataset,pad_sequence

def add_arguments(parser):
    parser.add_argument("eval_file", help="File that contains evaluation data")    
    parser.add_argument("model_file", help="File that contains the model")
    parser.add_argument("embedding_file", help="File that contains pre-trained embeddings")
    parser.add_argument('dicts_dir', type=str, help='Directory containing the word dictionaries')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for eval')
    parser.add_argument('--seed', type=int, default=6, help='Random seed for the experiment')
    parser.add_argument('--epochs', type=int, default=5, help='Train data iterations')
    parser.add_argument('--hidden_size', type=int, default=100, help='Hidden size for LSTM')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers for LSTM')
    parser.add_argument('--char_emb_size', type=int, default=50, help='Embedding size for characters')
    parser.add_argument('--pos_emb_size', type=int, default=50, help='Embedding size for pos tags')
    parser.add_argument('--ner_emb_size', type=int, default=50, help='Embedding size for ner')
    parser.add_argument('--log_file', type=str, default="RMR-eval.log", help='path to the log file')    

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
    with open(args.eval_file, 'r') as f:
        eval_data = json.load(f)
    rouge = Rouge()
    rrrouge = RRRouge()
    logger.info('Converting to index')
    
    dicts = []
    for d in ['word_dict', 'tag_dict', 'ner_dict', 'char_dict']:
        with open('%s/%s.json'%(args.dicts_dir, d), 'r') as f:
            dicts.append(json.load(f))
    [w2i, tag2i, ner2i, c2i] = dicts

    eval_data = convert_data(eval_data, w2i, tag2i, ner2i, c2i, 800)
    print (len(w2i), len(tag2i), len(ner2i), len(c2i))
    eval_dataset = TextDataset(list(eval_data))

    logger.info('Generating embeddings')
    embeddings, _ = generate_embeddings(args.embedding_file, w2i)
    eval_loader = torch.utils.data.DataLoader(eval_dataset, shuffle=False, batch_size=args.batch_size, num_workers=4, collate_fn=lambda batch : zip(*batch))

    use_cuda = torch.cuda.is_available()

    input_size = embeddings.shape[1] + args.char_emb_size * 2 + args.pos_emb_size + args.ner_emb_size + 1
    # model = MnemicReader(input_size, args.hidden_size, args.num_layers, 
    #                         args.char_emb_size, args.pos_emb_size, args.ner_emb_size, 
    #                         embeddings, len(c2i)+2, len(tag2i)+2, len(ner2i)+2)
    model = torch.load(args.model_file)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        model.cuda()
    
    model.eval()
    with torch.no_grad():
        eval_start_acc = 0.0
        eval_end_acc = 0.0
        rouge_scores = 0.0
        bleu1_scores = 0.0
        bleu4_scores = 0.0
        another_rouge = 0.0
        eval_loss = 0.0
        all_preds = []
        all_a1 = []
        all_a2 = []
        all_ids = []
        all_scores = []
        Eval_Rouge = []
        Eval_Loss = []
        nlloss = torch.nn.NLLLoss()
        for batch in eval_loader:
            c_vec, c_pos, c_ner, c_char, c_em, q_vec, q_pos, q_ner, q_char, q_em, start, end, c, q, c_a, a1, a2, _id = batch
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

            pred_start, pred_end, s_prob, e_prob = model.evaluate(c_vec, c_pos, c_ner, c_char, c_em, c_mask, q_vec, q_pos, q_ner, q_char, q_em, q_mask)
            loss1 = nlloss(s_prob.cpu(), start)
            loss2 = nlloss(e_prob.cpu(), end)
            CE_loss = loss1 + loss2
            eval_loss += CE_loss.cpu().item()
            batch_score = compute_scores(rouge, rrrouge, pred_start.tolist(), pred_end.tolist(), c, a1, a2)
            rouge_scores += batch_score[0]
            bleu1_scores += batch_score[1]
            bleu4_scores += batch_score[2]
            another_rouge += batch_score[3]
            all_preds.extend(batch_score[4])
            all_a1.extend(a1)
            all_a2.extend(a2)
            all_ids.extend(_id)
            all_scores.extend(batch_score[5])
            eval_start_acc += torch.sum(torch.eq(pred_start.cpu(), start)).item()
            eval_end_acc += torch.sum(torch.eq(pred_end.cpu(), end)).item()
            
        avg_rouge = rouge_scores / len(eval_dataset)
        eval_start_acc /= len(eval_dataset)
        eval_end_acc /= len(eval_dataset)
        avg_bleu1 = bleu1_scores / len(eval_dataset)
        avg_bleu4 = bleu4_scores / len(eval_dataset)
        Eval_Rouge.append(avg_rouge)
        Eval_Loss.append(eval_loss/len(eval_dataset))

        another_rouge_avg = another_rouge / len(eval_dataset)
        # word_target_dict = dict(enumerate(map(lambda item: [item[0], item[1]],zip(all_a1, all_a2))))
        # word_response_dict = dict(enumerate(map(lambda item: [item],all_preds)))
        # coco_bleu, coco_bleus = bleu_obj.compute_score(word_target_dict, word_response_dict)
        # coco_bleu1, _, _, coco_bleu4 = coco_bleu
        eval_output = [{'prediction': pred, 'answer1': a1, 'answer2':a2, 'rouge_score':s, '_id':_id} for pred, a1, a2, s, _id in zip(all_preds, all_a1, all_a2, all_scores, all_ids)]
        logger.info("eval loss %.4f eval average rouge score %.4f, another rouge %.4f, bleu1 score %.4f, bleu4 score %.4f, start acc %.4f, end acc %.4f" % (eval_loss/len(eval_loader), avg_rouge, another_rouge_avg, avg_bleu1, avg_bleu4, eval_start_acc, eval_end_acc))
        with open('Best_eval_output.json', 'w') as fout:
            json.dump(eval_output, fout)            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Reinforced Mnemonic Reader Model")
    add_arguments(parser)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #logger.setLevel(logging.INFO)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)