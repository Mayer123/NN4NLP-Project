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
from encoder import MnemicReader
import cProfile, pstats, io
#from bleu import compute_bleu
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
#import sacrebleu
from CSMrouge import RRRouge
from bleu import Bleu
from run import generate_embeddings,compute_scores,pad_sequence

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

class TextDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        data.sort(key=lambda x: len(x[0]))
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], sample[6], sample[7], sample[8]\
        , sample[9], sample[10], sample[11], sample[12], sample[13], sample[14], sample[15], sample[16], sample[17]

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
                    continue
                    #print (current_len)
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
            context_tokens, sample['question_tokens'], sample['chosen_answer'], answer1, answer2, sample['_id'])

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

            pred_start, pred_end, s_prob, e_prob = model.evaluate(c_vec, c_pos, c_ner, c_char, c_em, c_char_lens, c_mask, q_vec, q_pos, q_ner, q_char, q_em, q_char_lens, q_mask, c)
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
