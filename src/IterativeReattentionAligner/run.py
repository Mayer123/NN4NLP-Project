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
import re

stoplist = set(['.',',', '...', '..'])


from CSMrouge import RRRouge
from bleu import Bleu

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
        return sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], sample[6], sample[7], sample[8]\
        , sample[9], sample[10], sample[11], sample[12], sample[13], sample[14], sample[15], sample[16], sample[17], sample[18]

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
    parser.add_argument('--RL_loss_after', type=int, default=3, help='path to the log file')
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
        #print ('Extracted Span %s' % predicted_span)
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
        #print ('Generated Output %s' % pred_ans)
        #print (a1[i])
        #print (a2[i])
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
    rouge = Rouge()
    rrrouge = RRRouge()
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
    common_embeddings, _ = generate_embeddings(args.embedding_file, common_vocab)
    train_loader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=args.train_batch_size, num_workers=4, collate_fn=lambda batch : zip(*batch))
    dev_loader = torch.utils.data.DataLoader(dev, batch_size=args.dev_batch_size, num_workers=4, collate_fn=lambda batch : zip(*batch))
    #print (embeddings.shape)
    use_cuda = torch.cuda.is_available()
    #use_cuda = False
    input_size = embeddings.shape[1] + args.char_emb_size * 2 + args.pos_emb_size + args.ner_emb_size + 1
    if args.load_model == '':
        model = MnemicReader(input_size, args.hidden_size, args.num_layers, 
                            args.char_emb_size, args.pos_emb_size, args.ner_emb_size, 
                            embeddings, len(c2i)+2, len(tag2i)+2, len(ner2i)+2, common_embeddings,
                            args.emb_dropout, args.rnn_dropout)
    else:
        model = torch.load(args.load_model)
    
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
            
            batch_loss, CE_loss, s_index, e_index = model(c_vec, c_pos, c_ner, c_char, c_em, c_char_lens, c_mask, q_vec, q_pos, q_ner, q_char, q_em, q_char_lens, q_mask, start, end, c, a1, a2, a_vec)
            train_loss += batch_loss.cpu().item()
            train_CE_loss += CE_loss.cpu().item()
            #batch_score = compute_scores(rouge, rrrouge, s_index.tolist(), e_index.tolist(), c, a1, a2)
            #train_rouge_score += batch_score[0]
            optimizer.zero_grad()
            batch_loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(),10)
            optimizer.step()
            reset_embeddings(model.word_embeddings[0], embeddings, trained_idx)
            # pr.disable()
            # s = io.StringIO()
            # ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
            # ps.print_stats()
            # print(s.getvalue())
            # return
            # if global_step % 100 == 0:
            #     logger.info("iter %r global_step %s : batch loss=%.4f, time=%.2fs" % (ITER, global_step, batch_loss.cpu().item(), time.time() - start_time))

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

                pred_start, pred_end, s_prob, e_prob, generate_output = model.evaluate(c_vec, c_pos, c_ner, c_char, c_em, c_char_lens, c_mask, q_vec, q_pos, q_ner, q_char, q_em, q_char_lens, q_mask)
                loss1 = nlloss(s_prob.cpu(), start)
                loss2 = nlloss(e_prob.cpu(), end)
                CE_loss = loss1 + loss2
                dev_loss += CE_loss.cpu().item()
                batch_score = compute_scores(rouge, rrrouge, pred_start.tolist(), pred_end.tolist(), c, a1, a2)
                gen_rouge += generate_scores(rouge, generate_output.tolist(), id2words, a1, a2)
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
    main(args)
