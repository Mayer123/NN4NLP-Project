import json
from collections import defaultdict
import numpy as np
import tqdm
import random
import torch
import torch.utils.data
import time
import rouge
import argparse
import logging

logger = logging.getLogger()

class TextDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        data.sort(key=lambda x: len(x[0]))
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], sample[6], sample[7], sample[8]\
        , sample[9], sample[10], sample[11], sample[12], sample[13], sample[14]

def add_arguments(parser):
    parser.add_argument("train_file", help="File that contains training data")
    parser.add_argument("dev_file", help="File that contains dev data")
    parser.add_argument("embedding_file", help="File that contains pre-trained embeddings")
    parser.add_argument('--seed', type=int, default=6, help='Random seed for the experiment')
    parser.add_argument('--epochs', type=int, default=1, help='Train data iterations')
    parser.add_argument('--train_batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--dev_batch_size', type=int, default=32, help='Batch size for dev')


def convert_data(data, w2i, tag2i, ner2i):
    for sample in data:
        context_vector = [w2i[w] for w in sample['context_tokens']]
        context_pos_vec = [tag2i[t] for t in sample['context_pos']]
        context_ner_vec = [ner2i[e] for e in sample['context_ner']]
        question_vector = [w2i[w] for w in sample['question_tokens']]
        question_pos_vec = [tag2i[t] for t in sample['question_pos']]
        question_ner_vec = [ner2i[e] for e in sample['question_ner']]
        answer1 = sample['answers'][0].lower()
        answer2 = sample['answers'][1].lower()
        yield (context_vector, context_pos_vec, context_ner_vec, sample['context_em_feature'], \
            question_vector, question_pos_vec, question_ner_vec, sample['question_em_feature'], sample['start_index'], sample['end_index'], \
            sample['context_tokens'], sample['question_tokens'], sample['chosen_answer'], answer1, answer2)


def generate_embeddings(filename, word_dict):
    embeddings = np.random.uniform(-0.25, 0.25, (len(word_dict), 100))
    count = 0
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            tokens = line.split()
            if tokens[0] in word_dict:
                embeddings[word_dict[tokens[0]]] = np.array(list(map(float, tokens[1:])))
                count += 1
    logger.info('Total vocab size %s pre-trained words %s' % (count, len(word_dict)))
    return embeddings

def pad_sequence(sentences, pos, ner, em):
    max_len = max([len(sent) for sent in sentences])
    sent_batch = np.zeros((len(sentences), max_len), dtype=int)
    pos_batch = np.zeros((len(sentences), max_len), dtype=int)
    ner_batch = np.zeros((len(sentences), max_len), dtype=int)
    em_batch = np.zeros((len(sentences), max_len), dtype=int)
    masks = np.zeros((len(sentences), max_len), dtype=int)
    for i, sent in enumerate(sentences):
        sent_batch[i,:len(sent)] = np.array(sent)
        pos_batch[i,:len(pos[i])] = np.array(pos[i])
        ner_batch[i,:len(ner[i])] = np.array(ner[i])
        em_batch[i,:len(em[i])] = np.array(em[i])
        masks[i,:len(sent)] = 1
    return torch.as_tensor(sent_batch), torch.as_tensor(pos_batch), torch.as_tensor(ner_batch), torch.as_tensor(em_batch), torch.as_tensor(masks)

def compute_scores(start, end, context, a1, a2):
    score = 0.0
    for i in range(0, len(start)):
        predicted_span = ' '.join(context[i][start[i]:end[i]+1])
        score += max(rouge.get_scores(predicted_span, a1[i])[0]['rouge-l']['f'], rouge.get_scores(predicted_span, a2[i])[0]['rouge-l']['f'])
    return score


def main(args):
    logger.info('-' * 100)
    logger.info('Loading data')
    with open(args.train_file, 'r') as f:
        training_data = json.load(f)
    with open(args.dev_file, 'r') as f:
        dev_data = json.load(f)

    w2i = defaultdict(lambda: len(w2i))
    PAD = w2i["<PAD>"]
    UNK = w2i["<unk>"]
    tag2i = defaultdict(lambda: len(tag2i))
    pad_tag = tag2i["<PAD>"]
    unk_tag = tag2i["<unk>"]
    ner2i = defaultdict(lambda: len(ner2i))
    pad_ner = ner2i["<PAD>"]
    unk_ner = ner2i["<unk>"]

    logger.info('Converting to index')
    train = convert_data(training_data, w2i, tag2i, ner2i)
    w2i = defaultdict(lambda: UNK, w2i)
    tag2i = defaultdict(lambda: unk_tag, tag2i)
    ner2i = defaultdict(lambda: unk_ner, ner2i)
    dev = convert_data(dev_data, w2i, tag2i, ner2i)

    logger.info('Generating embeddings')
    embeddings = generate_embeddings(args.embedding_file, w2i)
    train_loader = torch.utils.data.DataLoader(list(train), shuffle=True, batch_size=args.train_batch_size, num_workers=4, collate_fn=lambda batch : zip(*batch))
    dev_loader = torch.utils.data.DataLoader(list(dev), batch_size=args.dev_batch_size, num_workers=4, collate_fn=lambda batch : zip(*batch))

    use_cuda = torch.cuda.is_available()
    #model = MnemicReader()
    #optimizer = torch.optim.Adam(model.parameters())
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        #model.cuda()

    logger.info('Start training')
    logger.info('-' * 100)
    global_step = 0
    best = 0.0
    for ITER in range(args.epochs):
        train_loss = 0.0
        train_correct = 0.0
        start = time.time()
        for batch in tqdm.tqdm(train_loader):
            global_step += 1
            c_vec, c_pos, c_ner, c_em, q_vec, q_pos, q_ner, q_em, start, end, c, q, c_a, a1, a2 = batch
            c_vec, c_pos, c_ner, c_em, c_mask = pad_sequence(c_vec, c_pos, c_ner, c_em)
            q_vec, q_pos, q_ner, q_em, q_mask = pad_sequence(q_vec, q_pos, q_ner, q_em)
            start = torch.as_tensor(start)
            end = torch.as_tensor(end)

            if use_cuda:
                c_vec = c_vec.cuda()
                c_pos = c_pos.cuda()
                c_ner = c_ner.cuda()
                c_em = c_em.cuda()
                c_mask = c_mask.cuda()
                q_vec = q_vec.cuda()
                q_pos = q_pos.cuda()
                q_ner = q_ner.cuda()
                q_em = q_em.cuda()
                q_mask = q_mask.cuda()
                start = start.cuda()
                end = end.cuda()

            batch_loss = model(c_vec, c_pos, c_ner, c_em, c_mask, q_vec, q_pos, q_ner, q_em, q_mask, start, end)
            train_loss += batch_loss.item()
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
        logger.info("iter %r: train loss/sample=%.4f, time=%.2fs" % (ITER, train_loss / len(train_loader), time.time() - start))

        rouge_scores = 0.0
        for batch in dev_loader:
            c_vec, c_pos, c_ner, c_em, q_vec, q_pos, q_ner, q_em, start, end, c, q, c_a, a1, a2 = batch
            c_vec, c_pos, c_ner, c_em, c_mask = pad_sequence(c_vec, c_pos, c_ner, c_em)
            q_vec, q_pos, q_ner, q_em, q_mask = pad_sequence(q_vec, q_pos, q_ner, q_em)
            if use_cuda:
                c_vec = c_vec.cuda()
                c_pos = c_pos.cuda()
                c_ner = c_ner.cuda()
                c_em = c_em.cuda()
                c_mask = c_mask.cuda()
                q_vec = q_vec.cuda()
                q_pos = q_pos.cuda()
                q_ner = q_ner.cuda()
                q_em = q_em.cuda()
                q_mask = q_mask.cuda()

            pred_start, pred_end = model.evaluate(c_vec, c_pos, c_ner, c_em, c_mask, q_vec, q_pos, q_ner, q_em, q_mask)

            batch_score = compute_scores(pred_start.tolist(), pred_end.tolist(), c, a1, a2)
            rouge_scores += batch_score
        avg_rouge = rouge_scores / len(dev_loader)
        logger.info("iter %r: dev average rouge score %.4f, time=%.2fs" % (ITER, avg_rouge, time.time() - start))
        if avg_rouge > best:
            best = avg_rouge
            torch.save(model, 'best_model')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Reinforced Mnemonic Reader Model")
    add_arguments(parser)
    args = parser.parse_args()
    logger.setLevel(logging.INFO)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)