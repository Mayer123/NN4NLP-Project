from collections import defaultdict, Counter
import numpy as np
import tqdm
import random
import torch
import re
import json

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

class FulltextDataset(torch.utils.data.Dataset):

    def __init__(self, data, batch_size):
        self.data = []
        batch = []
        count = 0
        for sample in data:  
            if len(batch) == 0:
                batch.append(sample)
            elif sample[0] != batch[-1][0]:
                self.data.append(batch)
                batch = [sample]
            else:
                batch.append(sample)

            if len(batch) == batch_size:
                self.data.append(batch)
                batch = []
        if len(batch) != 0:
            self.data.append(batch)
            batch = []
        print (len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        batch = self.data[idx]
        return batch

def mCollateFn(batch):
    Qwords = []
    Qtags = []
    Qners = []
    Qchars = []
    A1 = []
    A2 = []
    Passages = []
    Passagestags = []
    Passageswords = []
    assert len(batch) == 1
    batch = batch[0]
    idset = []
    for i, (cid,qw,qt,qn,qc,a1,a2,p) in enumerate(batch):
        #for i, (cid,qw,qt,qn,qc,a1,a2,p) in enumerate(sample):
        idset.append(cid)
        Qwords.append(qw)
        Qtags.append(qt)
        Qners.append(qn)
        Qchars.append(qc)
        A1.append(a1)
        A2.append(a2)
        if i == 0:
            Passages = [sent[0] for sent in p]
            Passagestags = [sent[1] for sent in p]
            Passageswords = [sent[4] for sent in p]
    assert len(set(idset)) == 1
    #max_q_len = max([len(q) for q in Qwords])
    #max_p_len = max([len(p) for p in Passages])
    #max_s_len = max([len(s) for s in Passages])
    #max_a_len = max([len(a) for a in A1])

    qlens = torch.tensor([len(q) for q in Qwords]).long()
    #plens = torch.tensor([len(p) for p in Passages]).long()
    slens = torch.tensor([len(s) for s in Passages]).long()   # The assumption is that passages in one batch are all the same
    alens = torch.tensor([len(a) for a in A1]).long()
    max_q_len = torch.max(qlens)
    max_s_len = torch.max(slens)
    max_a_len = torch.max(alens)
    Qtensor = torch.zeros(len(batch), max_q_len).long()
    Qtagtensor = torch.zeros(len(batch), max_q_len).long()
    Ptensor = torch.zeros(len(Passages), max_s_len).long()
    Ptagtensor = torch.zeros(len(Passages), max_s_len).long()
    A1tensor = torch.zeros(len(batch), max_a_len).long() 
    A2tensor = torch.zeros(len(batch), max_a_len).long()    
    for i in range(len(batch)):
        Qtensor[i, :qlens[i]] = torch.tensor(Qwords[i])
        Qtagtensor[i, :qlens[i]] = torch.tensor(Qtags[i])
        A1tensor[i, :alens[i]] = torch.tensor(A1[i])
        A2tensor[i, :alens[i]] = torch.tensor(A2[i])
        if i == 0:
            for j in range(len(Passages)):
                Ptensor[j,:slens[j]] = torch.tensor(Passages[j])
                Ptagtensor[j,:slens[j]] = torch.tensor(Passagestags[j])
    Ptensor = torch.cat([Ptensor.unsqueeze(2), Ptagtensor.unsqueeze(2)], dim=2)
    Qtensor = torch.cat([Qtensor.unsqueeze(2), Qtagtensor.unsqueeze(2)], dim=2)
    return Qtensor, Ptensor, A1tensor, A2tensor, qlens, slens, alens, A1, A2, Passageswords

def build_fulltext_dicts(data, min_occur=100):
    w2i = Counter()
    tag2i = Counter()
    ner2i = Counter()
    c2i = Counter()
    context_cache = {}
    for cid in tqdm.tqdm(data):
        context = data[cid]['full_text']
        for q in data[cid]['qaps']:
            for w in q['question_tokens']:
                w2i[w] += 1
            for w in q['answer1_tokens']:
                w2i[w] += 1
            for w in q['answer2_tokens']:
                w2i[w] += 1
            for w in q['question_pos']:
                tag2i[w] += 1
        if cid in context_cache:
            continue
        else:
            for para in context:
                for sent in para:
                    for w in sent[1]:
                        w2i[w] += 1
                        tag2i[w] += 1
    word_dict = {}
    tag_dict = {}
    ner_dict = {}
    char_dict = {}
    common_vocab = {}
    for i, (k, v) in enumerate(w2i.most_common()):
        if v >= min_occur:
            common_vocab[k] = i + 4                         # <SOS> for 2 <EOS> for 3
        word_dict[k] = i + 4 
    for i, (k, v) in enumerate(tag2i.most_common()):
        tag_dict[k] = i + 2

    for k,v in common_vocab.items():
        assert v == word_dict[k]
    return word_dict, tag_dict, ner_dict, char_dict, common_vocab


def convert_fulltext(data, w2i, tag2i, ner2i, c2i, common_vocab, max_len=None, build_chunks=False):
    context_cache = {}
    for cid in tqdm.tqdm(data):
        context = data[cid]['full_text']
        for q in data[cid]['qaps']:
            qners = []
            qchars = []
            qidx = [w2i.get(w, w2i['<unk>']) for w in q['question_tokens']]       
            qtags = [tag2i.get(w, tag2i['<unk>']) for w in q['question_pos']]
            if random.random() > 0.5:
                real_aidx = [w2i.get(w, w2i['<unk>']) for w in q['answer1_tokens']]
                target_aidx = [common_vocab.get(w, w2i['<unk>']) for w in q['answer1_tokens']]  
            else:
                real_aidx = [w2i.get(w, w2i['<unk>']) for w in q['answer2_tokens']]
                target_aidx = [common_vocab.get(w, w2i['<unk>']) for w in q['answer2_tokens']]  
            #qners = [ner2i.get(w, ner2i['<unk>']) for w in q['question_ner']]
            #qchars = [[c2i.get(c, c2i['<unk>']) for c in w] for w in q['question_tokens']]
            if cid in context_cache:
                passage_idxs = context_cache[cid]
            else:
                passage_idxs = []
                if not build_chunks:
                    for para in context:
                        for sent in para:
                            words = sent[1]
                            pos = sent[2]
                            ner = sent[3]
                            neridx = []
                            charidx = []
                            widx = [w2i.get(w, w2i['<unk>']) for w in words]
                            posidx = [tag2i.get(p, tag2i['<unk>']) for p in pos]
                            #neridx = [ner2i.get(n, ner2i['<unk>']) for n in ner]
                            #charidx = [[c2i.get(c, c2i['<unk>']) for c in w] for w in words]
                            if max_len is not None and len(widx) > max_len:
                                curr = 0
                                while curr+max_len < len(widx):
                                    passage_idxs.append((widx[curr:curr+max_len], posidx[curr:curr+max_len], [], [], words[curr:curr+max_len]))
                                    curr += max_len
                                passage_idxs.append((widx[curr:], posidx[curr:], [], [], words[curr:]))
                            else:
                                passage_idxs.append((widx, posidx, neridx, charidx, words))
                else:
                    assert max_len is not None
                    widx = []
                    posidx = [] 
                    neridx = []
                    charidx = []
                    words_buff = []
                    for para in context:
                        for sent in para:
                            words = sent[1]
                            pos = sent[2]
                            ner = sent[3]
                            tmp_w = [w2i.get(w, w2i['<unk>']) for w in words]
                            tmp_pos = [tag2i.get(p, tag2i['<unk>']) for p in pos]
                            if len(tmp_w) > max_len:
                                if len(widx) != 0:
                                    passage_idxs.append((widx, posidx, neridx, charidx, words_buff))
                                    widx = []
                                    posidx = []
                                    words_buff = []
                                curr = 0
                                while curr+max_len < len(tmp_w):
                                    passage_idxs.append((tmp_w[curr:curr+max_len], tmp_pos[curr:curr+max_len], [], [], words[curr:curr+max_len]))
                                    curr += max_len
                                assert len(widx) == 0
                                widx.extend(tmp_w[curr:])
                                posidx.extend(tmp_pos[curr:])
                                words_buff.extend(words[curr:])
                            elif len(tmp_w) + len(widx) > max_len:
                                passage_idxs.append((widx, posidx, neridx, charidx, words_buff))
                                widx = tmp_w
                                posidx = tmp_pos
                                words_buff = words
                            else:
                                widx.extend(tmp_w)
                                posidx.extend(tmp_pos)
                                words_buff.extend(words)
                    if len(widx) > 0:
                        passage_idxs.append((widx, posidx, neridx, charidx, words_buff))     
                context_cache[cid] = passage_idxs
            if len(passage_idxs) > 0:
                yield(cid, qidx, qtags, qners, qchars, real_aidx, target_aidx, passage_idxs)


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
        if v >= 0:
            common_vocab[k] = i + 4                         # <SOS> for 2 <EOS> for 3
    for i, (k, v) in enumerate(w2i.most_common()):
        word_dict[k] = i + 4                         # <SOS> for 2 <EOS> for 3
    for i, (k, v) in enumerate(tag2i.most_common()):
        tag_dict[k] = i + 2
    for i, (k, v) in enumerate(ner2i.most_common()):
        ner_dict[k] = i + 2
    for i, (k, v) in enumerate(c2i.most_common()):
        char_dict[k] = i + 2

    for k,v in common_vocab.items():
        assert v == word_dict[k]
    # 0 for padding and 1 for unk
    # for d in ['word_dict', 'tag_dict', 'ner_dict', 'char_dict', 'common_vocab']:
    #     with open('../../prepro/dicts/%s.json'%d, 'w') as f:
    #         json.dump(locals()[d], f)

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
        answer_vector = [common_vocab[w] if w in common_vocab else 1 for w in answer_tokens]+[3]
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

def reset_embeddings(word_embeddings, fixed_embeddings, trained_idx):
    word_embeddings.weight.data[trained_idx] = torch.FloatTensor(fixed_embeddings[trained_idx]).to(word_embeddings.weight.data.device)
    return 
