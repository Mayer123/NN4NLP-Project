import csv
import os.path
from os import listdir
from os.path import isfile, join
import json
from nltk.tokenize import sent_tokenize
import nltk
import tqdm
import spacy
import en_core_web_sm
from nltk.stem.porter import PorterStemmer
from CSMrouge import RRRouge
import multiprocessing as mp
from collections import Counter
import pickle
import string
import re
rouge = RRRouge()
nlp = en_core_web_sm.load()
ps = PorterStemmer()
with open('stopwords.json', 'r') as f:
    stopwords = json.load(f)
def check_file_exist(path):
    files = [f.split('.')[0] for f in listdir(path) if isfile(join(path, f)) and f[-4:] == '.txt']
    print (len(files))
    with open("/share/workhorse2/hyd/narrativeQA/narrativeqa/qaps.csv", 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        count = 0
        for line in csv_reader:
            if count == 0:
                count += 1
                print (line)
                continue
            else:
                count += 1 
                if line[0] not in files:
                    print ("fuck")
                    break
        print (count)

def reading_books(path, name):
    # if name == '37c11f984cb14401d85abfc20e8305ca7a472c9f.content':
    #     name = '37c11f984cb14401d85abfc20e8305ca7a472c9f.txt'
    if isfile('fulltext_tokenized/'+name+'.json'):
        with open('fulltext_tokenized/'+name+'.json', 'r') as fin:
            new_content = json.load(fin)
            return new_content
    filename = join(path, name)
    content = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                if line.strip() == '\n' or line.strip() == '':
                    continue
                content.append(' '.join(line.strip().split()))
    except:
        print (filename)
        with open(filename, 'r', encoding='latin-1') as f:
            for line in f:
                if line.strip() == '\n' or line.strip() == '':
                    continue
                content.append(' '.join(line.strip().split()))
    new_content = []
    for para in content:
        new_para = []
        para = para.lower().encode('utf-8').decode('utf-8', errors='ignore')
        sents = sent_tokenize(para)
        for sent in sents:
            doc = nlp(sent)
            sent_ner = [w.ent_type_ for w in doc]
            tokens = [w.text for w in doc]
            doc = nltk.pos_tag(tokens)
            sent_pos = [w[1] for w in doc]
            new_para.append([sent, tokens, sent_pos, sent_ner])
        new_content.append(new_para)
    with open('fulltext_tokenized/'+name+'.json', 'w') as fout:
       json.dump(new_content, fout)
    return new_content

def _normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace.
  Directly copied from official SQuAD eval script, SHOULD NOT BE MODIFIED.
  Args:
    s: Input text.
  Returns:
    Normalized text.
  """

  def remove_articles(text):
    return re.sub(r'\b(a|an|the)\b', ' ', text)

  def white_space_fix(text):
    return ' '.join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

def build_sample(bundle):
    i, content, a1, a2 = bundle
    #print (a1, a2)
    a1 = a1.lower()
    a2 = a2.lower()
    count  = 0
    scores = []
    for j, para in enumerate(content):
        for k, sent in enumerate(para):
            cand = ' '.join([w for w in sent[1] if w not in stopwords])
            curr = rouge.calc_score([_normalize_answer(cand)], [_normalize_answer(a1), _normalize_answer(a2)])
            count += 1
            if curr != 0:
                scores.append((j, k, curr))
    #print (count)
    return i, scores


def process_data(path, path1):
    books = [f.split('.')[0] for f in listdir(path) if isfile(join(path, f)) and f[-4:] == '.txt']
    books_content = [reading_books(path, b+'.txt') for b in tqdm.tqdm(books)]
    #with open('books.pickle', 'wb') as f:
    #    pickle.dump(books_content, f)
    scripts = [f.split('.')[0] for f in listdir(path1) if isfile(join(path1, f)) and (f[-8:] == '.content' or f[-4:] == '.txt')]
    scripts_content = [reading_books(path1, s+'.content') for s in tqdm.tqdm(scripts)]
    #with open('scripts.pickle', 'wb') as f:
    #    pickle.dump(scripts_content, f)
    print (len(books), len(scripts))
    file_dict = Counter()
    work_queue = []
    with open("/share/workhorse2/hyd/narrativeQA/narrativeqa/qaps.csv", 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        count = 0
        for line in csv_reader:
            if count == 0:
                count += 1
                print (line)
                continue
            else:
                file_dict[line[0]] += 1
                count += 1 
                sample = {}
                sample['_id'] = line[0]
                sample['question'] = line[5].lower()
                sample['answers'] = [line[6].lower(), line[7].lower()]
                que = nlp(sample['question'])
                question_ner = [w.ent_type_ for w in que]
                question_tokens = [w.text for w in que]
                que = nltk.pos_tag(question_tokens)
                question_pos = [w[1] for w in que]
                sample['question_tokens'] = question_tokens
                sample['question_pos'] = question_pos
                sample['question_ner'] = question_ner
                sample['answer1_tokens'] = [w.text for w in nlp(sample['answers'][0])]
                sample['answer2_tokens'] = [w.text for w in nlp(sample['answers'][1])]
                sample['set'] = line[1]
                if line[0] in books:
                    sample['source'] = 'books'
                    idx = books.index(line[0])
                    work_queue.append((books_content[idx], sample))
                elif line[0] in scripts:
                    sample['source'] = 'scripts'
                    idx = scripts.index(line[0])
                    work_queue.append((scripts_content[idx], sample))
                else:
                    print (line[0])
                    print ("What the fuck")
    print (len(file_dict))
    print (len(work_queue))
    finished_data = [None for i in range(len(work_queue))]
    input_data = [(i, d[0], d[1]['answers'][0], d[1]['answers'][0]) for i, d in enumerate(work_queue)]

    pool = mp.Pool(16)
    for (i, d) in tqdm.tqdm(pool.imap_unordered(build_sample, input_data, 100), total=len(input_data)):
        finished_data[i] = (d, work_queue[i]) 

    train = {}
    dev = {}
    test = {}
    for score, sample in finished_data:
        sample[1]['full_text_scores'] = score

        if sample[1]['set'] == 'train':
            if sample[1]['_id'] in train:
                train[sample[1]['_id']]['qaps'].append(sample[1])
            else:
                train[sample[1]['_id']] = {'full_text': sample[0], 'qaps':[sample[1]]}
        elif sample[1]['set'] == 'valid':
            if sample[1]['_id'] in dev:
                dev[sample[1]['_id']]['qaps'].append(sample[1])
            else:
                dev[sample[1]['_id']] = {'full_text': sample[0], 'qaps':[sample[1]]}
        elif sample[1]['set'] == 'test':
            if sample[1]['_id'] in test:
                test[sample[1]['_id']]['qaps'].append(sample[1])
            else:
                test[sample[1]['_id']] = {'full_text': sample[0], 'qaps':[sample[1]]}
        else:
            print ('what the fuck')

    with open('narrativeqa_train_fulltext_wostop.pickle', 'wb') as f:
        pickle.dump(train, f)
    with open('narrativeqa_dev_fulltext_wostop.pickle', 'wb') as f:
        pickle.dump(dev, f)
    with open('narrativeqa_test_fulltext_wostop.pickle', 'wb') as f:
        pickle.dump(test, f)
    # with open('narrativeqa_train_fulltext.json', 'w') as f:
    #     json.dump(train, f)
    # with open('narrativeqa_dev_fulltext.json', 'w') as f:
    #     json.dump(dev, f)
    # with open('narrativeqa_test_fulltext.json', 'w') as f:
    #     json.dump(test, f)
    
def merge_qaps(file1, file2):
    with open(file1, 'rb') as f:
        full_text = pickle.load(f)
    with open(file2, 'rb') as f:
        summary = json.load(f)
    for sample in tqdm.tqdm(summary):
        found = 0
        #print (len(target['qaps']))
        for qa in full_text[sample['_id']]['qaps']:
            #print (qa['question'])
            if qa['question'] == sample['question']:
                found += 1
                qa['question_tokens'] = sample['question_tokens']
                qa['question_pos'] = sample['question_pos']
                qa['question_ner'] = sample['question_ner']
            if 'answer1_tokens' not in qa:
                qa['answer1_tokens'] = [w.text for w in nlp(qa['answers'][0])]
                qa['answer2_tokens'] = [w.text for w in nlp(qa['answers'][1])]
        if found == 0:
            print (found)
            print (sample['_id'], sample['question'])
            print (sample['answers'])
            exit(0)

    with open('narrativeqa_train_fulltext.pickle', 'wb') as f:
        pickle.dump(full_text, f)

if __name__ == '__main__':
    #check_file_exist('/share/workhorse2/hyd/narrativeQA/data/narrativeqa/cleaned')
    process_data('/share/workhorse2/hyd/narrativeQA/data/narrativeqa/cleaned-books', '/share/workhorse2/hyd/narrativeQA/data/narrativeqa/cleaned-scripts')
    #merge_qaps('narrativeqa_train_fulltext_redo.pickle', 'narrativeqa_summary_train_index.json')





