import json
import nltk
from nltk import ngrams
from rouge import Rouge
import operator
import tqdm
import spacy
import en_core_web_sm
from nltk.stem.porter import PorterStemmer
import pickle
nlp = en_core_web_sm.load()
ps = PorterStemmer()

MAX_SPAN_LEN = 6      # what number should we set here ? 
stoplist = set(['the','a','.',',', '...', '..', '....', '.....', '......'])

def get_ngrams(passage, n):
    ngrams_list = []
    for i in range(0, n):
        current = ngrams(passage, i+1)
        for gram in current:
            ngrams_list.append(' '.join(gram))
    return ngrams_list 

def get_em_features(context, question):
    context_base = [ps.stem(w) for w in context]
    question_base =  [ps.stem(w) for w in question]
    context_match = []
    question_match = []
    for w in context_base:
        if w in question_base:
            context_match.append(1)
        else:
            context_match.append(0)
    for w in question_base:
        if w in context_base:
            question_match.append(1)
        else:
            question_match.append(0)
    return context_match, question_match


def process_data(filename):
    with open(filename) as f:
        data = json.load(f)
    rouge = Rouge()
    NER_dict = {}
    POS_dict = {}
    TOK_dict = {}
    for sample in tqdm.tqdm(data):
        context = sample['context'].lower()
        question = sample['question'].lower()
        answer1 = sample['answers'][0].lower()
        answer2 = sample['answers'][1].lower()
        #answer1 = [w.text for w in nlp(answer1)]
        #answer2 = [w.text for w in nlp(answer2)]
        if sample['_id'] not in NER_dict:
            doc = nlp(context)
            context_ner = [w.ent_type_ for w in doc]
            context_tokens = [w.text for w in doc]
            NER_dict[sample['_id']] = context_ner
            TOK_dict[sample['_id']] = context_tokens
        else:
            context_ner = NER_dict[sample['_id']]
            context_tokens = TOK_dict[sample['_id']]
        if sample['_id'] not in POS_dict:
            doc = nltk.pos_tag(context_tokens)
            context_pos = [w[1] for w in doc]
            POS_dict[sample['_id']] = context_pos
        else:
            context_pos = POS_dict[sample['_id']]

        que = nlp(question)
        question_ner = [w.ent_type_ for w in que]
        question_tokens = [w.text for w in que]
        que = nltk.pos_tag(question_tokens)
        question_pos = [w[1] for w in que]
        em_features = get_em_features(context_tokens, question_tokens)
        sample['context_tokens'] = context_tokens
        sample['context_pos'] = context_pos
        sample['context_ner'] = context_ner
        sample['context_em_feature'] = em_features[0]
        sample['question_tokens'] = question_tokens
        sample['question_pos'] = question_pos
        sample['question_ner'] = question_ner
        sample['question_em_feature'] = em_features[1]
        max_span_len = max(len(answer1.split()), len(answer2.split()))
        if answer1 == answer2:
            answers = [answer1]
        else:
            answers = [answer1, answer2]
        candidates = get_ngrams(context_tokens, max_span_len)
        scores = {}
        for span in candidates:
            if span in stoplist:
                continue
            scores[span] = max([rouge.get_scores(span, ans)[0]['rouge-l']['f'] for ans in answers])
        sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
        best_candidate = sorted_scores[0][0].split(' ')
        ans_len = len(best_candidate)
        
        start_index = -1
        end_index = -1
        for i in range(0, len(context_tokens) - ans_len + 1):
            if context_tokens[i: i+ans_len] == best_candidate:
                start_index = i
                end_index = i+ans_len-1
        if start_index == -1 or end_index == -1:
            print ('this should not happen')
        sample['chosen_answer'] = best_candidate
        sample['start_index'] = start_index
        sample['end_index'] = end_index

    new_name = filename.split('.')[0] + '_index.json'
    with open(new_name, "w") as fout:
        json.dump(data, fout, indent=4)

def checkstop(sent):
    for w in sent:
        if w not in stoplist:
            return False
    return True

def extract_span(filename):
    rouge = Rouge()
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    count = 0
    for cid in tqdm.tqdm(data):
        context = data[cid]['full_text']
        for q in data[cid]['qaps']:
            if len(q["full_text_scores"]) == 0:
                count += 1
                print ('fuck', q['question_tokens'], ' '.join(q['answer1_tokens']), ' '.join(q['answer2_tokens']))
                continue
            sorted_scores = sorted(q["full_text_scores"], key=lambda s: s[2],reverse=True)
            paraId = sorted_scores[0][0]
            sentId = sorted_scores[0][1]
            best_sent = context[paraId][sentId][1]
            if checkstop(best_sent) and len(sorted_scores) > 1:
                i = 1
                while checkstop(best_sent) and i < len(sorted_scores):
                    paraId = sorted_scores[i][0]
                    sentId = sorted_scores[i][1]
                    best_sent = context[paraId][sentId][1]
                    i += 1
            if checkstop(best_sent):
                count += 1
                print ('shit', q['question_tokens'])
                continue
            max_span_len = min(len(best_sent), max(len(q['answer1_tokens']), len(q['answer2_tokens'])))
            candidates = get_ngrams(best_sent, max_span_len)
            scores = {}
            answers = [' '.join(q['answer1_tokens']), ' '.join(q['answer2_tokens'])]
            for span in candidates:
                if span in stoplist:
                    continue
                try:
                    scores[span] = max([rouge.get_scores(span, ans)[0]['rouge-l']['f'] for ans in answers])
                except:
                    print ('what the fuck is this', span)
            sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
            best_candidate = sorted_scores[0][0].split(' ')
            ans_len = len(best_candidate)
        
            start_index = -1
            end_index = -1
            for i in range(0, len(best_sent) - ans_len + 1):
                if best_sent[i: i+ans_len] == best_candidate:
                    start_index = i
                    end_index = i+ans_len-1
            if start_index == -1 or end_index == -1:
                print ('this should not happen')
            q['best_span'] = best_candidate
            q['best_indices'] = (paraId, sentId, start_index, end_index)
            #sample['chosen_answer'] = best_candidate
            #sample['start_index'] = start_index
            #sample['end_index'] = end_index
    print (count)

def main():
    process_data('narrativeqa_summary_train.json')
    process_data('narrativeqa_summary_dev.json')
    process_data('narrativeqa_summary_test.json')

def fulltext_spans():
    extract_span('../../prepro/narrativeqa_dev_fulltext.pickle')

def baseline_spans(filename):
    rouge = Rouge()
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    new_data = []
    for sample in tqdm.tqdm(data):
        for ss in sample['context']:
            if len(ss[1]) > 10:
                print (ss[1], len(ss[1]))
        if len(sample['scores']) == 0:
            print (sample['qaps']['question'])
            print (sample['qaps']['_id'])
        raw_scores = [(i, s) for i, s in enumerate(sample['scores'])]
        sorted_scores = sorted(raw_scores, key=lambda s: s[1],reverse=True)
        max_ind = sorted_scores[0][0]
        best_sent = sample['context'][max_ind][1]
        if checkstop(best_sent) and len(sorted_scores) > 1:
            i = 1
            while checkstop(best_sent) and i < len(sorted_scores):
                max_ind = sorted_scores[i][0]
                best_sent = sample['context'][max_ind][1]
                i += 1
        if checkstop(best_sent):
            print ('This sample is fucked')
            continue
        max_span_len = min(len(best_sent), max(len(sample['qaps']['answer1_tokens']), len(sample['qaps']['answer2_tokens'])))
        candidates = get_ngrams(best_sent, max_span_len)
        scores = {}
        answers = [' '.join(sample['qaps']['answer1_tokens']), ' '.join(sample['qaps']['answer2_tokens'])]
        for span in candidates:
            if span in stoplist:
                continue
            try:
                scores[span] = max([rouge.get_scores(span, ans)[0]['rouge-l']['f'] for ans in answers])
            except:
                print ('what the fuck is this', span)
        sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
        best_candidate = sorted_scores[0][0].split(' ')
        ans_len = len(best_candidate)
        
        start_index = -1
        end_index = -1
        for i in range(0, len(best_sent) - ans_len + 1):
            if best_sent[i: i+ans_len] == best_candidate:
                start_index = i
                end_index = i+ans_len-1
        if start_index == -1 or end_index == -1:
            print ('this should not happen')
        newsample = {}
        newsample['_id'] = sample['id']
        newsample['question'] = sample['qaps']['question']
        newsample['question_tokens'] = sample['qaps']['question_tokens']
        newsample['question_pos'] = sample['qaps']['question_pos']
        newsample['question_ner'] = sample['qaps']['question_ner']
        newsample['answers'] = sample['qaps']['answers']
        newsample['answer1_tokens'] = sample['qaps']['answer1_tokens']
        newsample['answer2_tokens'] = sample['qaps']['answer2_tokens']
        newsample['chosen_answer'] = best_candidate
        context_tokens = []
        context_pos = []
        context_ner = []
        for i, sent in enumerate(sample['context']):
            if i == max_ind:
                start_index += len(context_tokens)
                end_index += len(context_tokens)
            context_tokens.extend(sent[1])
            context_pos.extend(sent[2])
            context_ner.extend(sent[3])
        em_features = get_em_features(context_tokens, newsample['question_tokens'])
        newsample['context_em_feature'] = em_features[0]
        newsample['question_em_feature'] = em_features[1]
        newsample['start_index'] = start_index
        newsample['end_index'] = end_index
        assert context_tokens[start_index:end_index+1] == best_candidate
        newsample['context_tokens'] = context_tokens
        newsample['context_pos'] = context_pos
        newsample['context_ner'] = context_ner
        new_data.append(newsample)
    # with open('../../prepro/retrived_dev_index.json', 'w') as fout:
    #     json.dump(new_data, fout)



if __name__ == '__main__':
    #main()
    #fulltext_spans()
    baseline_spans('../../prepro/retrived_dev.pickle')

