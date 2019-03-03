import json
import nltk
from nltk import ngrams
from rouge import Rouge
import operator
import tqdm
import spacy
import en_core_web_sm
from nltk.stem.porter import PorterStemmer
nlp = en_core_web_sm.load()
ps = PorterStemmer()

MAX_SPAN_LEN = 6      # what number should we set here ? 
stoplist = set(['the','a','.',',', '...'])

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
        for i in range(0, len(context_tokens) - ans_len):
            if context_tokens[i: i+ans_len] == best_candidate:
                start_index = i
                end_index = i+ans_len-1
        if start_index == -1 or end_index == -1:
            print ('this should not happen')
        sample['chosen_answer'] = best_candidate
        sample['start_index'] = start_index
        sample['end_index'] = end_index

    new_name = filename.split('.') + '_index.json'
    with open(new_name, "w") as fout:
        json.dump(data, fout, indent=4)
      

def main():
    process_data('narrative_summary_train.json')
    process_data('narrative_summary_dev.json')
    process_data('narrative_summary_test.json')

if __name__ == '__main__':
    main()