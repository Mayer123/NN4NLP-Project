import pickle
import operator
from math import log
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from collections import Counter
import tqdm

k1 = 1.2
k2 = 100
b = 0.75
R = 0.0

def score_BM25(n, f, qf, r, N, dl, avdl):
	K = compute_K(dl, avdl)
	first = log( ( (r + 0.5) / (R - r + 0.5) ) / ( (n - r + 0.5) / (N - n - R + r + 0.5)) )
	second = ((k1 + 1) * f) / (K + f)
	third = ((k2+1) * qf) / (k2 + qf)
	return first * second * third

def compute_K(dl, avdl):
	return k1 * ((1-b) + b * (float(dl)/float(avdl)) )

def get_stopwords(data, top=50):
	w2i = Counter()
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
		if cid in context_cache:
			continue
		else:
			for para in context:
				for sent in para:
					for w in sent[1]:
						w2i[w] += 1
	stopwords = []
	for i, (k, v) in enumerate(w2i.most_common(top)):
		stopwords.append(k)
	return stopwords

def retrieve(data, stopwords, outname, throw=False):
	newdata = []
	for cid in tqdm.tqdm(data):
		context = data[cid]['full_text']
		if len(context)== 0:
			print (cid)
			continue
		count = 0 
		Doc_frec = {}
		dlt = {}
		passage_context = []
		for para in context:
			for sent in para:
				passage_context.append(sent)	
				for w in sent[1]:
					w = ps.stem(w)
					if w in Doc_frec:
						if count in Doc_frec[w]:
							Doc_frec[w][count] += 1
						else:
							Doc_frec[w][count] = 1
					else:
						Doc_frec[w] = {count:1}
				dlt[count] = len(sent[1])
				count += 1
		# if len(dlt) == 0:
		# 	print (cid)
		# 	if throw:
		# 		continue
		# 	else:
		# 		assert len(passage_context) == 0
		# 		passage_context = [('some random stuff', ['some', 'random', 'stuff'], ['unk', 'unk', 'unk'], ['', '', ''])]
		# 		dlt = {0:3}

		avg = 0
		for k, v in dlt.items():
			avg += v
		avg /= float(len(dlt))
		for q in data[cid]['qaps']:
			passage_scores = []
			score_dict = {}
			for (paraI, sentI, score) in q["full_text_scores"]:
				score_dict[(paraI, sentI)] = score
			for i, para in enumerate(context):
				for j,sent in enumerate(para):
					if (i, j) in score_dict:
						passage_scores.append(score_dict[(i, j)])
					else:
						passage_scores.append(0.0)
			query_result = {}
			qwords = q['question_tokens']
			for w in qwords:
				w = ps.stem(w)
				if w in stopwords:
					continue
				if w in Doc_frec:
					doc_dict = Doc_frec[w]
					for docid, freq in doc_dict.items():
						score = score_BM25(n=len(doc_dict), f=freq, qf=1, r=0, N=len(dlt),
									   dl=dlt[docid], avdl=avg) 
						if docid in query_result:
							query_result[docid] += score
						else:
							query_result[docid] = score
			sorted_score = sorted(query_result.items(), key=operator.itemgetter(1), reverse=True)
			new_context = []
			context_scores = []
			for idx, s in sorted_score[0:50]:
				new_context.append(passage_context[idx])
				context_scores.append(passage_scores[idx])
			if len(new_context) == 0:
				new_context = passage_context[0:50]
				context_scores = passage_scores[0:50]
			sample = {}
			sample['id'] = q['_id']
			qo = {}
			for k, v in q.items():
				if k == 'full_text_scores':
					continue
				qo[k] = v
			sample['qaps'] = qo
			# sample['answers'] = q['answers']
			# sample['question_tokens'] = q['question_tokens']
			# sample['question_pos'] = q['question_pos']
			# sample['question_ner'] = q['question_ner']
			# sample['answer1_tokens'] = q['answer1_tokens']
			# sample['answer2_tokens'] = q['answer2_tokens']
			sample['context'] = new_context
			sample['scores'] = context_scores
			newdata.append(sample)
	with open(outname, 'wb') as f:
		pickle.dump(newdata, f)

if __name__ == '__main__':
	stopwords = []
	with open('narrativeqa_train_fulltext_redo.pickle', 'rb') as f:
		train_data = pickle.load(f)
	stopwords = get_stopwords(train_data)
	with open('narrativeqa_dev_fulltext_redo.pickle', 'rb') as f:
		dev_data = pickle.load(f)
	retrieve(train_data, stopwords, 'BM25_train.pickle', throw=True)
	retrieve(dev_data, stopwords, 'BM25_dev.pickle', throw=False)

