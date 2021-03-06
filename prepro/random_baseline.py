import pickle
import operator
from math import log
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from collections import Counter
import tqdm
import json
from rouge import Rouge
import nltk
from nltk import ngrams
stoplist = set(['the','a','.',',', '...', '..', '....', '.....', '......'])
rouge = Rouge()

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
	with open('stopwords.json', 'w') as fout:
		json.dump(stopwords, fout)
	return stopwords

def build_chunks(context, max_len=100):
	words_buff = []
	pos_buff = []
	ner_buff = []
	flat_context = []
	scores_map = {}
	count = 0
	for para in context:
		for sent in para:
			words = sent[1]
			pos = sent[2]
			ner = sent[3]
			if len(words) > max_len:
				if len(words_buff) != 0:
					flat_context.append((words_buff, pos_buff, ner_buff))
					words_buff = []
					pos_buff = []
					ner_buff = []
				curr = 0
				while curr+max_len < len(words):
					flat_context.append((words[curr:curr+max_len], pos[curr:curr+max_len], ner[curr:curr+max_len]))
					curr += max_len
				assert len(words_buff) == 0
				words_buff.extend(words[curr:])
				pos_buff.extend(pos[curr:])
				ner_buff.extend(ner[curr:])
			elif len(words) + len(words_buff) > max_len:
				flat_context.append((words_buff, pos_buff, ner_buff))
				words_buff = words
				pos_buff = pos
				ner_buff = ner
			else:
				words_buff.extend(words)
				pos_buff.extend(pos)
				ner_buff.extend(ner)
			scores_map[count] = len(flat_context)
			count += 1
	if len(words_buff) > 0:
		flat_context.append((words_buff, pos_buff, ner_buff))
	return flat_context, scores_map

def add_scores(idx, all_scores, scores_map):
	total = 0.0
	for k, v in scores_map.items():
		if v == idx:
			total += all_scores[k]
	return total 

def get_ngrams(passage, n):
	ngrams_list = []
	for i in range(0, n):
		current = ngrams(passage, i+1)
		for gram in current:
			ngrams_list.append(' '.join(gram))
	return ngrams_list

def compute_rouge(candidates, context_tokens, answers):
	scores = {}
	for span in candidates:
		if span in stoplist:
			continue
		try:
			scores[span] = max([rouge.get_scores(span, ans)[0]['rouge-l']['f'] for ans in answers])
		except:
			print ('what the fuck is this', span)
	sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
	best_candidate = sorted_scores[0][0].split(' ')
	best_rouge = sorted_scores[0][1]
	ans_len = len(best_candidate)
	
	start_index = -1
	end_index = -1
	for i in range(0, len(context_tokens) - ans_len + 1):
		if context_tokens[i: i+ans_len] == best_candidate:
			start_index = i
			end_index = i+ans_len-1
	if start_index == -1 or end_index == -1:
		print ('this should not happen')
	return best_candidate, start_index, end_index, best_rouge

def retrieve_chunks(data, stopwords, outname, use_answer=True, num_chunks=10, target=None):
	newdata = {}
	document_count = 0
	for cid in tqdm.tqdm(data):
		context = data[cid]['full_text']
		if len(context) == 0:
			print (cid)
			continue
		passage_context, scores_map = build_chunks(context, max_len=100)
		count = 0 
		Doc_frec = {}
		dlt = {}
		for con in passage_context:
			for w in con[0]:
				w = ps.stem(w)
				if w in Doc_frec:
					if count in Doc_frec[w]:
						Doc_frec[w][count] += 1
					else:
						Doc_frec[w][count] = 1
				else:
					Doc_frec[w] = {count:1}
			dlt[count] = len(con[0])
			count += 1
		avg = 0
		for k, v in dlt.items():
			avg += v
		avg /= float(len(dlt))
		newdata[cid] = {}
		newdata[cid]['full_text'] = passage_context
		newdata[cid]['qaps'] = []
		for q in data[cid]['qaps']:
			# passage_scores = []
			# score_dict = {}
			# for (paraI, sentI, score) in q["full_text_scores"]:
			# 	score_dict[(paraI, sentI)] = score
			# for i, para in enumerate(context):
			# 	for j,sent in enumerate(para):
			# 		if (i, j) in score_dict:
			# 			passage_scores.append(score_dict[(i, j)])
			# 		else:
			# 			passage_scores.append(0.0)
			query_result = {}
			if use_answer:
				if len(q['answer1_tokens']) > len(q['answer2_tokens']):
					qwords = q['answer1_tokens']
				else:
					qwords = q['answer2_tokens']
			else:
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
			passage_labels = []
			max_span_len = max(len(q['answer1_tokens']), len(q['answer2_tokens']))
			answer1 = q['answers'][0].lower()
			answer2 = q['answers'][1].lower()
			if answer1 == answer2:
				answers = [answer1]
			else:
				answers = [answer1, answer2]
			for idx, s in sorted_score[0:num_chunks]:
				candidates = get_ngrams(passage_context[idx][0], max_span_len)
				best_span, start_idx, end_idx, best_rouge = compute_rouge(candidates, passage_context[idx][0], answers)
				passage_labels.append((idx, best_span, start_idx, end_idx, best_rouge, 1))
				
			if len(passage_labels) == 0:
				print ('none found')
				new_context = passage_context[0:num_chunks]
				for i, con in enumerate(new_context):
					candidates = get_ngrams(con[0], max_span_len)
					best_span, start_idx, end_idx, best_rouge = compute_rouge(candidates, con[0], answers)
					passage_labels.append((i, best_span, start_idx, end_idx, best_rouge, 0))
			q['passage_labels'] = passage_labels
			del q["full_text_scores"]
			newdata[cid]['qaps'].append(q)
		document_count += 1
		if target == document_count:
			break	
	with open(outname, 'wb') as f:
		pickle.dump(newdata, f)

def retrieve(data, stopwords, outname, use_answer=False, num_sent=50):
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
				assert len(sent[1]) == len(sent[2])
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
			if use_answer:
				if len(q['answer1_tokens']) > len(q['answer2_tokens']):
					qwords = q['answer1_tokens']
				else:
					qwords = q['answer2_tokens']
			else:
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
			BM25_scores = []
			for idx, s in sorted_score[0:num_sent]:
				new_context.append(passage_context[idx])
				context_scores.append(passage_scores[idx])
				BM25_scores.append(s)
			if len(new_context) == 0:
				new_context = passage_context[0:num_sent]
				context_scores = passage_scores[0:num_sent]
				BM25_scores = [0.0 for _ in range(len(new_context))]
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
			sample['BM25_scores'] = BM25_scores
			newdata.append(sample)
	#with open(outname, 'wb') as f:
	#	pickle.dump(newdata, f)

if __name__ == '__main__':
	stopwords = []
	with open('narrativeqa_train_fulltext_wostop.pickle', 'rb') as f:
		train_data = pickle.load(f)
	stopwords = get_stopwords(train_data)
	with open('narrativeqa_dev_fulltext_wostop.pickle', 'rb') as f:
		dev_data = pickle.load(f)
	#retrieve(train_data, stopwords, 'BM25_train_followpaper.pickle', use_answer=True, num_sent=50)
	#retrieve(dev_data, stopwords, 'whatever.pickle', use_answer=False, num_sent=500)
	retrieve_chunks(train_data, stopwords, 'e2e_train_subset.pickle', target=100)
	#retrieve_chunks(dev_data, stopwords, 'e2e_dev_subset.pickle')


