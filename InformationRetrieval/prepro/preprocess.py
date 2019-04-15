import numpy as np
import json
import pickle
import os
import sys
from nltk.tokenize import word_tokenize, sent_tokenize
import itertools

def convert_data(datafile, docdir, w2i={}, update_dict=True):	
	if datafile.split('.')[-1] == 'json':
		with open(datafile) as f:
			data = json.load(f)
	if datafile.split('.')[-1] == 'pickle':
		with open(datafile, 'rb') as f:
			data = pickle.load(f)
	context_cache = {}
	for cid in data:
		context = data[cid]['full_text']
		for q in data[cid]['qaps']:
			qwords = word_tokenize(q['question'].lower())	
			awords = [word_tokenize(a.lower()) for a in q['answers']]

			if update_dict:
				qidx = [w2i.setdefault(w, len(w2i)) for w in qwords]		
				a1idx = [w2i.setdefault(w, len(w2i)) for w in awords[0]]
				a2idx = [w2i.setdefault(w, len(w2i)) for w in awords[1]]
			else:
				qidx = [w2i.get(w, w2i['<unk>']) for w in qwords]		
				a1idx = [w2i.get(w, w2i['<unk>']) for w in awords[0]]
				a2idx = [w2i.get(w, w2i['<unk>']) for w in awords[1]]		

			# if q['_id'] in context_cache:
			# 	context = context_cache[q['_id']]
			# else:
			# 	context = os.path.join(docdir, "%s.%s" % (q['_id'], docext))
			# 	if not os.path.exists(context):
			# 		continue				
			# 	with open(context) as f:
			# 		context = json.load(f)
			# 	context_cache[q['_id']] = context

			passages = q["full_text_scores"]
			passage_idxs = []
			passage_scores = []
			for (paraI, sentI, score) in passages:
				passage_scores.append(score)
				sent = context[paraI][sentI][0]
				words = sent.split()				
				if update_dict:
					passage_idxs.append([w2i.setdefault(w, len(w2i)) for w in words])
				else:
					passage_idxs.append([w2i.get(w, w2i['<unk>']) for w in words])
			if len(passage_idxs) > 0:
				yield(qidx, a1idx, a2idx, passage_idxs, passage_scores)

def getIRPretrainData(data_gen, padval=0):
	Qs = []
	Ps = []
	Ns = []
	y = []

	max_lens = [0,0,0]
	count = 0
	for qidx, _, _, didxs, dscores in data_gen:
		count += 1
		# if count == 10:
		# 	break
		sys.stdout.write("\r%d" % count)
		sys.stdout.flush()
		score_sorted_idx = sorted(range(len(dscores)), 
									key=lambda i: dscores[i],
									reverse=True)
		posIdx = score_sorted_idx[:len(score_sorted_idx)//2]
		negIdx = score_sorted_idx[len(posIdx):]

		np.random.shuffle(posIdx)
		np.random.shuffle(negIdx)

		pairs = zip(posIdx, negIdx)

		for pi, ni in pairs:			
			Qs.append(qidx)
			Ps.append(didxs[pi])
			Ns.append(didxs[ni])
			
			y.append(dscores[pi]-dscores[ni])

			max_lens[0] = max(max_lens[0], len(qidx))
			max_lens[1] = max(max_lens[1], len(didxs[pi]))
			max_lens[2] = max(max_lens[2], len(didxs[ni]))

	Qs = np.array(Qs)
	Ps = np.array(Ps)
	Ns = np.array(Ns)
	y = np.array(y)

	return Qs, Ps, Ns, y

if __name__ == '__main__':
	gen = convert_data('/home/kaixinm/NN4NLP-Project/prepro/narrativeqa_train_fulltext_subset.pickle',
					'/home/kaixinm/NN4NLP-Project/prepro/fulltext',
					'txt.json')
	q, p, n, y = getIRPretrainData(gen)
	print(q.shape, p.shape, n.shape, y.shape)
	print(q[0])
	print(p[0])
	print(n[0])
	print(y[0])