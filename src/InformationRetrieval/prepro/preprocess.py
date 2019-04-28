import numpy as np
import json
import pickle
import os
import sys
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.preprocessing import StandardScaler
import itertools

def convert_data(datafile, w2i={}, pos2i={}, update_dict=True):	
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
			qwords = q['question_tokens']
			qpos = q['question_pos']
			awords = [q['answer1_tokens'],
					  q['answer2_tokens']]

			if update_dict:
				qidx = [w2i.setdefault(w, len(w2i)) for w in qwords]
				qposidx = [pos2i.setdefault(p, len(pos2i)) for p in qpos]
				a1idx = [w2i.setdefault(w, len(w2i)) for w in awords[0]]
				a2idx = [w2i.setdefault(w, len(w2i)) for w in awords[1]]
			else:
				qidx = [w2i.get(w, w2i['<unk>']) for w in qwords]
				qposidx = [pos2i.get(p, pos2i['<unk>']) for p in qpos]
				a1idx = [w2i.get(w, w2i['<unk>']) for w in awords[0]]
				a2idx = [w2i.get(w, w2i['<unk>']) for w in awords[1]]		
			qidx = np.stack((qidx, qposidx), axis=1)
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
				words = context[paraI][sentI][1]
				pos = context[paraI][sentI][2]
				if update_dict:
					widx = [w2i.setdefault(w, len(w2i)) for w in words]
					posidx = [pos2i.setdefault(p, len(pos2i)) for p in pos]
				else:
					widx = [w2i.get(w, w2i['<unk>']) for w in words]
					posidx = [pos2i.get(p, pos2i['<unk>']) for p in pos]
				passage_idxs.append(np.stack((widx, posidx), axis=1))
			if len(passage_idxs) > 0:
				yield(qidx, a1idx, a2idx, passage_idxs, passage_scores)

def getIRPretrainData(data_gen):
	Qs = []
	Ps = []
	Ns = []
	y = []

	max_lens = [0,0,0]
	count = 0
	np.random.seed(0)
	for qidx, _, _, didxs, dscores in data_gen:
		count += 1
		if count == 10:
			break
		sys.stdout.write("\r%d" % count)
		sys.stdout.flush()	

		dscores = np.array(dscores)	
		dscores = (dscores-np.min(dscores))
		dscores /= (max(dscores) - min(dscores) + 1e-8)	
			
		score_sorted_idx = sorted(range(len(dscores)), 
									key=lambda i: dscores[i],
									reverse=True)		
		# idxs = np.arange(len(dscores))
		posIdx = score_sorted_idx[:100]
		negIdx = score_sorted_idx[len(posIdx):]

		np.random.shuffle(posIdx)
		np.random.shuffle(negIdx)

		# # pairs = zip(posIdx, negIdx)

		if len(negIdx) == 0:
			continue

		# idxs = np.arange(len(dscores))
		# np.random.shuffle(idxs)
		pairs_gen = itertools.product(posIdx, negIdx)
		# np.random.shuffle(pairs)
		pairs = [next(pairs_gen) for i in range(min(len(posIdx) * len(negIdx), 200))]

		for pi, ni in pairs:			
			Qs.append(np.array(qidx))
			Ps.append(np.array(didxs[pi]))
			Ns.append(np.array(didxs[ni]))
			
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
	gen = convert_data('/home/kaixinm/NN4NLP-Project/prepro/narrativeqa_dev_fulltext.pickle')
	q, p, n, y = getIRPretrainData(gen)
	print(q.shape, p.shape, n.shape, y.shape)	
	print(q[500].shape)
	print(p[500].shape)
	print(n[500].shape)
	print(y[500].shape)