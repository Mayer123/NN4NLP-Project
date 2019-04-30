import numpy as np
import json
import pickle
import os
import sys
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.preprocessing import StandardScaler
import itertools

def convert_data(datafile, w2i={}, pos2i={}, update_dict=True, all_sents=False):	
	if datafile.split('.')[-1] == 'json':
		with open(datafile) as f:
			data = json.load(f)
	if datafile.split('.')[-1] == 'pickle':
		with open(datafile, 'rb') as f:
			data = pickle.load(f)
	context_cache = {}

	np.random.seed(0)
	cids = list(data.keys())
	np.random.shuffle(cids)
	for cid in cids:
		context = data[cid]['full_text']
		questions = data[cid]['qaps']
		np.random.shuffle(questions)
		for q in questions:
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
			
			passages = q["full_text_scores"]
			non_zero_rouge_sents = {}
			for (paraI, sentI, score) in passages:				
				non_zero_rouge_sents[(paraI, sentI)] = score

			passage_idxs = []
			passage_scores = []
			passage_context = []

			if all_sents:				
				for para_i in range(len(context)):
					para = context[para_i]
					for sent_i in range(len(para)):
						sent = para[sent_i]
						passage_context.append(sent)

						words = sent[1]
						pos = sent[2]

						if update_dict:
							widx = [w2i.setdefault(w, len(w2i)) for w in words]
							posidx = [pos2i.setdefault(p, len(pos2i)) for p in pos]
						else:
							widx = [w2i.get(w, w2i['<unk>']) for w in words]
							posidx = [pos2i.get(p, pos2i['<unk>']) for p in pos]
						
						passage_idxs.append(np.stack((widx, posidx), axis=1))

						if (para_i, sent_i) in non_zero_rouge_sents:
							passage_scores.append(non_zero_rouge_sents[(para_i, sent_i)])
						else:
							passage_scores.append(0.0)
			else:
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

			qo = {}
			for k, v in q.items():
				if k == 'full_text_scores':
					continue
				qo[k] = v
			if len(passage_idxs) > 0:
				yield(cid, qo, passage_context, qidx, a1idx, a2idx, passage_idxs, passage_scores)

def getEvalData(data_gen):
	Qs = []
	Cs = []
	y = []

	for _, _, _, qidx, _, _, didxs, dscores in data_gen:
		Qs.append(qidx)
		Cs.append(didxs)
		y.append(dscores)
	Qs = np.array(Qs)
	Cs = np.array(Cs)
	y = np.array(y)

	return Qs, Cs, y

def getIRPretrainData(data_gen, pos_thres=50, npairs=10, nques=-1):
	Qs = []
	Ps = []
	Ns = []
	y = []

	max_lens = [0,0,0]
	count = 0
	np.random.seed(0)
	for  _, _, _, qidx, _, _, didxs, dscores in data_gen:
		count += 1
		if count == nques:
			break
		sys.stdout.write("\r%d" % count)
		sys.stdout.flush()	

		# print('dscores range:', max(dscores), min(dscores))

		dscores = np.array(dscores)
		dscores = (dscores-np.min(dscores))
		dscores /= (max(dscores) - min(dscores) + 1e-8)				

		score_sorted_idx = sorted(range(len(dscores)), 
									key=lambda i: dscores[i],
									reverse=True)

		thresh = dscores[score_sorted_idx[len(score_sorted_idx) // 100]]
		# print('thresh:', thresh)
		
		# posIdx = score_sorted_idx[:pos_thres]
		# negIdx = score_sorted_idx[len(posIdx):]		
		posIdx = np.arange(len(dscores))[dscores >= thresh]
		negIdx = np.arange(len(dscores))[dscores < thresh]

		# # pairs = zip(posIdx, negIdx)

		if len(negIdx) == 0 or len(posIdx) == 0:
			continue

		# print(max(dscores[posIdx]), min(dscores[posIdx]))
		# print(max(dscores[negIdx]), min(dscores[negIdx]))

		np.random.shuffle(posIdx)
		np.random.shuffle(negIdx)

		k = int(round(np.sqrt(npairs)))

		posIdx = posIdx[:k]
		negIdx = negIdx[:k]		

		# idxs = np.arange(len(dscores))
		# np.random.shuffle(idxs)
		pairs_gen = itertools.product(posIdx, negIdx)
		pairs = list(pairs_gen)
		# np.random.shuffle(pairs)
		# pairs = [next(pairs_gen) for i in range(min(len(posIdx) * len(negIdx), npairs))]

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
	gen = convert_data('/home/kaixinm/NN4NLP-Project/prepro/narrativeqa_dev_fulltext.pickle', all_sents=True)
	q, p, n, y = getIRPretrainData(gen, nques=50, npairs=900)
	# print(q.shape, p.shape, n.shape, y.shape)	
	# print(q[10])
	# print(p[10])
	# print(n[10])
	# print(y[10])