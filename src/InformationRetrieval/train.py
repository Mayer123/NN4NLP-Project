import sys 
sys.path.append('../')
import numpy as np
import torch
from torch import nn
import torch.utils.data as D
import tqdm
import os
import argparse
import logging
import pickle
from InformationRetrieval.AttentionRM.modules import AttentionRM
from InformationRetrieval.prepro.preprocess import *
from utils.utils import reset_embeddings

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

class PairwiseDataset(D.Dataset):
    """docstring for PairwiseDataloader"""
    def __init__(self, Qs, Ps, Ns, y):
        super(PairwiseDataset, self).__init__()
        self.Qs = Qs
        self.Ps = Ps
        self.Ns = Ns
        self.y = y

    def __getitem__(self, i):
        return self.Qs[i], self.Ps[i], self.Ns[i], self.y[i]

    def __len__(self):
        return len(self.Qs)

def pad(a, padlen):
    print (a)    
    pad = np.zeros((padlen, a.shape[1]))
    padded = np.concatenate((a,pad), axis=0)
    return padded

def mCollateFn(samples):
    Qs = []
    Ps = []
    Ns = []
    ys = []

    for q,p,n,y in samples:
        Qs.append(q)
        Ps.append(p)
        Ns.append(n)
        ys.append(y)

    max_q_len = max([len(q) for q in Qs])
    max_p_len = max([len(p) for p in Ps])
    max_n_len = max([len(n) for n in Ns])

    qlens = torch.tensor([len(q) for q in Qs]).long()
    plens = torch.tensor([len(p) for p in Ps]).long()
    nlens = torch.tensor([len(n) for n in Ns]).long()

    Qs = torch.tensor([np.pad(q, ((0,max_q_len-len(q)),) + ((0,0),)*(len(q.shape)-1), 'constant') for q in Qs]).long()
    Ps = torch.tensor([np.pad(p, ((0,max_p_len-len(p)),) + ((0,0),)*(len(p.shape)-1), 'constant') for p in Ps]).long()
    Ns = torch.tensor([np.pad(n, ((0,max_n_len-len(n)),) + ((0,0),)*(len(n.shape)-1), 'constant') for n in Ns]).long()
    # Qs = torch.tensor([pad(q, max_q_len-len(q)) for q in Qs]).long()
    # Ps = torch.tensor([pad(p, max_p_len-len(p)) for p in Ps]).long()
    # Ns = torch.tensor([pad(n, max_n_len-len(n)) for n in Ns]).long()
    ys = torch.tensor(ys).float()

    return Qs, Ps, Ns, ys, qlens, plens, nlens

def otherCollateFn(batch):
    Q = []
    A1 = []
    A2 = []
    Questions = []
    Passages = []
    P = []
    Pscore = []
    assert len(batch) == 1
    batch = batch[0]
    idset = []
    for i, (cid,qo,co,q,a1,a2,p,ps) in enumerate(batch):
        #for i, (cid,qw,qt,qn,qc,a1,a2,p) in enumerate(sample):
        idset.append(cid)
        Questions.append(qo)
        Q.append(q)
        A1.append(a1)
        A2.append(a2)
        Pscore.append(ps)
        if i == 0:
            Passages = co
            P = p
            
    assert len(set(idset)) == 1
    #max_q_len = max([len(q) for q in Qwords])
    #max_p_len = max([len(p) for p in Passages])
    #max_s_len = max([len(s) for s in Passages])
    #max_a_len = max([len(a) for a in A1])

    qlens = torch.tensor([len(q) for q in Q]).long()
    #plens = torch.tensor([len(p) for p in Passages]).long()
    slens = torch.tensor([len(s) for s in P]).long()   # The assumption is that passages in one batch are all the same
    #alens = torch.tensor([len(a) for a in A1]).long()
    max_q_len = torch.max(qlens)
    max_s_len = torch.max(slens)
    #max_a_len = torch.max(alens)
    Qtensor = torch.zeros(len(batch), max_q_len, 2).long()
    #Qtagtensor = torch.zeros(len(batch), max_q_len).long()
    Ptensor = torch.zeros(len(Passages), max_s_len, 2).long()
    #Ptagtensor = torch.zeros(len(Passages), max_s_len).long()
    #A1tensor = torch.zeros(len(batch), max_a_len).long() 
    #A2tensor = torch.zeros(len(batch), max_a_len).long()    
    for i in range(len(batch)):
        Qtensor[i, :qlens[i],:] = torch.tensor(Q[i])
        if i == 0:
            for j in range(len(P)):
                Ptensor[j,:slens[j],:] = torch.tensor(P[j])
    return Qtensor, Ptensor, qlens, slens, idset, Questions, Passages, Pscore, A1, A2

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
    #logger.info('Total vocab size %s pre-trained words %s' % (len(word_dict), count))
    
    return embeddings, trained_idx

def kendallTau(p_scores, n_scores):
    p_scores = p_scores.view(-1)
    n_scores = n_scores.view(-1)

    ccordantPairs = torch.sum(p_scores > n_scores).float()
    dcordantPairs = torch.sum(p_scores < n_scores).float()

    n = p_scores.shape[0]
    kt = (ccordantPairs - dcordantPairs) / n
    return kt

def train(args):
    global logger
    
    logger = logging.getLogger()
    fh = logging.FileHandler(args.log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info('-' * 100)
    logger.info('Loading data')    

    w2i = {'<pad>': 0,
            '<unk>' : 1}
    pos2i = w2i.copy()

    train_data_gen = convert_data(args.train_file, w2i, pos2i)
    Q_train, P_train, N_train, y_train = getIRPretrainData(train_data_gen)
    train_ds = PairwiseDataset(Q_train, P_train, N_train, y_train)
    train_dl = D.DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True,
                            collate_fn=mCollateFn)    

    dev_data_gen = convert_data(args.dev_file, w2i, pos2i, update_dict=False)
    Q_dev, P_dev, N_dev, y_dev = getIRPretrainData(dev_data_gen)
    dev_ds = PairwiseDataset(Q_dev, P_dev, N_dev, y_dev)
    dev_dl = D.DataLoader(dev_ds, batch_size=args.dev_batch_size, shuffle=False,
                            collate_fn=mCollateFn)

    logger.info('Generating embeddings')
    embeddings, trained_idx = generate_embeddings(args.embedding_file, w2i)
    embeddings = torch.from_numpy(embeddings).float()
    fixed_embeddings = embeddings[trained_idx]

    use_cuda = torch.cuda.is_available()

    if args.load_model != '':
        model = torch.load(args.load_model)        
    else:
        model = AttentionRM(init_emb=embeddings, pos_vocab_size=len(pos2i), use_rnn=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0008, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 
                                                            factor=0.5, patience=0,
                                                            verbose=True)
    criterion = nn.MarginRankingLoss(margin=1.0)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        model = model.cuda()
        fixed_embeddings = fixed_embeddings.cuda()

    logger.info('Start training')
    logger.info('-' * 100)
    global_step = 0
    best = float('inf')    
    Train_Loss = []
    Train_metric = []
    Dev_Loss = []
    Dev_metric = []
    for ITER in range(args.epochs):
        train_loss = 0.0
        model.train()
        with tqdm.tqdm(train_dl) as t:
            for batch in t:
                global_step += 1
                q, p, n, y, qlen, plen, nlen = batch
                if use_cuda:
                    q = q.cuda()
                    p = p.cuda()
                    n = n.cuda()
                    y = y.cuda()
                    qlen = qlen.cuda()
                    plen = plen.cuda()
                    nlen = nlen.cuda()
                o1 = model(q, p, qlen, plen)
                o2 = model(q, n, qlen, nlen)
                loss = criterion(o1, o2, y)
                kt = kendallTau(o1, o2)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                model.emb.weight.data[trained_idx] = fixed_embeddings
                Train_Loss.append(float(loss))
                Train_metric.append(float(kt))

                t.set_postfix(loss=float(np.mean(Train_Loss)), kt=np.mean(Train_metric))
        model = model.eval()
        with torch.no_grad():
            for batch in dev_dl:
                q, p, n, y, qlen, plen, nlen = batch
                if use_cuda:
                    q = q.cuda()
                    p = p.cuda()
                    n = n.cuda()
                    y = y.cuda()
                    qlen = qlen.cuda()
                    plen = plen.cuda()
                    nlen = nlen.cuda()
                o1 = model(q, p, qlen, plen)
                o2 = model(q, n, qlen, nlen)
                # print(o1)
                # print(o2)
                # print(y)
                loss = criterion(o1, o2, y)
                kt = kendallTau(o1, o2)

                Dev_metric.append(float(kt))
                Dev_Loss.append(float(loss))
            avg_dev_loss = np.mean(Dev_Loss)
            # print(Dev_metric)
            avg_dev_metric = np.mean(Dev_metric)

            scheduler.step(avg_dev_metric)

            logger.info("avg_dev_loss = %.4f avg_dev_metric = %.4f" % (avg_dev_loss, 
                                                                       avg_dev_metric))
            if avg_dev_loss < best:
                best = avg_dev_loss
                if args.save_results:
                    torch.save(model, args.model_name)

def score_sentences(args):
    global logger
    
    logger = logging.getLogger()
    fh = logging.FileHandler(args.log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info('-' * 100)
    logger.info('Loading data')    

    # w2i = {'<pad>': 0,
    #         '<unk>' : 1}
    # pos2i = w2i.copy()
    # w2i, pos2i = build_dict(args.train_file, w2i, pos2i)
    # print (len(w2i), len(pos2i))
    with open('fulltext_dict.json', 'r') as f:
        w2i = json.load(f)
    with open('fulltext_pos_dict.json', 'r') as f:
        pos2i = json.load(f)

    train = convert_data(args.train_file, w2i, pos2i, update_dict=False)    
    train = FulltextDataset(train, args.train_batch_size)
    
    dev = convert_data(args.dev_file, w2i, pos2i, update_dict=False)
    dev = FulltextDataset(dev, args.dev_batch_size)
    train_loader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=1, num_workers=4, collate_fn=otherCollateFn)
    dev_loader = torch.utils.data.DataLoader(dev, batch_size=1, num_workers=4, collate_fn=otherCollateFn)

    logger.info('Generating embeddings')
    embeddings, trained_idx = generate_embeddings(args.embedding_file, w2i)
    embeddings = torch.from_numpy(embeddings).float()
    fixed_embeddings = embeddings[trained_idx]

    use_cuda = torch.cuda.is_available()
    model = torch.load(args.load_model) 
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
        model = model.cuda()
        fixed_embeddings = fixed_embeddings.cuda()  

    model = model.eval()
    retrived_train = []
    retrived_dev = []
    # with torch.no_grad():
    #     for batch in tqdm.tqdm(train_loader):
    #         q, p, qlen, plen, ids, ques, cons, scores, a1, a2 = batch
    #         #print (q.shape, p.shape, qlen.shape, plen.shape, len(scores), len(scores[0]))
    #         if use_cuda:
    #             q = q.cuda()
    #             p = p.cuda()
    #             qlen = qlen.cuda()
    #             plen = plen.cuda()
    #         pred_score = model.forward_singleContext(q, p, qlen, plen, batch_size=1024)

    #         #print (pred_score.shape)
    #         _, topk_idx = torch.topk(pred_score, min(50, plen.shape[0]), dim=1, sorted=False)     
    #         #print (topk_idx)
    #         topk_idx = topk_idx.data
    #         for i in range(len(ques)):
    #             selected_context = []
    #             selected_scores = []
    #             for selected_idx in topk_idx[i,:]:
    #                 #print (topk_idx[i,:])
    #                 #print (topk_idx[i,:].shape)
    #                 selected_context.append(cons[selected_idx])
    #                 #print (i, selected_idx, len(scores), len(scores[i]))
    #                 selected_scores.append(scores[i][selected_idx])
    #             retrived_train.append({'id':ids[i], 'qaps':ques[i], 'context':selected_context, 'scores':selected_scores})
    # print (len(retrived_train))
    # with open('retrived_train.pickle', 'wb') as f:
    #     pickle.dump(retrived_train, f)
    with torch.no_grad():
        for batch in tqdm.tqdm(dev_loader):
            q, p, qlen, plen, ids, ques, cons, scores, a1, a2 = batch
            if use_cuda:
                q = q.cuda()
                p = p.cuda()
                qlen = qlen.cuda()
                plen = plen.cuda()
            pred_score = model.forward_singleContext(q, p, qlen, plen, batch_size=512)
            _, topk_idx = torch.topk(pred_score, min(50, plen.shape[0]), dim=1, sorted=False)     
            topk_idx = topk_idx.data
            for i in range(len(ques)):
                selected_context = []
                selected_scores = []
                #print (i, topk_idx[i,:], ids[i])
                for selected_idx in topk_idx[i,:]:
                    #print (cons[selected_idx])
                    selected_context.append(cons[selected_idx])
                    selected_scores.append(scores[i][selected_idx])
                retrived_dev.append({'id':ids[i], 'qaps':ques[i], 'context':selected_context, 'scores':selected_scores})
    print (len(retrived_dev))
    with open('retrived_dev_conv.pickle', 'wb') as f:
        pickle.dump(retrived_dev, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ConvKNRM")
    parser.add_argument("train_file", help="File that contains training data")
    parser.add_argument("dev_file", help="File that contains dev data")
    parser.add_argument("embedding_file", help="File that contains pre-trained embeddings")
    parser.add_argument('--seed', type=int, default=6, help='Random seed for the experiment')
    parser.add_argument('--epochs', type=int, default=20, help='Train data iterations')
    parser.add_argument('--train_batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--dev_batch_size', type=int, default=16, help='Batch size for dev')
    parser.add_argument('--pos_emb_size', type=int, default=50, help='Embedding size for pos tags')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate for embedding layers')    
    parser.add_argument('--log_file', type=str, default="convKNRM.log", help='path to the log file')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--model_name', type=str, default="convKNRM.pt", help='path to the log file')
    parser.add_argument('--load_model', type=str, default="", help='path to the log file')
    parser.add_argument('--mode', type=str, default="train", help='mode to run')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.mode == 'train':
        train(args)
    else:
        score_sentences(args)
